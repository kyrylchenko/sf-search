import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from collections.abc import Callable
from typing import Protocol

import numpy as np

from main_service.downloader.storage import safe_storage_segment

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    kind: str
    path: str

    def add(self, *, vector_id: int, vector: np.ndarray, metadata: dict[str, object]) -> str:
        ...

    def search(self, vector: np.ndarray, limit: int) -> list[tuple[str, float]]:
        ...


@dataclass(frozen=True)
class HnswMetadata:
    dimension: int
    max_elements: int
    items: dict[str, dict[str, object]]


@dataclass
class CachedSearchIndex:
    index: object
    state: HnswMetadata
    loaded_at: float


class LocalHnswVectorStore:
    kind = "local_hnsw"

    def __init__(
        self,
        *,
        root_dir: Path,
        model_id: str,
        dimension: int,
        max_elements: int = 100_000,
        search_cache_ttl_seconds: float = 300.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.root_dir = root_dir / safe_storage_segment(model_id)
        self.dimension = dimension
        self.max_elements = max_elements
        self.index_path = self.root_dir / "index.bin"
        self.metadata_path = self.root_dir / "metadata.json"
        self.path = str(self.index_path)
        self.search_cache_ttl_seconds = search_cache_ttl_seconds
        self._clock = clock
        self._search_cache: CachedSearchIndex | None = None

    def add(self, *, vector_id: int, vector: np.ndarray, metadata: dict[str, object]) -> str:
        hnswlib = _import_hnswlib()
        normalized = _as_vector(vector, self.dimension)
        state = self._read_metadata()
        key = str(vector_id)
        if key in state.items:
            logger.info(
                "vector_store_add_skipped_existing vector_id=%s index_path=%s",
                key,
                self.index_path,
            )
            state.items[key] = metadata
            self._write_metadata(state)
            self._invalidate_search_cache()
            return key

        logger.info(
            "vector_store_add_start vector_id=%s dimension=%s index_path=%s",
            vector_id,
            self.dimension,
            self.index_path,
        )
        index = self._load_or_create_index(hnswlib, state)
        if len(state.items) >= state.max_elements:
            state = HnswMetadata(
                dimension=state.dimension,
                max_elements=state.max_elements + self.max_elements,
                items=state.items,
            )
            index.resize_index(state.max_elements)
        index.add_items(normalized.reshape(1, -1), np.array([vector_id], dtype=np.int64))
        state.items[key] = metadata
        self.root_dir.mkdir(parents=True, exist_ok=True)
        index.save_index(str(self.index_path))
        self._write_metadata(state)
        self._invalidate_search_cache()
        logger.info(
            "vector_store_add_complete vector_id=%s total_items=%s index_path=%s",
            key,
            len(state.items),
            self.index_path,
        )
        return key

    def search(self, vector: np.ndarray, limit: int) -> list[tuple[str, float]]:
        hnswlib = _import_hnswlib()
        cached = self._get_cached_search_index()
        if cached is None:
            state = self._read_metadata()
            if not self.index_path.exists() or not state.items:
                logger.warning("vector_store_search_empty index_path=%s", self.index_path)
                return []
            index = self._load_search_index(hnswlib, state)
            cached = CachedSearchIndex(
                index=index,
                state=state,
                loaded_at=self._clock(),
            )
            self._search_cache = cached
            logger.info(
                "vector_store_search_cache_loaded total_items=%s ttl_seconds=%s index_path=%s",
                len(state.items),
                self.search_cache_ttl_seconds,
                self.index_path,
            )
        else:
            state = cached.state
            index = cached.index
            logger.info(
                "vector_store_search_cache_hit total_items=%s index_path=%s",
                len(state.items),
                self.index_path,
            )

        if not state.items:
            logger.warning("vector_store_search_empty index_path=%s", self.index_path)
            return []
        normalized = _as_vector(vector, state.dimension)
        logger.info(
            "vector_store_search_start limit=%s total_items=%s index_path=%s",
            limit,
            len(state.items),
            self.index_path,
        )
        index.set_ef(max(50, limit))
        labels, distances = index.knn_query(
            normalized.reshape(1, -1),
            k=min(limit, len(state.items)),
        )
        results = [
            (str(label), float(1.0 - distance))
            for label, distance in zip(labels[0], distances[0], strict=True)
        ]
        logger.info(
            "vector_store_search_complete results=%s index_path=%s",
            len(results),
            self.index_path,
        )
        return results

    def _get_cached_search_index(self) -> CachedSearchIndex | None:
        if self._search_cache is None:
            return None
        if self.search_cache_ttl_seconds <= 0:
            self._invalidate_search_cache()
            return None
        age = self._clock() - self._search_cache.loaded_at
        if age >= self.search_cache_ttl_seconds:
            logger.info(
                "vector_store_search_cache_expired age_seconds=%.3f ttl_seconds=%s index_path=%s",
                age,
                self.search_cache_ttl_seconds,
                self.index_path,
            )
            self._invalidate_search_cache()
            return None
        return self._search_cache

    def _load_search_index(self, hnswlib: object, state: HnswMetadata) -> object:
        index = hnswlib.Index(space="cosine", dim=state.dimension)
        index.load_index(str(self.index_path), max_elements=state.max_elements)
        return index

    def _invalidate_search_cache(self) -> None:
        self._search_cache = None

    def _load_or_create_index(self, hnswlib: object, state: HnswMetadata) -> object:
        index = hnswlib.Index(space="cosine", dim=state.dimension)
        if self.index_path.exists():
            index.load_index(str(self.index_path), max_elements=state.max_elements)
        else:
            index.init_index(
                max_elements=state.max_elements,
                ef_construction=200,
                M=32,
            )
        index.set_ef(50)
        return index

    def _read_metadata(self) -> HnswMetadata:
        if not self.metadata_path.exists():
            return HnswMetadata(
                dimension=self.dimension,
                max_elements=self.max_elements,
                items={},
            )
        payload = json.loads(self.metadata_path.read_text())
        dimension = int(payload["dimension"])
        if dimension != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {dimension}"
            )
        return HnswMetadata(
            dimension=dimension,
            max_elements=int(payload["max_elements"]),
            items=dict(payload["items"]),
        )

    def _write_metadata(self, state: HnswMetadata) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(
            json.dumps(
                {
                    "dimension": state.dimension,
                    "max_elements": state.max_elements,
                    "items": state.items,
                },
                sort_keys=True,
            )
        )


def _import_hnswlib() -> object:
    try:
        import hnswlib
    except ImportError as exc:
        raise RuntimeError("Local HNSW vector store requires hnswlib.") from exc
    return hnswlib


def _as_vector(vector: np.ndarray, dimension: int) -> np.ndarray:
    normalized = np.asarray(vector, dtype=np.float32).reshape(-1)
    if normalized.shape != (dimension,):
        raise ValueError(
            f"Expected vector with dimension {dimension}, got {normalized.shape[0]}"
        )
    norm = np.linalg.norm(normalized)
    if norm > 0:
        normalized = normalized / norm
    return normalized.astype(np.float32)
