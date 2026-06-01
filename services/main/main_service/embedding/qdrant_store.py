import logging
from typing import Any

import numpy as np

from main_service.embedding.vector_store import VectorStoreRecord, _as_vector
from main_service.logging_config import format_log_event

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    kind = "qdrant"

    def __init__(
        self,
        *,
        url: str,
        collection_name: str,
        dimension: int,
        vector_on_disk: bool,
        hnsw_on_disk: bool,
        on_disk_payload: bool,
        upsert_wait: bool,
        timeout_seconds: float = 30.0,
        client: object | None = None,
        models: object | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.collection_name = collection_name
        self.dimension = dimension
        self.vector_on_disk = vector_on_disk
        self.hnsw_on_disk = hnsw_on_disk
        self.on_disk_payload = on_disk_payload
        self.upsert_wait = upsert_wait
        self.timeout_seconds = timeout_seconds
        self.path = f"{self.url}/{self.collection_name}"
        self._client = client
        self._models = models
        self._collection_ready = False

    def add(self, *, vector_id: int, vector: np.ndarray, metadata: dict[str, object]) -> str:
        return self.add_many(
            [VectorStoreRecord(vector_id=vector_id, vector=vector, metadata=metadata)]
        )[0]

    def add_many(self, records: list[VectorStoreRecord]) -> list[str]:
        if not records:
            return []
        self._ensure_collection()
        models = self._get_models()
        points = [
            models.PointStruct(
                id=record.vector_id,
                vector=_vector_list(record.vector, self.dimension),
                payload=dict(record.metadata),
            )
            for record in records
        ]
        logger.info(
            "%s",
            format_log_event(
                "qdrant_upsert_start",
                {
                    "collection": self.collection_name,
                    "records": len(points),
                    "wait": self.upsert_wait,
                    "url": self.url,
                },
            ),
        )
        self._get_client().upsert(
            collection_name=self.collection_name,
            points=points,
            wait=self.upsert_wait,
        )
        logger.info(
            "%s",
            format_log_event(
                "qdrant_upsert_complete",
                {
                    "collection": self.collection_name,
                    "records": len(points),
                    "wait": self.upsert_wait,
                },
            ),
        )
        return [str(record.vector_id) for record in records]

    def search(self, vector: np.ndarray, limit: int) -> list[tuple[str, float]]:
        if not self._collection_exists():
            logger.warning(
                "%s",
                format_log_event(
                    "qdrant_search_empty_missing_collection",
                    {
                        "collection": self.collection_name,
                        "url": self.url,
                    },
                ),
            )
            return []
        query = _vector_list(vector, self.dimension)
        logger.info(
            "%s",
            format_log_event(
                "qdrant_search_start",
                {
                    "collection": self.collection_name,
                    "limit": limit,
                    "url": self.url,
                },
            ),
        )
        response = self._get_client().query_points(
            collection_name=self.collection_name,
            query=query,
            limit=limit,
            with_payload=False,
            with_vectors=False,
        )
        points = getattr(response, "points", response)
        results = [(str(point.id), float(point.score)) for point in points]
        logger.info(
            "%s",
            format_log_event(
                "qdrant_search_complete",
                {
                    "collection": self.collection_name,
                    "results": len(results),
                },
            ),
        )
        return results

    def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        if self._collection_exists():
            self._collection_ready = True
            return
        logger.info(
            "%s",
            format_log_event(
                "qdrant_collection_create_start",
                {
                    "collection": self.collection_name,
                    "dimension": self.dimension,
                    "url": self.url,
                },
            ),
        )
        models = self._get_models()
        try:
            self._get_client().create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE,
                    on_disk=self.vector_on_disk,
                ),
                hnsw_config=models.HnswConfigDiff(on_disk=self.hnsw_on_disk),
                on_disk_payload=self.on_disk_payload,
            )
        except Exception:
            if self._collection_exists():
                logger.info(
                    "%s",
                    format_log_event(
                        "qdrant_collection_create_race_existing",
                        {"collection": self.collection_name},
                    ),
                )
                self._collection_ready = True
                return
            raise
        self._collection_ready = True
        logger.info(
            "%s",
            format_log_event(
                "qdrant_collection_create_complete",
                {
                    "collection": self.collection_name,
                    "dimension": self.dimension,
                },
            ),
        )

    def _collection_exists(self) -> bool:
        return bool(self._get_client().collection_exists(self.collection_name))

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise RuntimeError(
                    "Qdrant vector store requires qdrant-client."
                ) from exc
            self._client = QdrantClient(
                url=self.url,
                timeout=self.timeout_seconds,
            )
        return self._client

    def _get_models(self) -> Any:
        if self._models is None:
            try:
                from qdrant_client import models
            except ImportError as exc:
                raise RuntimeError(
                    "Qdrant vector store requires qdrant-client."
                ) from exc
            self._models = models
        return self._models


def _vector_list(vector: np.ndarray, dimension: int) -> list[float]:
    return [float(value) for value in _as_vector(vector, dimension)]
