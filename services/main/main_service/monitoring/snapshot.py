import logging
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from main_service.db.models.map_tile import MapTile
from main_service.db.models.panorama import Panorama
from main_service.db.models.panorama_view import PanoramaView
from main_service.db.models.panorama_view_embedding import PanoramaViewEmbedding

logger = logging.getLogger(__name__)


class PendingQueue(Protocol):
    def pending_count(self) -> int:
        ...


class QdrantSnapshotSource(Protocol):
    def collection_snapshot(self) -> dict[str, object]:
        ...


@dataclass(frozen=True)
class QueueSnapshotSource:
    download: PendingQueue
    processing: PendingQueue
    embedding: PendingQueue


@dataclass
class QdrantCollectionSnapshotSource:
    url: str
    collection_name: str
    timeout_seconds: float
    client: object | None = None

    def collection_snapshot(self) -> dict[str, object]:
        info = self._get_client().get_collection(self.collection_name)
        return {
            "collection": self.collection_name,
            "status": _scalar_or_none(_read_field(info, "status")),
            "points_count": _int_or_none(_read_field(info, "points_count")),
            "vectors_count": _int_or_none(_read_field(info, "vectors_count")),
            "indexed_vectors_count": _int_or_none(
                _read_field(info, "indexed_vectors_count")
            ),
        }

    def _get_client(self) -> Any:
        if self.client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise RuntimeError("Qdrant monitoring requires qdrant-client.") from exc
            self.client = QdrantClient(
                url=self.url,
                timeout=self.timeout_seconds,
            )
        return self.client


@dataclass(frozen=True)
class PipelineSnapshot:
    status_counts: dict[str, dict[str, int]]
    queue_depths: dict[str, int | None]
    queue_errors: dict[str, str]
    coverage: dict[str, int | float | None]
    qdrant: dict[str, object] | None
    qdrant_errors: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_pipeline_snapshot(
    *,
    engine: Engine,
    queues: QueueSnapshotSource,
    qdrant: QdrantSnapshotSource | None = None,
) -> PipelineSnapshot:
    with Session(engine) as session:
        status_counts = {
            "map_tiles": _count_by_status(
                session,
                MapTile.discovery_status,
            ),
            "panoramas": _count_by_status(
                session,
                Panorama.download_status,
            ),
            "panorama_views": _count_by_status(
                session,
                PanoramaView.processing_status,
            ),
            "embeddings": _count_by_status(
                session,
                PanoramaViewEmbedding.embedding_status,
            ),
        }
        coverage = _coverage_summary(session)

    queue_depths, queue_errors = _queue_depths(queues)
    qdrant_snapshot, qdrant_errors = _qdrant_snapshot(qdrant)
    return PipelineSnapshot(
        status_counts=status_counts,
        queue_depths=queue_depths,
        queue_errors=queue_errors,
        coverage=coverage,
        qdrant=qdrant_snapshot,
        qdrant_errors=qdrant_errors,
    )


def _count_by_status(session: Session, column: object) -> dict[str, int]:
    rows = session.execute(
        select(column, func.count()).group_by(column)
    ).all()
    return {str(status): int(count) for status, count in rows}


def _coverage_summary(session: Session) -> dict[str, int | float | None]:
    row = session.execute(
        select(
            func.count(Panorama.id),
            func.count(Panorama.latitude),
            func.min(Panorama.latitude),
            func.max(Panorama.latitude),
            func.min(Panorama.longitude),
            func.max(Panorama.longitude),
        )
    ).one()
    return {
        "panos_total": int(row[0] or 0),
        "panos_with_location": int(row[1] or 0),
        "min_latitude": float(row[2]) if row[2] is not None else None,
        "max_latitude": float(row[3]) if row[3] is not None else None,
        "min_longitude": float(row[4]) if row[4] is not None else None,
        "max_longitude": float(row[5]) if row[5] is not None else None,
    }


def _queue_depths(
    queues: QueueSnapshotSource,
) -> tuple[dict[str, int | None], dict[str, str]]:
    queue_map = {
        "download": queues.download,
        "processing": queues.processing,
        "embedding": queues.embedding,
    }
    depths: dict[str, int | None] = {}
    errors: dict[str, str] = {}
    for name, queue in queue_map.items():
        try:
            depths[name] = queue.pending_count()
        except Exception as exc:
            depths[name] = None
            errors[name] = str(exc)
            logger.warning(
                "monitoring_queue_depth_failed queue=%s error=%s",
                name,
                exc,
            )
    return depths, errors


def _qdrant_snapshot(
    qdrant: QdrantSnapshotSource | None,
) -> tuple[dict[str, object] | None, dict[str, str]]:
    if qdrant is None:
        return None, {}
    try:
        return qdrant.collection_snapshot(), {}
    except Exception as exc:
        logger.warning("monitoring_qdrant_snapshot_failed error=%s", exc)
        return None, {"collection": str(exc)}


def _read_field(value: object, field: str) -> object:
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _scalar_or_none(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return enum_value
    return str(value)


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    return None
