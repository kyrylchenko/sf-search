import logging
from dataclasses import asdict, dataclass
from typing import Protocol

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


@dataclass(frozen=True)
class QueueSnapshotSource:
    download: PendingQueue
    processing: PendingQueue
    embedding: PendingQueue


@dataclass(frozen=True)
class PipelineSnapshot:
    status_counts: dict[str, dict[str, int]]
    queue_depths: dict[str, int | None]
    queue_errors: dict[str, str]
    coverage: dict[str, int | float | None]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_pipeline_snapshot(
    *,
    engine: Engine,
    queues: QueueSnapshotSource,
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
    return PipelineSnapshot(
        status_counts=status_counts,
        queue_depths=queue_depths,
        queue_errors=queue_errors,
        coverage=coverage,
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
