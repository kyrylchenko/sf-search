import logging
from typing import Protocol

from main_service.db.services.panorama_service import PanoramaService
from main_service.ingestion.download_queue import PanoDownloadMessage

logger = logging.getLogger(__name__)


class PanoDownloadQueueWriter(Protocol):
    def enqueue(self, message: PanoDownloadMessage) -> None:
        ...


def requeue_download_jobs_from_db(
    *,
    panorama_service: PanoramaService,
    download_queue: PanoDownloadQueueWriter,
    limit: int,
) -> int:
    candidates = panorama_service.list_download_queue_candidates(limit)
    for candidate in candidates:
        download_queue.enqueue(
            PanoDownloadMessage(
                pano_id=candidate.pano_id,
                source_tile=candidate.source_tile,
            )
        )
        panorama_service.mark_panorama_download_queued(candidate.panorama_id)
    if candidates:
        logger.info("📤 downloader_requeued_from_db count=%s", len(candidates))
    return len(candidates)
