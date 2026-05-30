import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncContextManager, Protocol

from main_service.db.models.panorama import Panorama
from main_service.db.services.panorama_service import PanoramaService
from main_service.downloader.storage import (
    finalize_temp_file,
    pano_image_path,
    sha256_file,
    temp_pano_image_path,
)
from main_service.downloader.streetview_client import (
    RealStreetViewClient,
    StreetViewClient,
    create_streetview_session,
)
from main_service.ingestion.download_queue import PanoProcessingMessage, PanoProcessingQueue
from main_service.ingestion.types import PanoramaId

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PanoDownloadJob:
    pano_id: PanoramaId


class ReceivedPanoDownloadJob(Protocol):
    job: PanoDownloadJob

    async def ack(self) -> None:
        ...


class PanoDownloadJobSource(Protocol):
    async def fetch(self, limit: int) -> list[ReceivedPanoDownloadJob]:
        ...


@dataclass(frozen=True)
class DownloadRunResult:
    downloaded: int
    skipped: int
    failed: int
    paused: bool = False
    pause_reason: str | None = None


SessionFactory = Callable[[], AsyncContextManager[object]]


async def run_downloader_batch(
    *,
    panorama_service: PanoramaService,
    job_source: PanoDownloadJobSource,
    processing_queue: PanoProcessingQueue,
    storage_dir: Path,
    limit: int,
    concurrency: int,
    max_processing_queue_depth: int,
    streetview_client: StreetViewClient | None = None,
    max_attempts: int = 3,
    zoom: int = 5,
    session_factory: SessionFactory = create_streetview_session,
) -> DownloadRunResult:
    pending = processing_queue.pending_count()
    logger.debug(
        "downloader_backpressure_check %s",
        json.dumps(
            {
                "processing_queue_pending": pending,
                "max_processing_queue_depth": max_processing_queue_depth,
            },
            sort_keys=True,
        ),
    )
    if pending >= max_processing_queue_depth:
        logger.warning(
            "downloader_paused %s",
            json.dumps({"pause_reason": "processing_queue_backlog"}, sort_keys=True),
        )
        return DownloadRunResult(
            downloaded=0,
            skipped=0,
            failed=0,
            paused=True,
            pause_reason="processing_queue_backlog",
        )

    logger.info("downloader_fetch_start %s", json.dumps({"limit": limit}, sort_keys=True))
    jobs = await job_source.fetch(limit)
    logger.info(
        "downloader_fetch_complete %s",
        json.dumps({"jobs": len(jobs)}, sort_keys=True),
    )
    client = streetview_client or RealStreetViewClient()
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async with session_factory() as session:
        statuses = await asyncio.gather(
            *[
                _process_download_job(
                    received_job=received_job,
                    panorama_service=panorama_service,
                    processing_queue=processing_queue,
                    streetview_client=client,
                    storage_dir=storage_dir,
                    session=session,
                    semaphore=semaphore,
                    max_attempts=max_attempts,
                    zoom=zoom,
                )
                for received_job in jobs
            ]
        )

    result = DownloadRunResult(
        downloaded=statuses.count("downloaded"),
        skipped=statuses.count("skipped"),
        failed=statuses.count("failed"),
    )
    logger.info("downloader_batch_complete %s", json.dumps(result.__dict__, sort_keys=True))
    return result


async def _process_download_job(
    *,
    received_job: ReceivedPanoDownloadJob,
    panorama_service: PanoramaService,
    processing_queue: PanoProcessingQueue,
    streetview_client: StreetViewClient,
    storage_dir: Path,
    session: object,
    semaphore: asyncio.Semaphore,
    max_attempts: int,
    zoom: int,
) -> str:
    async with semaphore:
        pano_id = received_job.job.pano_id
        logger.info(
            "downloader_job_start %s",
            json.dumps({"pano_id": pano_id.value}, sort_keys=True),
        )
        claimed = panorama_service.claim_panorama_for_download(pano_id)
        if claimed is None:
            await received_job.ack()
            logger.warning(
                "downloader_job_skipped %s",
                json.dumps({"pano_id": pano_id.value}, sort_keys=True),
            )
            return "skipped"

        final_path = pano_image_path(storage_dir, pano_id)
        temp_path = temp_pano_image_path(final_path)
        try:
            await _download_claimed_panorama(
                claimed=claimed,
                pano_id=pano_id,
                final_path=final_path,
                temp_path=temp_path,
                panorama_service=panorama_service,
                processing_queue=processing_queue,
                streetview_client=streetview_client,
                session=session,
                zoom=zoom,
            )
            await received_job.ack()
            logger.info(
                "downloader_job_complete %s",
                json.dumps(
                    {"pano_id": pano_id.value, "image_path": str(final_path)},
                    sort_keys=True,
                ),
            )
            return "downloaded"
        except Exception as exc:
            if temp_path.exists():
                temp_path.unlink()
            panorama_service.mark_panorama_download_failed(
                pano_id,
                error=str(exc),
                max_attempts=max_attempts,
            )
            await received_job.ack()
            logger.error(
                "downloader_job_failed %s",
                json.dumps({"pano_id": pano_id.value, "error": str(exc)}, sort_keys=True),
            )
            return "failed"


async def _download_claimed_panorama(
    *,
    claimed: Panorama,
    pano_id: PanoramaId,
    final_path: Path,
    temp_path: Path,
    panorama_service: PanoramaService,
    processing_queue: PanoProcessingQueue,
    streetview_client: StreetViewClient,
    session: object,
    zoom: int,
) -> None:
    logger.info(
        "downloader_resolve_start %s",
        json.dumps({"pano_id": pano_id.value}, sort_keys=True),
    )
    resolved = await streetview_client.resolve(
        pano_id,
        session,
        latitude=claimed.latitude,
        longitude=claimed.longitude,
    )
    if resolved is None:
        raise RuntimeError(f"Panorama could not be resolved: {pano_id.value}")
    logger.info(
        "downloader_resolve_complete %s",
        json.dumps(
            {
                "pano_id": pano_id.value,
                "resolved_pano_id": resolved.resolved_pano_id,
            },
            sort_keys=True,
        ),
    )

    temp_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "downloader_download_start %s",
        json.dumps({"pano_id": pano_id.value, "zoom": zoom}, sort_keys=True),
    )
    await streetview_client.download(resolved.panorama, temp_path, session, zoom=zoom)
    finalize_temp_file(temp_path, final_path)
    image_hash = sha256_file(final_path)
    logger.info(
        "downloader_download_complete %s",
        json.dumps(
            {
                "pano_id": pano_id.value,
                "image_path": str(final_path),
                "image_hash": image_hash,
            },
            sort_keys=True,
        ),
    )

    panorama_service.mark_panorama_downloaded(
        pano_id,
        image_path=str(final_path),
        image_hash=image_hash,
        metadata_json=resolved.metadata_json,
        latitude=resolved.latitude,
        longitude=resolved.longitude,
    )
    processing_queue.enqueue(
        PanoProcessingMessage(
            pano_id=pano_id,
            image_path=str(final_path),
        )
    )
    logger.info(
        "downloader_processing_enqueued %s",
        json.dumps({"pano_id": pano_id.value, "image_path": str(final_path)}, sort_keys=True),
    )
