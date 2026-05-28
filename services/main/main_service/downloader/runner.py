import asyncio
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
    if processing_queue.pending_count() >= max_processing_queue_depth:
        return DownloadRunResult(
            downloaded=0,
            skipped=0,
            failed=0,
            paused=True,
            pause_reason="processing_queue_backlog",
        )

    jobs = await job_source.fetch(limit)
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

    return DownloadRunResult(
        downloaded=statuses.count("downloaded"),
        skipped=statuses.count("skipped"),
        failed=statuses.count("failed"),
    )


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
        claimed = panorama_service.claim_panorama_for_download(pano_id)
        if claimed is None:
            await received_job.ack()
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
    resolved = await streetview_client.resolve(
        pano_id,
        session,
        latitude=claimed.latitude,
        longitude=claimed.longitude,
    )
    if resolved is None:
        raise RuntimeError(f"Panorama could not be resolved: {pano_id.value}")

    temp_path.parent.mkdir(parents=True, exist_ok=True)
    await streetview_client.download(resolved.panorama, temp_path, session, zoom=zoom)
    finalize_temp_file(temp_path, final_path)
    image_hash = sha256_file(final_path)

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
