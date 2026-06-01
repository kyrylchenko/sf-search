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
from main_service.logging_config import format_log_event
from main_service.observability import NoopTelemetry
from main_service.observability.telemetry import PipelineTelemetry, observed_span

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, dict[str, object]], None]


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
    progress: ProgressCallback | None = None,
    telemetry: PipelineTelemetry | None = None,
) -> DownloadRunResult:
    telemetry = telemetry or NoopTelemetry()
    _emit(progress, "downloader_batch_start", {"limit": limit})
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
        _emit(progress, "downloader_paused", {"pause_reason": "processing_queue_backlog"})
        return DownloadRunResult(
            downloaded=0,
            skipped=0,
            failed=0,
            paused=True,
            pause_reason="processing_queue_backlog",
        )

    _emit(progress, "downloader_fetch_start", {"limit": limit})
    jobs = await job_source.fetch(limit)
    _emit(progress, "downloader_fetch_complete", {"jobs": len(jobs)})
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
                    progress=progress,
                    telemetry=telemetry,
                )
                for received_job in jobs
            ]
        )

    result = DownloadRunResult(
        downloaded=statuses.count("downloaded"),
        skipped=statuses.count("skipped"),
        failed=statuses.count("failed"),
    )
    _emit(progress, "downloader_batch_complete", result.__dict__)
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
    progress: ProgressCallback | None,
    telemetry: PipelineTelemetry,
) -> str:
    async with semaphore:
        pano_id = received_job.job.pano_id
        _emit(progress, "downloader_job_start", {"pano_id": pano_id.value})
        with observed_span(
            telemetry,
            "downloader.job",
            "downloader_job",
            {"service": "downloader", "pano_id": pano_id.value},
            metric_attributes={"service": "downloader", "stage": "job"},
        ):
            _emit(progress, "downloader_claim_start", {"pano_id": pano_id.value})
            with observed_span(
                telemetry,
                "downloader.claim",
                "downloader_claim",
                {"service": "downloader", "pano_id": pano_id.value},
                metric_attributes={"service": "downloader", "stage": "claim"},
            ):
                claimed = panorama_service.claim_panorama_for_download(pano_id)
            _emit(progress, "downloader_claim_complete", {"pano_id": pano_id.value})
            if claimed is None:
                _emit(progress, "downloader_ack_start", {"pano_id": pano_id.value})
                with observed_span(
                    telemetry,
                    "downloader.ack",
                    "downloader_ack",
                    {"service": "downloader", "pano_id": pano_id.value},
                    metric_attributes={"service": "downloader", "stage": "ack"},
                ):
                    await received_job.ack()
                _emit(progress, "downloader_ack_complete", {"pano_id": pano_id.value})
                _emit(progress, "downloader_job_skipped", {"pano_id": pano_id.value})
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
                    progress=progress,
                    telemetry=telemetry,
                )
                _emit(progress, "downloader_ack_start", {"pano_id": pano_id.value})
                with observed_span(
                    telemetry,
                    "downloader.ack",
                    "downloader_ack",
                    {"service": "downloader", "pano_id": pano_id.value},
                    metric_attributes={"service": "downloader", "stage": "ack"},
                ):
                    await received_job.ack()
                _emit(progress, "downloader_ack_complete", {"pano_id": pano_id.value})
                _emit(
                    progress,
                    "downloader_job_complete",
                    {"pano_id": pano_id.value, "image_path": str(final_path)},
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
                _emit(progress, "downloader_ack_start", {"pano_id": pano_id.value})
                with observed_span(
                    telemetry,
                    "downloader.ack",
                    "downloader_ack",
                    {"service": "downloader", "pano_id": pano_id.value},
                    metric_attributes={"service": "downloader", "stage": "ack"},
                ):
                    await received_job.ack()
                _emit(progress, "downloader_ack_complete", {"pano_id": pano_id.value})
                _emit(
                    progress,
                    "downloader_job_failed",
                    {"pano_id": pano_id.value, "error": str(exc)},
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
    progress: ProgressCallback | None,
    telemetry: PipelineTelemetry,
) -> None:
    _emit(progress, "downloader_resolve_start", {"pano_id": pano_id.value})
    with observed_span(
        telemetry,
        "downloader.resolve",
        "downloader_resolve",
        {"service": "downloader", "pano_id": pano_id.value},
        metric_attributes={"service": "downloader", "stage": "resolve"},
    ):
        resolved = await streetview_client.resolve(
            pano_id,
            session,
            latitude=claimed.latitude,
            longitude=claimed.longitude,
        )
    if resolved is None:
        raise RuntimeError(f"Panorama could not be resolved: {pano_id.value}")
    _emit(
        progress,
        "downloader_resolve_complete",
        {
            "pano_id": pano_id.value,
            "resolved_pano_id": resolved.resolved_pano_id,
        },
    )

    temp_path.parent.mkdir(parents=True, exist_ok=True)
    _emit(progress, "downloader_download_start", {"pano_id": pano_id.value, "zoom": zoom})
    with observed_span(
        telemetry,
        "downloader.download",
        "downloader_download",
        {"service": "downloader", "pano_id": pano_id.value, "zoom": zoom},
        metric_attributes={"service": "downloader", "stage": "download", "zoom": zoom},
    ):
        await streetview_client.download(resolved.panorama, temp_path, session, zoom=zoom)
        finalize_temp_file(temp_path, final_path)
        image_hash = sha256_file(final_path)
    _emit(
        progress,
        "downloader_download_complete",
        {
            "pano_id": pano_id.value,
            "image_path": str(final_path),
            "image_hash": image_hash,
        },
    )

    _emit(progress, "downloader_db_update_start", {"pano_id": pano_id.value})
    with observed_span(
        telemetry,
        "downloader.db_update",
        "downloader_db_update",
        {"service": "downloader", "pano_id": pano_id.value},
        metric_attributes={"service": "downloader", "stage": "db_update"},
    ):
        panorama_service.mark_panorama_downloaded(
            pano_id,
            image_path=str(final_path),
            image_hash=image_hash,
            metadata_json=resolved.metadata_json,
            latitude=resolved.latitude,
            longitude=resolved.longitude,
        )
    _emit(progress, "downloader_db_update_complete", {"pano_id": pano_id.value})

    _emit(
        progress,
        "downloader_enqueue_processing_start",
        {"pano_id": pano_id.value, "image_path": str(final_path)},
    )
    with observed_span(
        telemetry,
        "downloader.enqueue_processing",
        "downloader_enqueue_processing",
        {"service": "downloader", "pano_id": pano_id.value},
        metric_attributes={"service": "downloader", "stage": "enqueue_processing"},
    ):
        processing_queue.enqueue(
            PanoProcessingMessage(
                pano_id=pano_id,
                image_path=str(final_path),
            )
        )
    _emit(
        progress,
        "downloader_enqueue_processing_complete",
        {"pano_id": pano_id.value, "image_path": str(final_path)},
    )


def _emit(
    progress: ProgressCallback | None,
    event: str,
    payload: dict[str, object],
) -> None:
    _log_event(event, payload)
    if progress is not None:
        progress(event, payload)


def _log_event(event: str, payload: dict[str, object]) -> None:
    if event.endswith("_failed"):
        level = logging.ERROR
    elif event.endswith("_skipped") or event.endswith("_paused"):
        level = logging.WARNING
    elif event.endswith("_start") or event.endswith("_complete"):
        level = logging.INFO
    else:
        level = logging.DEBUG
    logger.log(level, "%s", format_log_event(event, payload))
