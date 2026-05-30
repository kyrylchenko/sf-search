import asyncio
import gc
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from PIL import Image

from main_service.db.models.panorama import Panorama
from main_service.db.services.panorama_view_service import (
    PanoramaViewService,
    PanoramaViewSpecRecord,
)
from main_service.downloader.storage import finalize_temp_file, sha256_file
from main_service.ingestion.download_queue import (
    PanoEmbeddingMessage,
    PanoEmbeddingQueue,
)
from main_service.ingestion.types import PanoramaId
from main_service.logging_config import format_log_event
from main_service.processing.nats_source import (
    PanoProcessingJob,
    ReceivedPanoProcessingJob,
)
from main_service.processing.storage import (
    panorama_view_image_path,
    temp_view_image_path,
)
from main_service.processing.view_rendering import (
    PerspectiveViewSpec,
    perspective_renderer_backend,
    render_perspective_view,
)
from main_service.tools.viewset_visualizer.geometry import ViewSpec
from main_service.tools.viewset_visualizer.viewsets import Viewset, load_viewsets

Image.MAX_IMAGE_PIXELS = None

INTERPOLATION_MODE = "bicubic"
RENDERER_VERSION = "py360convert.e2p"
ProgressCallback = Callable[[str, dict[str, object]], None]
logger = logging.getLogger(__name__)


class PanoProcessingJobSource(Protocol):
    async def fetch(self, limit: int) -> list[ReceivedPanoProcessingJob]:
        ...


@dataclass(frozen=True)
class ProcessingRunResult:
    processed_jobs: int
    failed_jobs: int
    generated_views: int
    skipped_views: int
    failed_views: int
    queued_embeddings: int = 0
    paused: bool = False
    pause_reason: str | None = None


@dataclass(frozen=True)
class ProcessingJobOutcome:
    failed_job: bool
    generated_views: int = 0
    skipped_views: int = 0
    failed_views: int = 0
    queued_embeddings: int = 0


@dataclass(frozen=True)
class ClaimedViewRender:
    claim_id: int
    pano_id: PanoramaId
    viewset_name: str
    view: ViewSpec
    spec: PanoramaViewSpecRecord
    output_path: Path


@dataclass(frozen=True)
class RenderedView:
    claim: ClaimedViewRender
    image_hash: str
    image_bytes: int


async def run_processing_batch(
    *,
    panorama_view_service: PanoramaViewService,
    job_source: PanoProcessingJobSource,
    viewsets_dir: Path,
    storage_dir: Path,
    limit: int,
    concurrency: int,
    render_scale: int,
    max_view_concurrency: int = 4,
    output_format: str = "jpeg",
    image_quality: int = 95,
    embedding_queue: PanoEmbeddingQueue | None = None,
    max_embedding_queue_depth: int | None = None,
    progress: ProgressCallback | None = None,
) -> ProcessingRunResult:
    _emit(progress, "processing_backpressure_check", {})
    if (
        embedding_queue is not None
        and max_embedding_queue_depth is not None
        and embedding_queue.pending_count() >= max_embedding_queue_depth
    ):
        _emit(
            progress,
            "processing_paused",
            {"pause_reason": "embedding_queue_backlog"},
        )
        return ProcessingRunResult(
            processed_jobs=0,
            failed_jobs=0,
            generated_views=0,
            skipped_views=0,
            failed_views=0,
            queued_embeddings=0,
            paused=True,
            pause_reason="embedding_queue_backlog",
        )

    _emit(progress, "processing_fetch_start", {"limit": limit})
    _emit(
        progress,
        "processing_renderer_backend",
        {
            "renderer_version": RENDERER_VERSION,
            "backend": perspective_renderer_backend(),
            "interpolation_mode": INTERPOLATION_MODE,
        },
    )
    jobs = await job_source.fetch(limit)
    _emit(progress, "processing_fetch_complete", {"jobs": len(jobs)})
    viewsets = load_viewsets(viewsets_dir)
    view_concurrency = _bounded_view_concurrency(
        requested=concurrency,
        maximum=max_view_concurrency,
    )
    if view_concurrency != max(1, concurrency):
        _emit(
            progress,
            "processing_view_concurrency_capped",
            {
                "requested": concurrency,
                "maximum": max(1, max_view_concurrency),
                "effective": view_concurrency,
            },
        )

    outcomes = []
    for received_job in jobs:
        outcomes.append(
            await _process_received_job(
                received_job=received_job,
                panorama_view_service=panorama_view_service,
                viewsets=viewsets,
                storage_dir=storage_dir,
                view_concurrency=view_concurrency,
                render_scale=render_scale,
                output_format=output_format,
                image_quality=image_quality,
                embedding_queue=embedding_queue,
                progress=progress,
            )
        )
    result = ProcessingRunResult(
        processed_jobs=sum(1 for outcome in outcomes if not outcome.failed_job),
        failed_jobs=sum(1 for outcome in outcomes if outcome.failed_job),
        generated_views=sum(outcome.generated_views for outcome in outcomes),
        skipped_views=sum(outcome.skipped_views for outcome in outcomes),
        failed_views=sum(outcome.failed_views for outcome in outcomes),
        queued_embeddings=sum(outcome.queued_embeddings for outcome in outcomes),
    )
    _emit(progress, "processing_batch_complete", result.__dict__)
    return result


async def _process_received_job(
    *,
    received_job: ReceivedPanoProcessingJob,
    panorama_view_service: PanoramaViewService,
    viewsets: list[Viewset],
    storage_dir: Path,
    view_concurrency: int,
    render_scale: int,
    output_format: str,
    image_quality: int,
    embedding_queue: PanoEmbeddingQueue | None,
    progress: ProgressCallback | None,
) -> ProcessingJobOutcome:
    outcome = _process_job(
        job=received_job.job,
        panorama_view_service=panorama_view_service,
        viewsets=viewsets,
        storage_dir=storage_dir,
        view_concurrency=view_concurrency,
        render_scale=render_scale,
        output_format=output_format,
        image_quality=image_quality,
        embedding_queue=embedding_queue,
        progress=progress,
    )
    await received_job.ack()
    return outcome


def _process_job(
    *,
    job: PanoProcessingJob,
    panorama_view_service: PanoramaViewService,
    viewsets: list[Viewset],
    storage_dir: Path,
    view_concurrency: int,
    render_scale: int,
    output_format: str,
    image_quality: int,
    embedding_queue: PanoEmbeddingQueue | None,
    progress: ProgressCallback | None,
) -> ProcessingJobOutcome:
    _emit(progress, "processing_job_start", {"pano_id": job.pano_id.value})
    if render_scale < 1:
        raise ValueError("render_scale must be positive")
    normalized_format = _normalize_output_format(output_format)
    panorama = panorama_view_service.get_downloaded_panorama(job.pano_id)
    if panorama is None:
        _emit(
            progress,
            "processing_job_failed",
            {"pano_id": job.pano_id.value, "reason": "panorama_not_downloaded"},
        )
        return ProcessingJobOutcome(failed_job=True)

    source_path = _resolve_source_path(job, panorama)
    if source_path is None:
        _emit(
            progress,
            "processing_job_failed",
            {"pano_id": job.pano_id.value, "reason": "source_image_missing"},
        )
        return ProcessingJobOutcome(failed_job=True)

    source_hash = (
        panorama.image_hash
        if panorama.image_path == str(source_path) and panorama.image_hash
        else sha256_file(source_path)
    )
    try:
        with Image.open(source_path) as image:
            pano_array = np.asarray(image.convert("RGB"))
    except Exception:
        _emit(
            progress,
            "processing_job_failed",
            {"pano_id": job.pano_id.value, "reason": "source_image_unreadable"},
        )
        return ProcessingJobOutcome(failed_job=True)

    skipped = 0
    claimed_views: list[ClaimedViewRender] = []
    for viewset in viewsets:
        for view in viewset.views:
            spec = _build_spec_record(
                viewset=viewset,
                view=view,
                source_image_path=str(source_path),
                source_image_hash=source_hash,
                render_scale=render_scale,
                output_format=normalized_format,
                image_quality=image_quality,
            )
            claim = panorama_view_service.claim_view_for_processing(job.pano_id, spec)
            if claim is None:
                skipped += 1
                _emit(
                    progress,
                    "processing_view_skipped",
                    {
                        "pano_id": job.pano_id.value,
                        "viewset_name": viewset.name,
                        "view_id": view.id,
                    },
                )
                continue
            output_path = panorama_view_image_path(
                storage_dir,
                pano_id=job.pano_id,
                viewset_name=viewset.name,
                view_id=view.id,
                view_spec_hash=spec.view_spec_hash,
                render_scale=render_scale,
                output_format=normalized_format,
            )
            claimed_views.append(
                ClaimedViewRender(
                    claim_id=claim.id,
                    pano_id=job.pano_id,
                    viewset_name=viewset.name,
                    view=view,
                    spec=spec,
                    output_path=output_path,
                )
            )

    generated = 0
    failed = 0
    queued_embeddings = 0
    workers = min(max(1, view_concurrency), max(1, len(claimed_views)))
    if claimed_views:
        _emit(
            progress,
            "processing_view_render_pool_start",
            {
                "pano_id": job.pano_id.value,
                "views": len(claimed_views),
                "workers": workers,
            },
        )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_claim = {
            executor.submit(
                _render_claimed_view,
                pano_array=pano_array,
                claim=claim,
                render_scale=render_scale,
                output_format=normalized_format,
                image_quality=image_quality,
                progress=progress,
            ): claim
            for claim in claimed_views
        }
        for future in as_completed(future_to_claim):
            claim = future_to_claim[future]
            try:
                rendered = future.result()
                completed = panorama_view_service.mark_view_complete(
                    rendered.claim.claim_id,
                    image_path=str(rendered.claim.output_path),
                    image_hash=rendered.image_hash,
                    image_bytes=rendered.image_bytes,
                )
                if completed.image_path and embedding_queue is not None:
                    embedding_queue.enqueue(
                        PanoEmbeddingMessage(
                            pano_id=job.pano_id,
                            view_id=completed.id,
                            image_path=completed.image_path,
                        )
                    )
                    queued_embeddings += 1
                generated += 1
                _emit(
                    progress,
                    "processing_view_complete",
                    {
                        "pano_id": job.pano_id.value,
                        "view_id": completed.id,
                        "image_path": completed.image_path or "",
                    },
                )
            except Exception as exc:
                panorama_view_service.mark_view_failed(claim.claim_id, str(exc))
                failed += 1
                _emit(
                    progress,
                    "processing_view_failed",
                    {
                        "pano_id": job.pano_id.value,
                        "viewset_name": claim.viewset_name,
                        "view_id": claim.view.id,
                        "error": str(exc),
                    },
                )
    if claimed_views:
        _emit(
            progress,
            "processing_view_render_pool_complete",
            {
                "pano_id": job.pano_id.value,
                "generated_views": generated,
                "failed_views": failed,
            },
        )
    outcome = ProcessingJobOutcome(
        failed_job=False,
        generated_views=generated,
        skipped_views=skipped,
        failed_views=failed,
        queued_embeddings=queued_embeddings,
    )
    _emit(
        progress,
        "processing_job_complete",
        {"pano_id": job.pano_id.value, **outcome.__dict__},
    )
    del pano_array
    gc.collect()
    return outcome


def _render_claimed_view(
    *,
    pano_array: np.ndarray,
    claim: ClaimedViewRender,
    render_scale: int,
    output_format: str,
    image_quality: int,
    progress: ProgressCallback | None,
) -> RenderedView:
    _emit(
        progress,
        "processing_view_start",
        {
            "pano_id": claim.pano_id.value,
            "viewset_name": claim.viewset_name,
            "view_id": claim.view.id,
        },
    )
    _render_and_store_view(
        pano_array=pano_array,
        view=claim.view,
        render_scale=render_scale,
        output_path=claim.output_path,
        output_format=output_format,
        image_quality=image_quality,
    )
    return RenderedView(
        claim=claim,
        image_hash=sha256_file(claim.output_path),
        image_bytes=claim.output_path.stat().st_size,
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
    elif event.endswith("_paused"):
        level = logging.WARNING
    elif event.endswith("_start") or event.endswith("_complete"):
        level = logging.INFO
    else:
        level = logging.DEBUG
    logger.log(level, "%s", format_log_event(event, payload))


def _resolve_source_path(job: PanoProcessingJob, panorama: Panorama) -> Path | None:
    candidates = [Path(job.image_path)]
    if panorama.image_path and panorama.image_path != job.image_path:
        candidates.append(Path(panorama.image_path))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _build_spec_record(
    *,
    viewset: Viewset,
    view: ViewSpec,
    source_image_path: str,
    source_image_hash: str,
    render_scale: int,
    output_format: str,
    image_quality: int,
) -> PanoramaViewSpecRecord:
    view_spec_json = _view_spec_json(view)
    return PanoramaViewSpecRecord(
        source_image_path=source_image_path,
        source_image_hash=source_image_hash,
        viewset_name=viewset.name,
        viewset_description=viewset.description,
        view_id=view.id,
        view_kind=view.view_kind,
        view_spec_json=view_spec_json,
        view_spec_hash=_stable_hash(view_spec_json),
        relative_heading=view.relative_heading,
        pitch=view.pitch,
        fov=view.fov,
        output_width=view.output_width,
        output_height=view.output_height,
        render_scale=render_scale,
        rendered_width=view.output_width * render_scale,
        rendered_height=view.output_height * render_scale,
        output_format=output_format,
        image_quality=image_quality if output_format == "jpeg" else None,
        interpolation_mode=INTERPOLATION_MODE,
        renderer_version=RENDERER_VERSION,
    )


def _view_spec_json(view: ViewSpec) -> dict[str, object]:
    return {
        "id": view.id,
        "relative_heading": view.relative_heading,
        "pitch": view.pitch,
        "fov": view.fov,
        "view_kind": view.view_kind,
        "output_width": view.output_width,
        "output_height": view.output_height,
    }


def _stable_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _render_and_store_view(
    *,
    pano_array: np.ndarray,
    view: ViewSpec,
    render_scale: int,
    output_path: Path,
    output_format: str,
    image_quality: int,
) -> None:
    rendered = render_perspective_view(
        pano_array,
        PerspectiveViewSpec(
            relative_heading=view.relative_heading,
            pitch=view.pitch,
            fov=view.fov,
            output_width=view.output_width * render_scale,
            output_height=view.output_height * render_scale,
        ),
    )
    temp_path = temp_view_image_path(output_path)
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rendered)
    if output_format == "jpeg":
        image.save(temp_path, format="JPEG", quality=image_quality, subsampling=0)
    elif output_format == "png":
        image.save(temp_path, format="PNG")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    finalize_temp_file(temp_path, output_path)


def _normalize_output_format(output_format: str) -> str:
    normalized = output_format.lower()
    if normalized == "jpg":
        return "jpeg"
    if normalized not in {"jpeg", "png"}:
        raise ValueError(f"Unsupported output format: {output_format}")
    return normalized


def _bounded_view_concurrency(*, requested: int, maximum: int) -> int:
    safe_maximum = max(1, maximum)
    return min(max(1, requested), safe_maximum)
