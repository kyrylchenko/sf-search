import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from main_service.db.models.panorama import Panorama
from main_service.db.services.panorama_view_service import (
    PanoramaViewService,
    PanoramaViewSpecRecord,
)
from main_service.downloader.storage import finalize_temp_file, sha256_file
from main_service.ingestion.types import PanoramaId
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
    render_perspective_view,
)
from main_service.tools.viewset_visualizer.geometry import ViewSpec
from main_service.tools.viewset_visualizer.viewsets import Viewset, load_viewsets

Image.MAX_IMAGE_PIXELS = None

INTERPOLATION_MODE = "bicubic"
RENDERER_VERSION = "py360convert.e2p"


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


@dataclass(frozen=True)
class ProcessingJobOutcome:
    failed_job: bool
    generated_views: int = 0
    skipped_views: int = 0
    failed_views: int = 0


async def run_processing_batch(
    *,
    panorama_view_service: PanoramaViewService,
    job_source: PanoProcessingJobSource,
    viewsets_dir: Path,
    storage_dir: Path,
    limit: int,
    concurrency: int,
    render_scale: int,
    output_format: str = "jpeg",
    image_quality: int = 95,
) -> ProcessingRunResult:
    jobs = await job_source.fetch(limit)
    viewsets = load_viewsets(viewsets_dir)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    outcomes = await asyncio.gather(
        *[
            _process_received_job(
                received_job=received_job,
                panorama_view_service=panorama_view_service,
                viewsets=viewsets,
                storage_dir=storage_dir,
                semaphore=semaphore,
                render_scale=render_scale,
                output_format=output_format,
                image_quality=image_quality,
            )
            for received_job in jobs
        ]
    )
    return ProcessingRunResult(
        processed_jobs=sum(1 for outcome in outcomes if not outcome.failed_job),
        failed_jobs=sum(1 for outcome in outcomes if outcome.failed_job),
        generated_views=sum(outcome.generated_views for outcome in outcomes),
        skipped_views=sum(outcome.skipped_views for outcome in outcomes),
        failed_views=sum(outcome.failed_views for outcome in outcomes),
    )


async def _process_received_job(
    *,
    received_job: ReceivedPanoProcessingJob,
    panorama_view_service: PanoramaViewService,
    viewsets: list[Viewset],
    storage_dir: Path,
    semaphore: asyncio.Semaphore,
    render_scale: int,
    output_format: str,
    image_quality: int,
) -> ProcessingJobOutcome:
    async with semaphore:
        outcome = _process_job(
            job=received_job.job,
            panorama_view_service=panorama_view_service,
            viewsets=viewsets,
            storage_dir=storage_dir,
            render_scale=render_scale,
            output_format=output_format,
            image_quality=image_quality,
        )
        await received_job.ack()
        return outcome


def _process_job(
    *,
    job: PanoProcessingJob,
    panorama_view_service: PanoramaViewService,
    viewsets: list[Viewset],
    storage_dir: Path,
    render_scale: int,
    output_format: str,
    image_quality: int,
) -> ProcessingJobOutcome:
    if render_scale < 1:
        raise ValueError("render_scale must be positive")
    normalized_format = _normalize_output_format(output_format)
    panorama = panorama_view_service.get_downloaded_panorama(job.pano_id)
    if panorama is None:
        return ProcessingJobOutcome(failed_job=True)

    source_path = _resolve_source_path(job, panorama)
    if source_path is None:
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
        return ProcessingJobOutcome(failed_job=True)

    generated = 0
    skipped = 0
    failed = 0
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
                continue
            try:
                output_path = panorama_view_image_path(
                    storage_dir,
                    pano_id=job.pano_id,
                    viewset_name=viewset.name,
                    view_id=view.id,
                    view_spec_hash=spec.view_spec_hash,
                    render_scale=render_scale,
                    output_format=normalized_format,
                )
                _render_and_store_view(
                    pano_array=pano_array,
                    view=view,
                    render_scale=render_scale,
                    output_path=output_path,
                    output_format=normalized_format,
                    image_quality=image_quality,
                )
                panorama_view_service.mark_view_complete(
                    claim.id,
                    image_path=str(output_path),
                    image_hash=sha256_file(output_path),
                    image_bytes=output_path.stat().st_size,
                )
                generated += 1
            except Exception as exc:
                panorama_view_service.mark_view_failed(claim.id, str(exc))
                failed += 1
    return ProcessingJobOutcome(
        failed_job=False,
        generated_views=generated,
        skipped_views=skipped,
        failed_views=failed,
    )


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
