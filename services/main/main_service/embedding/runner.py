import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Protocol

import numpy as np

from main_service.db.services.panorama_view_embedding_service import (
    EmbeddingModelSpec,
    PanoramaViewEmbeddingService,
)
from main_service.db.models.panorama_view_embedding import PanoramaViewEmbedding
from main_service.db.services.panorama_view_service import PanoramaViewService
from main_service.embedding.nats_source import ReceivedPanoEmbeddingJob
from main_service.embedding.vector_store import VectorStore, VectorStoreRecord
from main_service.logging_config import format_log_event
from main_service.observability import NoopTelemetry
from main_service.observability.telemetry import PipelineTelemetry, observed_span

ProgressCallback = Callable[[str, dict[str, object]], None]
logger = logging.getLogger(__name__)


class PanoEmbeddingJobSource(Protocol):
    async def fetch(self, limit: int) -> list[ReceivedPanoEmbeddingJob]:
        ...


class ImageEmbedder(Protocol):
    def embed_image(self, image_path: Path) -> np.ndarray:
        ...

    def embed_images(self, image_paths: list[Path]) -> list[np.ndarray]:
        ...


@dataclass(frozen=True)
class EmbeddingRunResult:
    embedded: int
    skipped: int
    failed: int


@dataclass(frozen=True)
class ClaimedEmbeddingJob:
    received_job: ReceivedPanoEmbeddingJob
    claim: PanoramaViewEmbedding
    image_path: Path
    started_at: float


async def run_embedding_batch(
    *,
    embedding_service: PanoramaViewEmbeddingService,
    job_source: PanoEmbeddingJobSource,
    image_embedder: ImageEmbedder,
    vector_store: VectorStore,
    model_spec: EmbeddingModelSpec,
    limit: int,
    concurrency: int,
    batch_size: int = 1,
    progress: ProgressCallback | None = None,
    telemetry: PipelineTelemetry | None = None,
) -> EmbeddingRunResult:
    telemetry = telemetry or NoopTelemetry()
    _emit(progress, "embedding_batch_start", {"limit": limit, "batch_size": batch_size})
    with observed_span(
        telemetry,
        "embedding.batch",
        "embedding_batch",
        {"service": "embedding", "limit": limit, "batch_size": batch_size},
        metric_attributes={
            "service": "embedding",
            "stage": "batch",
            "batch_size": batch_size,
        },
    ):
        _emit(progress, "embedding_fetch_start", {"limit": limit})
        with observed_span(
            telemetry,
            "embedding.fetch",
            "embedding_fetch",
            {"service": "embedding", "limit": limit},
            metric_attributes={"service": "embedding", "stage": "fetch"},
        ):
            jobs = await job_source.fetch(limit)
        _emit(progress, "embedding_fetch_complete", {"jobs": len(jobs)})
        if batch_size <= 1:
            semaphore = asyncio.Semaphore(max(1, concurrency))
            statuses = await asyncio.gather(
                *[
                    _process_received_job(
                        received_job=received_job,
                        embedding_service=embedding_service,
                        image_embedder=image_embedder,
                        vector_store=vector_store,
                        model_spec=model_spec,
                        semaphore=semaphore,
                        progress=progress,
                        telemetry=telemetry,
                    )
                    for received_job in jobs
                ]
            )
        else:
            statuses = []
            for batch in _chunks(jobs, batch_size):
                statuses.extend(
                    await _process_received_job_batch(
                        received_jobs=batch,
                        embedding_service=embedding_service,
                        image_embedder=image_embedder,
                        vector_store=vector_store,
                        model_spec=model_spec,
                        progress=progress,
                        telemetry=telemetry,
                    )
                )
        result = EmbeddingRunResult(
            embedded=statuses.count("embedded"),
            skipped=statuses.count("skipped"),
            failed=statuses.count("failed"),
        )
    _emit(
        progress,
        "embedding_batch_complete",
        {**result.__dict__, "batch_size": batch_size},
    )
    return result


async def _process_received_job(
    *,
    received_job: ReceivedPanoEmbeddingJob,
    embedding_service: PanoramaViewEmbeddingService,
    image_embedder: ImageEmbedder,
    vector_store: VectorStore,
    model_spec: EmbeddingModelSpec,
    semaphore: asyncio.Semaphore,
    progress: ProgressCallback | None,
    telemetry: PipelineTelemetry,
) -> str:
    async with semaphore:
        job_payload = {
            "pano_id": received_job.job.pano_id.value,
            "view_id": received_job.job.view_id,
        }
        _emit(progress, "embedding_job_start", job_payload)
        with observed_span(
            telemetry,
            "embedding.job",
            "embedding_job",
            {"service": "embedding", **job_payload},
            metric_attributes={"service": "embedding", "stage": "job"},
        ):
            _emit(progress, "embedding_claim_start", job_payload)
            with observed_span(
                telemetry,
                "embedding.claim",
                "embedding_claim",
                {"service": "embedding", **job_payload},
                metric_attributes={"service": "embedding", "stage": "claim"},
            ):
                claim = embedding_service.claim_embedding_for_view(
                    received_job.job.view_id,
                    model_spec,
                )
            _emit(progress, "embedding_claim_complete", job_payload)
            if claim is None:
                _emit(progress, "embedding_ack_start", job_payload)
                with observed_span(
                    telemetry,
                    "embedding.ack",
                    "embedding_ack",
                    {"service": "embedding", **job_payload},
                    metric_attributes={"service": "embedding", "stage": "ack"},
                ):
                    _cleanup_temp_tile_file(
                        Path(received_job.job.image_path),
                        progress=progress,
                    )
                    await received_job.ack()
                _emit(progress, "embedding_ack_complete", job_payload)
                _emit(progress, "embedding_job_skipped", job_payload)
                return "skipped"

            try:
                image_path = Path(claim.source_image_path)
                if not image_path.exists():
                    raise FileNotFoundError(str(image_path))
                image_payload = {
                    **job_payload,
                    "embedding_id": claim.id,
                    "image_path": str(image_path),
                    "model_id": claim.model_id,
                }
                _emit(progress, "embedding_image_start", image_payload)
                _emit(progress, "embedding_model_encode_start", image_payload)
                with observed_span(
                    telemetry,
                    "embedding.model_encode",
                    "embedding_model_encode",
                    {"service": "embedding", **image_payload},
                    metric_attributes={
                        "service": "embedding",
                        "stage": "model_encode",
                        "model_id": claim.model_id,
                    },
                ):
                    vector = image_embedder.embed_image(image_path)
                _emit(progress, "embedding_model_encode_complete", image_payload)
                _emit(progress, "embedding_image_complete", image_payload)
                vector_payload = {
                    **job_payload,
                    "embedding_id": claim.id,
                    "dimension": int(vector.shape[0]),
                }
                _emit(progress, "embedding_vector_store_start", vector_payload)
                with observed_span(
                    telemetry,
                    "embedding.vector_store",
                    "embedding_vector_store",
                    {"service": "embedding", **vector_payload},
                    metric_attributes={"service": "embedding", "stage": "vector_store"},
                ):
                    vector_id = vector_store.add(
                        vector_id=claim.id,
                        vector=vector,
                        metadata={
                            "embedding_id": claim.id,
                            "view_id": claim.panorama_view_id,
                            "model_id": claim.model_id,
                            "source_image_hash": claim.source_image_hash,
                        },
                    )
                _emit(progress, "embedding_vector_store_complete", vector_payload)
                db_payload = {**job_payload, "embedding_id": claim.id, "vector_id": vector_id}
                _emit(progress, "embedding_db_update_start", db_payload)
                with observed_span(
                    telemetry,
                    "embedding.db_update",
                    "embedding_db_update",
                    {"service": "embedding", **db_payload},
                    metric_attributes={"service": "embedding", "stage": "db_update"},
                ):
                    embedding_service.mark_embedding_complete(
                        claim.id,
                        vector_store_kind=vector_store.kind,
                        vector_store_path=vector_store.path,
                        vector_id=vector_id,
                    )
                _emit(progress, "embedding_db_update_complete", db_payload)
                _cleanup_claimed_temp_tile(
                    embedding_service=embedding_service,
                    claim=claim,
                    progress=progress,
                )
                _emit(progress, "embedding_ack_start", job_payload)
                with observed_span(
                    telemetry,
                    "embedding.ack",
                    "embedding_ack",
                    {"service": "embedding", **job_payload},
                    metric_attributes={"service": "embedding", "stage": "ack"},
                ):
                    await received_job.ack()
                _emit(progress, "embedding_ack_complete", job_payload)
                _emit(
                    progress,
                    "embedding_job_complete",
                    {
                        **job_payload,
                        "embedding_id": claim.id,
                        "vector_id": vector_id,
                    },
                )
                return "embedded"
            except Exception as exc:
                embedding_service.mark_embedding_failed(claim.id, str(exc))
                _cleanup_claimed_temp_tile(
                    embedding_service=embedding_service,
                    claim=claim,
                    progress=progress,
                )
                _emit(progress, "embedding_ack_start", job_payload)
                with observed_span(
                    telemetry,
                    "embedding.ack",
                    "embedding_ack",
                    {"service": "embedding", **job_payload},
                    metric_attributes={"service": "embedding", "stage": "ack"},
                ):
                    await received_job.ack()
                _emit(progress, "embedding_ack_complete", job_payload)
                _emit(
                    progress,
                    "embedding_job_failed",
                    {
                        **job_payload,
                        "embedding_id": claim.id,
                        "error": str(exc),
                    },
                )
                return "failed"


async def _process_received_job_batch(
    *,
    received_jobs: list[ReceivedPanoEmbeddingJob],
    embedding_service: PanoramaViewEmbeddingService,
    image_embedder: ImageEmbedder,
    vector_store: VectorStore,
    model_spec: EmbeddingModelSpec,
    progress: ProgressCallback | None,
    telemetry: PipelineTelemetry,
) -> list[str]:
    claimed_jobs: list[ClaimedEmbeddingJob] = []
    statuses: list[str] = []
    actual_batch_size = len(received_jobs)
    for received_job in received_jobs:
        job_started_at = perf_counter()
        job_payload = {
            "pano_id": received_job.job.pano_id.value,
            "view_id": received_job.job.view_id,
        }
        _emit(
            progress,
            "embedding_job_start",
            job_payload,
        )
        _emit(progress, "embedding_claim_start", job_payload)
        with observed_span(
            telemetry,
            "embedding.claim",
            "embedding_claim",
            {"service": "embedding", **job_payload},
            metric_attributes={"service": "embedding", "stage": "claim"},
        ):
            claim = embedding_service.claim_embedding_for_view(
                received_job.job.view_id,
                model_spec,
            )
        _emit(progress, "embedding_claim_complete", job_payload)
        if claim is None:
            _emit(progress, "embedding_ack_start", job_payload)
            with observed_span(
                telemetry,
                "embedding.ack",
                "embedding_ack",
                {"service": "embedding", **job_payload},
                metric_attributes={"service": "embedding", "stage": "ack"},
            ):
                _cleanup_temp_tile_file(Path(received_job.job.image_path), progress=progress)
                await received_job.ack()
            _emit(progress, "embedding_ack_complete", job_payload)
            statuses.append("skipped")
            _record_batch_job_duration(
                telemetry,
                started_at=job_started_at,
                status="skipped",
                batch_size=actual_batch_size,
            )
            _emit(
                progress,
                "embedding_job_skipped",
                job_payload,
            )
            continue

        image_path = Path(claim.source_image_path)
        if not image_path.exists():
            error = str(FileNotFoundError(str(image_path)))
            embedding_service.mark_embedding_failed(claim.id, error)
            _cleanup_claimed_temp_tile(
                embedding_service=embedding_service,
                claim=claim,
                progress=progress,
            )
            _emit(progress, "embedding_ack_start", job_payload)
            with observed_span(
                telemetry,
                "embedding.ack",
                "embedding_ack",
                {"service": "embedding", **job_payload},
                metric_attributes={"service": "embedding", "stage": "ack"},
            ):
                await received_job.ack()
            _emit(progress, "embedding_ack_complete", job_payload)
            statuses.append("failed")
            _record_batch_job_duration(
                telemetry,
                started_at=job_started_at,
                status="failed",
                batch_size=actual_batch_size,
            )
            _emit(
                progress,
                "embedding_job_failed",
                {
                    **job_payload,
                    "embedding_id": claim.id,
                    "error": error,
                },
            )
            continue

        claimed_jobs.append(
            ClaimedEmbeddingJob(
                received_job=received_job,
                claim=claim,
                image_path=image_path,
                started_at=job_started_at,
            )
        )

    if not claimed_jobs:
        return statuses

    _emit(
        progress,
        "embedding_image_batch_start",
        {
            "batch_size": len(claimed_jobs),
            "model_id": model_spec.model_id,
        },
    )
    _emit(
        progress,
        "embedding_model_encode_start",
        {"batch_size": len(claimed_jobs), "model_id": model_spec.model_id},
    )
    try:
        with observed_span(
            telemetry,
            "embedding.model_encode",
            "embedding_model_encode",
            {
                "service": "embedding",
                "batch_size": len(claimed_jobs),
                "model_id": model_spec.model_id,
            },
            metric_attributes={
                "service": "embedding",
                "stage": "model_encode",
                "model_id": model_spec.model_id,
                "batch_size": len(claimed_jobs),
            },
        ):
            vectors = image_embedder.embed_images([job.image_path for job in claimed_jobs])
        if len(vectors) != len(claimed_jobs):
            raise ValueError(
                f"Embedder returned {len(vectors)} vectors for {len(claimed_jobs)} images."
            )
    except Exception as exc:
        for job in claimed_jobs:
            embedding_service.mark_embedding_failed(job.claim.id, str(exc))
            _cleanup_claimed_temp_tile(
                embedding_service=embedding_service,
                claim=job.claim,
                progress=progress,
            )
            _emit(
                progress,
                "embedding_ack_start",
                {
                    "pano_id": job.received_job.job.pano_id.value,
                    "view_id": job.claim.panorama_view_id,
                },
            )
            with observed_span(
                telemetry,
                "embedding.ack",
                "embedding_ack",
                {
                    "service": "embedding",
                    "pano_id": job.received_job.job.pano_id.value,
                    "view_id": job.claim.panorama_view_id,
                },
                metric_attributes={"service": "embedding", "stage": "ack"},
            ):
                await job.received_job.ack()
            statuses.append("failed")
            _record_batch_job_duration(
                telemetry,
                started_at=job.started_at,
                status="failed",
                batch_size=len(claimed_jobs),
            )
            _emit(
                progress,
                "embedding_job_failed",
                {
                    "pano_id": job.received_job.job.pano_id.value,
                    "embedding_id": job.claim.id,
                    "view_id": job.claim.panorama_view_id,
                    "error": str(exc),
                },
            )
        return statuses
    _emit(
        progress,
        "embedding_model_encode_complete",
        {"batch_size": len(claimed_jobs), "model_id": model_spec.model_id},
    )

    _emit(
        progress,
        "embedding_image_batch_complete",
        {
            "batch_size": len(claimed_jobs),
            "model_id": model_spec.model_id,
        },
    )
    records = [
        VectorStoreRecord(
            vector_id=job.claim.id,
            vector=vector,
            metadata={
                "embedding_id": job.claim.id,
                "view_id": job.claim.panorama_view_id,
                "model_id": job.claim.model_id,
                "source_image_hash": job.claim.source_image_hash,
            },
        )
        for job, vector in zip(claimed_jobs, vectors, strict=True)
    ]
    try:
        _emit(
            progress,
            "embedding_vector_store_batch_start",
            {
                "batch_size": len(records),
                "dimension": int(vectors[0].shape[0]),
            },
        )
        with observed_span(
            telemetry,
            "embedding.vector_store",
            "embedding_vector_store",
            {
                "service": "embedding",
                "batch_size": len(records),
                "vector_store_kind": vector_store.kind,
            },
            metric_attributes={
                "service": "embedding",
                "stage": "vector_store",
                "batch_size": len(records),
                "vector_store_kind": vector_store.kind,
            },
        ):
            vector_ids = vector_store.add_many(records)
        if len(vector_ids) != len(claimed_jobs):
            raise ValueError(
                f"Vector store returned {len(vector_ids)} IDs for {len(claimed_jobs)} vectors."
            )
    except Exception as exc:
        for job in claimed_jobs:
            embedding_service.mark_embedding_failed(job.claim.id, str(exc))
            _cleanup_claimed_temp_tile(
                embedding_service=embedding_service,
                claim=job.claim,
                progress=progress,
            )
            _emit(
                progress,
                "embedding_ack_start",
                {
                    "pano_id": job.received_job.job.pano_id.value,
                    "view_id": job.claim.panorama_view_id,
                },
            )
            with observed_span(
                telemetry,
                "embedding.ack",
                "embedding_ack",
                {
                    "service": "embedding",
                    "pano_id": job.received_job.job.pano_id.value,
                    "view_id": job.claim.panorama_view_id,
                },
                metric_attributes={"service": "embedding", "stage": "ack"},
            ):
                await job.received_job.ack()
            statuses.append("failed")
            _record_batch_job_duration(
                telemetry,
                started_at=job.started_at,
                status="failed",
                batch_size=len(claimed_jobs),
            )
            _emit(
                progress,
                "embedding_job_failed",
                {
                    "pano_id": job.received_job.job.pano_id.value,
                    "embedding_id": job.claim.id,
                    "view_id": job.claim.panorama_view_id,
                    "error": str(exc),
                },
            )
        return statuses

    _emit(
        progress,
        "embedding_vector_store_batch_complete",
        {
            "batch_size": len(records),
            "vector_store_kind": vector_store.kind,
        },
    )
    for job, vector_id in zip(claimed_jobs, vector_ids, strict=True):
        job_payload = {
            "pano_id": job.received_job.job.pano_id.value,
            "view_id": job.claim.panorama_view_id,
            "embedding_id": job.claim.id,
            "vector_id": vector_id,
        }
        _emit(progress, "embedding_db_update_start", job_payload)
        with observed_span(
            telemetry,
            "embedding.db_update",
            "embedding_db_update",
            {"service": "embedding", **job_payload},
            metric_attributes={"service": "embedding", "stage": "db_update"},
        ):
            embedding_service.mark_embedding_complete(
                job.claim.id,
                vector_store_kind=vector_store.kind,
                vector_store_path=vector_store.path,
                vector_id=vector_id,
            )
        _emit(progress, "embedding_db_update_complete", job_payload)
        _cleanup_claimed_temp_tile(
            embedding_service=embedding_service,
            claim=job.claim,
            progress=progress,
        )
        _emit(
            progress,
            "embedding_ack_start",
            {
                "pano_id": job.received_job.job.pano_id.value,
                "view_id": job.claim.panorama_view_id,
            },
        )
        with observed_span(
            telemetry,
            "embedding.ack",
            "embedding_ack",
            {
                "service": "embedding",
                "pano_id": job.received_job.job.pano_id.value,
                "view_id": job.claim.panorama_view_id,
            },
            metric_attributes={"service": "embedding", "stage": "ack"},
        ):
            await job.received_job.ack()
        _emit(
            progress,
            "embedding_ack_complete",
            {
                "pano_id": job.received_job.job.pano_id.value,
                "view_id": job.claim.panorama_view_id,
            },
        )
        statuses.append("embedded")
        _record_batch_job_duration(
            telemetry,
            started_at=job.started_at,
            status="embedded",
            batch_size=len(claimed_jobs),
        )
        _emit(
            progress,
            "embedding_job_complete",
            job_payload,
        )
    return statuses


def _record_batch_job_duration(
    telemetry: PipelineTelemetry,
    *,
    started_at: float,
    status: str,
    batch_size: int,
) -> None:
    telemetry.record_duration(
        "embedding_job",
        max(0.0, perf_counter() - started_at),
        {
            "service": "embedding",
            "stage": "job",
            "mode": "batch",
            "status": status,
            "batch_size": batch_size,
        },
    )


def _cleanup_claimed_temp_tile(
    *,
    embedding_service: PanoramaViewEmbeddingService,
    claim: PanoramaViewEmbedding,
    progress: ProgressCallback | None,
) -> None:
    _cleanup_temp_tile_file(Path(claim.source_image_path), progress=progress)
    try:
        PanoramaViewService(embedding_service.engine).clear_view_temp_image_path(
            claim.panorama_view_id,
            expected_path=claim.source_image_path,
        )
    except Exception as exc:
        _emit(
            progress,
            "embedding_temp_tile_db_clear_failed",
            {
                "embedding_id": claim.id,
                "view_id": claim.panorama_view_id,
                "image_path": claim.source_image_path,
                "error": str(exc),
            },
        )


def _cleanup_temp_tile_file(path: Path, *, progress: ProgressCallback | None) -> None:
    payload = {"image_path": str(path)}
    if not path.exists():
        _emit(progress, "embedding_temp_tile_missing", payload)
        return
    if not path.is_file():
        _emit(progress, "embedding_temp_tile_cleanup_skipped", payload)
        return
    try:
        path.unlink()
    except Exception as exc:
        _emit(
            progress,
            "embedding_temp_tile_cleanup_failed",
            {**payload, "error": str(exc)},
        )
        return
    _emit(progress, "embedding_temp_tile_deleted", payload)


def _chunks[T](items: list[T], size: int) -> list[list[T]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


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
    elif event.endswith("_skipped"):
        level = logging.WARNING
    elif event.endswith("_start") or event.endswith("_complete"):
        level = logging.INFO
    else:
        level = logging.DEBUG
    logger.log(level, "%s", format_log_event(event, payload))
