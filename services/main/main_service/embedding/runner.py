import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from main_service.db.services.panorama_view_embedding_service import (
    EmbeddingModelSpec,
    PanoramaViewEmbeddingService,
)
from main_service.db.models.panorama_view_embedding import PanoramaViewEmbedding
from main_service.embedding.nats_source import ReceivedPanoEmbeddingJob
from main_service.embedding.vector_store import VectorStore, VectorStoreRecord
from main_service.logging_config import format_log_event

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
) -> EmbeddingRunResult:
    _emit(progress, "embedding_batch_start", {"limit": limit, "batch_size": batch_size})
    _emit(progress, "embedding_fetch_start", {"limit": limit})
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
) -> str:
    async with semaphore:
        _emit(
            progress,
            "embedding_job_start",
            {
                "pano_id": received_job.job.pano_id.value,
                "view_id": received_job.job.view_id,
            },
        )
        claim = embedding_service.claim_embedding_for_view(
            received_job.job.view_id,
            model_spec,
        )
        if claim is None:
            await received_job.ack()
            _emit(
                progress,
                "embedding_job_skipped",
                {
                    "pano_id": received_job.job.pano_id.value,
                    "view_id": received_job.job.view_id,
                },
            )
            return "skipped"

        try:
            image_path = Path(claim.source_image_path)
            if not image_path.exists():
                raise FileNotFoundError(str(image_path))
            _emit(
                progress,
                "embedding_image_start",
                {
                    "embedding_id": claim.id,
                    "view_id": claim.panorama_view_id,
                    "image_path": str(image_path),
                    "model_id": claim.model_id,
                },
            )
            vector = image_embedder.embed_image(image_path)
            _emit(
                progress,
                "embedding_image_complete",
                {
                    "embedding_id": claim.id,
                    "view_id": claim.panorama_view_id,
                    "image_path": str(image_path),
                    "model_id": claim.model_id,
                },
            )
            _emit(
                progress,
                "embedding_vector_store_start",
                {
                    "embedding_id": claim.id,
                    "view_id": claim.panorama_view_id,
                    "dimension": int(vector.shape[0]),
                },
            )
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
            _emit(
                progress,
                "embedding_vector_store_complete",
                {
                    "embedding_id": claim.id,
                    "view_id": claim.panorama_view_id,
                    "dimension": int(vector.shape[0]),
                },
            )
            embedding_service.mark_embedding_complete(
                claim.id,
                vector_store_kind=vector_store.kind,
                vector_store_path=vector_store.path,
                vector_id=vector_id,
            )
            await received_job.ack()
            _emit(
                progress,
                "embedding_job_complete",
                {
                    "pano_id": received_job.job.pano_id.value,
                    "embedding_id": claim.id,
                    "view_id": claim.panorama_view_id,
                    "vector_id": vector_id,
                },
            )
            return "embedded"
        except Exception as exc:
            embedding_service.mark_embedding_failed(claim.id, str(exc))
            await received_job.ack()
            _emit(
                progress,
                "embedding_job_failed",
                {
                    "pano_id": received_job.job.pano_id.value,
                    "embedding_id": claim.id,
                    "view_id": claim.panorama_view_id,
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
) -> list[str]:
    claimed_jobs: list[ClaimedEmbeddingJob] = []
    statuses: list[str] = []
    for received_job in received_jobs:
        _emit(
            progress,
            "embedding_job_start",
            {
                "pano_id": received_job.job.pano_id.value,
                "view_id": received_job.job.view_id,
            },
        )
        claim = embedding_service.claim_embedding_for_view(
            received_job.job.view_id,
            model_spec,
        )
        if claim is None:
            await received_job.ack()
            statuses.append("skipped")
            _emit(
                progress,
                "embedding_job_skipped",
                {
                    "pano_id": received_job.job.pano_id.value,
                    "view_id": received_job.job.view_id,
                },
            )
            continue

        image_path = Path(claim.source_image_path)
        if not image_path.exists():
            error = str(FileNotFoundError(str(image_path)))
            embedding_service.mark_embedding_failed(claim.id, error)
            await received_job.ack()
            statuses.append("failed")
            _emit(
                progress,
                "embedding_job_failed",
                {
                    "embedding_id": claim.id,
                    "view_id": claim.panorama_view_id,
                    "error": error,
                },
            )
            continue

        claimed_jobs.append(
            ClaimedEmbeddingJob(
                received_job=received_job,
                claim=claim,
                image_path=image_path,
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
    try:
        vectors = image_embedder.embed_images([job.image_path for job in claimed_jobs])
        if len(vectors) != len(claimed_jobs):
            raise ValueError(
                f"Embedder returned {len(vectors)} vectors for {len(claimed_jobs)} images."
            )
    except Exception as exc:
        for job in claimed_jobs:
            embedding_service.mark_embedding_failed(job.claim.id, str(exc))
            await job.received_job.ack()
            statuses.append("failed")
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
        vector_ids = vector_store.add_many(records)
        if len(vector_ids) != len(claimed_jobs):
            raise ValueError(
                f"Vector store returned {len(vector_ids)} IDs for {len(claimed_jobs)} vectors."
            )
    except Exception as exc:
        for job in claimed_jobs:
            embedding_service.mark_embedding_failed(job.claim.id, str(exc))
            await job.received_job.ack()
            statuses.append("failed")
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
        embedding_service.mark_embedding_complete(
            job.claim.id,
            vector_store_kind=vector_store.kind,
            vector_store_path=vector_store.path,
            vector_id=vector_id,
        )
        await job.received_job.ack()
        statuses.append("embedded")
        _emit(
            progress,
            "embedding_job_complete",
            {
                "pano_id": job.received_job.job.pano_id.value,
                "embedding_id": job.claim.id,
                "view_id": job.claim.panorama_view_id,
                "vector_id": vector_id,
            },
        )
    return statuses


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
