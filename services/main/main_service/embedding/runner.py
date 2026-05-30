import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from main_service.db.services.panorama_view_embedding_service import (
    EmbeddingModelSpec,
    PanoramaViewEmbeddingService,
)
from main_service.embedding.nats_source import ReceivedPanoEmbeddingJob
from main_service.embedding.vector_store import VectorStore

ProgressCallback = Callable[[str, dict[str, object]], None]
logger = logging.getLogger(__name__)


class PanoEmbeddingJobSource(Protocol):
    async def fetch(self, limit: int) -> list[ReceivedPanoEmbeddingJob]:
        ...


class ImageEmbedder(Protocol):
    def embed_image(self, image_path: Path) -> np.ndarray:
        ...


@dataclass(frozen=True)
class EmbeddingRunResult:
    embedded: int
    skipped: int
    failed: int


async def run_embedding_batch(
    *,
    embedding_service: PanoramaViewEmbeddingService,
    job_source: PanoEmbeddingJobSource,
    image_embedder: ImageEmbedder,
    vector_store: VectorStore,
    model_spec: EmbeddingModelSpec,
    limit: int,
    concurrency: int,
    progress: ProgressCallback | None = None,
) -> EmbeddingRunResult:
    _emit(progress, "embedding_fetch_start", {"limit": limit})
    jobs = await job_source.fetch(limit)
    _emit(progress, "embedding_fetch_complete", {"jobs": len(jobs)})
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
    result = EmbeddingRunResult(
        embedded=statuses.count("embedded"),
        skipped=statuses.count("skipped"),
        failed=statuses.count("failed"),
    )
    _emit(progress, "embedding_batch_complete", result.__dict__)
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
                "embedding_vector_store_start",
                {"embedding_id": claim.id, "dimension": int(vector.shape[0])},
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
                    "embedding_id": claim.id,
                    "view_id": claim.panorama_view_id,
                    "error": str(exc),
                },
            )
            return "failed"


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
    logger.log(level, "%s %s", event, json.dumps(payload, sort_keys=True))
