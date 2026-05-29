import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from main_service.db.services.panorama_view_embedding_service import (
    EmbeddingModelSpec,
    PanoramaViewEmbeddingService,
)
from main_service.embedding.nats_source import ReceivedPanoEmbeddingJob
from main_service.embedding.vector_store import VectorStore


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
) -> EmbeddingRunResult:
    jobs = await job_source.fetch(limit)
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
            )
            for received_job in jobs
        ]
    )
    return EmbeddingRunResult(
        embedded=statuses.count("embedded"),
        skipped=statuses.count("skipped"),
        failed=statuses.count("failed"),
    )


async def _process_received_job(
    *,
    received_job: ReceivedPanoEmbeddingJob,
    embedding_service: PanoramaViewEmbeddingService,
    image_embedder: ImageEmbedder,
    vector_store: VectorStore,
    model_spec: EmbeddingModelSpec,
    semaphore: asyncio.Semaphore,
) -> str:
    async with semaphore:
        claim = embedding_service.claim_embedding_for_view(
            received_job.job.view_id,
            model_spec,
        )
        if claim is None:
            await received_job.ack()
            return "skipped"

        try:
            image_path = Path(claim.source_image_path)
            if not image_path.exists():
                raise FileNotFoundError(str(image_path))
            vector = image_embedder.embed_image(image_path)
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
            return "embedded"
        except Exception as exc:
            embedding_service.mark_embedding_failed(claim.id, str(exc))
            await received_job.ack()
            return "failed"
