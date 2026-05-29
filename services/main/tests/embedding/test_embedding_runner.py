import asyncio
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sqlalchemy import create_engine

from main_service.db.models.base import Base
from main_service.db.services.panorama_service import PanoramaService
from main_service.db.services.panorama_view_embedding_service import (
    EmbeddingModelSpec,
    PanoramaViewEmbeddingService,
)
from main_service.db.services.panorama_view_service import (
    PanoramaViewService,
    PanoramaViewSpecRecord,
)
from main_service.embedding.nats_source import PanoEmbeddingJob, ReceivedPanoEmbeddingJob
from main_service.embedding.runner import EmbeddingRunResult, run_embedding_batch
from main_service.ingestion.types import PanoramaId, ProcessingStatus


def make_services() -> tuple[
    PanoramaService,
    PanoramaViewService,
    PanoramaViewEmbeddingService,
]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return (
        PanoramaService(engine),
        PanoramaViewService(engine),
        PanoramaViewEmbeddingService(engine),
    )


def make_view_spec(image_path: str, image_hash: str) -> PanoramaViewSpecRecord:
    return PanoramaViewSpecRecord(
        source_image_path=".local/panoramas/pano-a.jpg",
        source_image_hash="source-hash",
        viewset_name="candidate",
        viewset_description="test viewset",
        view_id="center",
        view_kind="center_context",
        view_spec_json={"id": "center"},
        view_spec_hash="view-spec-hash",
        relative_heading=0,
        pitch=10,
        fov=77,
        output_width=512,
        output_height=512,
        render_scale=2,
        rendered_width=1024,
        rendered_height=1024,
        output_format="jpeg",
        image_quality=95,
        interpolation_mode="bicubic",
        renderer_version="py360convert.e2p",
    )


def make_model_spec() -> EmbeddingModelSpec:
    return EmbeddingModelSpec(
        model_provider="transformers",
        model_id="google/siglip2-so400m-patch14-384",
        model_revision="main",
        preprocess_version="siglip2-384-rgb-v1",
        embedding_dimension=4,
        embedding_dtype="float16",
        embedding_normalized=True,
    )


def write_image(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(20, 40, 80)).save(path)
    return "view-image-hash"


def create_completed_view(tmp_path: Path) -> tuple[int, PanoramaViewEmbeddingService]:
    panorama_service, view_service, embedding_service = make_services()
    image_path = tmp_path / "views" / "center.jpg"
    image_hash = write_image(image_path)
    panorama_service.upsert_discovered_panorama(PanoramaId("pano-a"))
    claim = view_service.claim_view_for_processing(
        PanoramaId("pano-a"),
        make_view_spec(str(image_path), image_hash),
    )
    assert claim is not None
    completed = view_service.mark_view_complete(
        claim.id,
        image_path=str(image_path),
        image_hash=image_hash,
        image_bytes=image_path.stat().st_size,
    )
    return completed.id, embedding_service


@dataclass
class FakeReceivedEmbeddingJob:
    job: PanoEmbeddingJob
    acked: bool = False

    async def ack(self) -> None:
        self.acked = True


class FakeJobSource:
    def __init__(self, jobs: list[FakeReceivedEmbeddingJob]) -> None:
        self.jobs = jobs
        self.fetch_calls: list[int] = []

    async def fetch(self, limit: int) -> list[ReceivedPanoEmbeddingJob]:
        self.fetch_calls.append(limit)
        return self.jobs[:limit]


class FakeImageEmbedder:
    def __init__(self, vector: np.ndarray | None = None, error: Exception | None = None):
        self.vector = vector if vector is not None else np.array([1, 0, 0, 0], dtype=np.float32)
        self.error = error
        self.image_paths: list[Path] = []

    def embed_image(self, image_path: Path) -> np.ndarray:
        self.image_paths.append(image_path)
        if self.error is not None:
            raise self.error
        return self.vector


class FakeVectorStore:
    kind = "local_hnsw"
    path = ".local/embedding-indexes/test/index.bin"

    def __init__(self) -> None:
        self.added: list[dict[str, object]] = []

    def add(self, *, vector_id: int, vector: np.ndarray, metadata: dict[str, object]) -> str:
        self.added.append(
            {
                "vector_id": vector_id,
                "vector": vector,
                "metadata": metadata,
            }
        )
        return str(vector_id)


def test_embedding_runner_embeds_view_stores_vector_marks_complete_and_acks(
    tmp_path: Path,
) -> None:
    view_id, embedding_service = create_completed_view(tmp_path)
    received = FakeReceivedEmbeddingJob(
        PanoEmbeddingJob(
            pano_id=PanoramaId("pano-a"),
            view_id=view_id,
            image_path=str(tmp_path / "views" / "center.jpg"),
        )
    )
    vector_store = FakeVectorStore()
    embedder = FakeImageEmbedder()

    result = asyncio.run(
        run_embedding_batch(
            embedding_service=embedding_service,
            job_source=FakeJobSource([received]),
            image_embedder=embedder,
            vector_store=vector_store,
            model_spec=make_model_spec(),
            limit=5,
            concurrency=1,
        )
    )

    rows = embedding_service.list_embeddings_for_view(view_id)
    assert result == EmbeddingRunResult(embedded=1, skipped=0, failed=0)
    assert received.acked
    assert len(rows) == 1
    assert rows[0].embedding_status == ProcessingStatus.COMPLETE.value
    assert rows[0].vector_id == str(rows[0].id)
    assert vector_store.added[0]["vector_id"] == rows[0].id
    assert embedder.image_paths == [tmp_path / "views" / "center.jpg"]


def test_embedding_runner_skips_duplicate_completed_embedding(tmp_path: Path) -> None:
    view_id, embedding_service = create_completed_view(tmp_path)
    vector_store = FakeVectorStore()
    first = FakeReceivedEmbeddingJob(
        PanoEmbeddingJob(PanoramaId("pano-a"), view_id, str(tmp_path / "views" / "center.jpg"))
    )
    second = FakeReceivedEmbeddingJob(
        PanoEmbeddingJob(PanoramaId("pano-a"), view_id, str(tmp_path / "views" / "center.jpg"))
    )

    asyncio.run(
        run_embedding_batch(
            embedding_service=embedding_service,
            job_source=FakeJobSource([first]),
            image_embedder=FakeImageEmbedder(),
            vector_store=vector_store,
            model_spec=make_model_spec(),
            limit=5,
            concurrency=1,
        )
    )
    result = asyncio.run(
        run_embedding_batch(
            embedding_service=embedding_service,
            job_source=FakeJobSource([second]),
            image_embedder=FakeImageEmbedder(),
            vector_store=vector_store,
            model_spec=make_model_spec(),
            limit=5,
            concurrency=1,
        )
    )

    assert result == EmbeddingRunResult(embedded=0, skipped=1, failed=0)
    assert second.acked
    assert len(vector_store.added) == 1


def test_embedding_runner_marks_model_failure_and_acks(tmp_path: Path) -> None:
    view_id, embedding_service = create_completed_view(tmp_path)
    received = FakeReceivedEmbeddingJob(
        PanoEmbeddingJob(PanoramaId("pano-a"), view_id, str(tmp_path / "views" / "center.jpg"))
    )

    result = asyncio.run(
        run_embedding_batch(
            embedding_service=embedding_service,
            job_source=FakeJobSource([received]),
            image_embedder=FakeImageEmbedder(error=RuntimeError("model failed")),
            vector_store=FakeVectorStore(),
            model_spec=make_model_spec(),
            limit=5,
            concurrency=1,
        )
    )

    rows = embedding_service.list_embeddings_for_view(view_id)
    assert result == EmbeddingRunResult(embedded=0, skipped=0, failed=1)
    assert received.acked
    assert rows[0].embedding_status == ProcessingStatus.FAILED.value
    assert rows[0].last_error == "model failed"
