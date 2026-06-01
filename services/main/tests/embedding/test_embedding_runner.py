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
from main_service.embedding.vector_store import VectorStoreRecord
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
        self.image_batches: list[list[Path]] = []

    def embed_image(self, image_path: Path) -> np.ndarray:
        self.image_paths.append(image_path)
        if self.error is not None:
            raise self.error
        return self.vector

    def embed_images(self, image_paths: list[Path]) -> list[np.ndarray]:
        self.image_batches.append(image_paths)
        if self.error is not None:
            raise self.error
        return [self.vector for _ in image_paths]


class FakeVectorStore:
    kind = "local_hnsw"
    path = ".local/embedding-indexes/test/index.bin"

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.added: list[dict[str, object]] = []
        self.add_many_calls: list[int] = []

    def add(self, *, vector_id: int, vector: np.ndarray, metadata: dict[str, object]) -> str:
        return self.add_many(
            [VectorStoreRecord(vector_id=vector_id, vector=vector, metadata=metadata)]
        )[0]

    def add_many(self, records: list[VectorStoreRecord]) -> list[str]:
        if self.error is not None:
            raise self.error
        self.add_many_calls.append(len(records))
        for record in records:
            self.added.append(
                {
                    "vector_id": record.vector_id,
                    "vector": record.vector,
                    "metadata": record.metadata,
                }
            )
        return [str(record.vector_id) for record in records]


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


def test_embedding_runner_reports_progress_events(tmp_path: Path) -> None:
    view_id, embedding_service = create_completed_view(tmp_path)
    received = FakeReceivedEmbeddingJob(
        PanoEmbeddingJob(PanoramaId("pano-a"), view_id, str(tmp_path / "views" / "center.jpg"))
    )
    events: list[tuple[str, dict[str, object]]] = []

    asyncio.run(
        run_embedding_batch(
            embedding_service=embedding_service,
            job_source=FakeJobSource([received]),
            image_embedder=FakeImageEmbedder(),
            vector_store=FakeVectorStore(),
            model_spec=make_model_spec(),
            limit=5,
            concurrency=1,
            progress=lambda event, payload: events.append((event, payload)),
        )
    )

    event_names = [name for name, _ in events]
    assert event_names[:2] == ["embedding_fetch_start", "embedding_fetch_complete"]
    assert "embedding_job_start" in event_names
    assert "embedding_job_complete" in event_names


def test_embedding_runner_batches_image_embedding(tmp_path: Path) -> None:
    first_view_id, embedding_service = create_completed_view(tmp_path)
    engine = embedding_service.engine
    panorama_service = PanoramaService(engine)
    view_service = PanoramaViewService(engine)
    second_image_path = tmp_path / "views" / "second.jpg"
    second_hash = write_image(second_image_path)
    panorama_service.upsert_discovered_panorama(PanoramaId("pano-b"))
    second_claim = view_service.claim_view_for_processing(
        PanoramaId("pano-b"),
        make_view_spec(str(second_image_path), second_hash),
    )
    assert second_claim is not None
    second_view = view_service.mark_view_complete(
        second_claim.id,
        image_path=str(second_image_path),
        image_hash=second_hash,
        image_bytes=second_image_path.stat().st_size,
    )
    first = FakeReceivedEmbeddingJob(
        PanoEmbeddingJob(PanoramaId("pano-a"), first_view_id, str(tmp_path / "views" / "center.jpg"))
    )
    second = FakeReceivedEmbeddingJob(
        PanoEmbeddingJob(PanoramaId("pano-b"), second_view.id, str(second_image_path))
    )
    embedder = FakeImageEmbedder()
    vector_store = FakeVectorStore()

    result = asyncio.run(
        run_embedding_batch(
            embedding_service=embedding_service,
            job_source=FakeJobSource([first, second]),
            image_embedder=embedder,
            vector_store=vector_store,
            model_spec=make_model_spec(),
            limit=5,
            concurrency=1,
            batch_size=2,
        )
    )

    assert result == EmbeddingRunResult(embedded=2, skipped=0, failed=0)
    assert first.acked
    assert second.acked
    assert embedder.image_batches == [
        [tmp_path / "views" / "center.jpg", second_image_path]
    ]
    assert len(vector_store.added) == 2
    assert vector_store.add_many_calls == [2]


def test_embedding_runner_marks_batch_failed_when_vector_store_fails(tmp_path: Path) -> None:
    first_view_id, embedding_service = create_completed_view(tmp_path)
    engine = embedding_service.engine
    panorama_service = PanoramaService(engine)
    view_service = PanoramaViewService(engine)
    second_image_path = tmp_path / "views" / "second.jpg"
    second_hash = write_image(second_image_path)
    panorama_service.upsert_discovered_panorama(PanoramaId("pano-b"))
    second_claim = view_service.claim_view_for_processing(
        PanoramaId("pano-b"),
        make_view_spec(str(second_image_path), second_hash),
    )
    assert second_claim is not None
    second_view = view_service.mark_view_complete(
        second_claim.id,
        image_path=str(second_image_path),
        image_hash=second_hash,
        image_bytes=second_image_path.stat().st_size,
    )
    first = FakeReceivedEmbeddingJob(
        PanoEmbeddingJob(PanoramaId("pano-a"), first_view_id, str(tmp_path / "views" / "center.jpg"))
    )
    second = FakeReceivedEmbeddingJob(
        PanoEmbeddingJob(PanoramaId("pano-b"), second_view.id, str(second_image_path))
    )

    result = asyncio.run(
        run_embedding_batch(
            embedding_service=embedding_service,
            job_source=FakeJobSource([first, second]),
            image_embedder=FakeImageEmbedder(),
            vector_store=FakeVectorStore(error=RuntimeError("qdrant unavailable")),
            model_spec=make_model_spec(),
            limit=5,
            concurrency=1,
            batch_size=2,
        )
    )

    first_rows = embedding_service.list_embeddings_for_view(first_view_id)
    second_rows = embedding_service.list_embeddings_for_view(second_view.id)
    assert result == EmbeddingRunResult(embedded=0, skipped=0, failed=2)
    assert first.acked
    assert second.acked
    assert first_rows[0].embedding_status == ProcessingStatus.FAILED.value
    assert second_rows[0].embedding_status == ProcessingStatus.FAILED.value
    assert first_rows[0].last_error == "qdrant unavailable"
    assert second_rows[0].last_error == "qdrant unavailable"
