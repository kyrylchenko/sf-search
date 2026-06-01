from pathlib import Path

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
from main_service.ingestion.download_queue import (
    InMemoryPanoEmbeddingQueue,
    InMemoryPanoProcessingQueue,
)
from main_service.ingestion.types import DownloadStatus, PanoramaId
from main_service.ops.requeue import (
    requeue_embedding_jobs_from_db,
    requeue_processing_jobs_from_db,
)


def make_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


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
        embedding_dtype="float32",
        embedding_normalized=True,
    )


def test_requeue_processing_jobs_from_db_publishes_downloaded_unprocessed_panos(
    tmp_path: Path,
) -> None:
    engine = make_engine()
    panorama_service = PanoramaService(engine)
    downloaded_path = tmp_path / "pano-a.jpg"
    downloaded_path.write_bytes(b"fake")
    pano = panorama_service.upsert_discovered_panorama(PanoramaId("pano-a"))
    panorama_service.mark_panorama_download_status(pano.id, DownloadStatus.DOWNLOADING)
    panorama_service.mark_panorama_downloaded(
        PanoramaId("pano-a"),
        image_path=str(downloaded_path),
        image_hash="hash-a",
        metadata_json={},
        latitude=None,
        longitude=None,
    )
    queue = InMemoryPanoProcessingQueue()

    count = requeue_processing_jobs_from_db(
        engine=engine,
        processing_queue=queue,
        limit=10,
    )

    assert count == 1
    assert [message.pano_id.value for message in queue.messages] == ["pano-a"]
    assert queue.messages[0].image_path == str(downloaded_path)


def test_requeue_embedding_jobs_from_db_skips_already_completed_embeddings(
    tmp_path: Path,
) -> None:
    engine = make_engine()
    panorama_service = PanoramaService(engine)
    view_service = PanoramaViewService(engine)
    embedding_service = PanoramaViewEmbeddingService(engine)
    image_path = tmp_path / "view-a.jpg"
    image_path.write_bytes(b"fake")
    panorama_service.upsert_discovered_panorama(PanoramaId("pano-a"))
    claim = view_service.claim_view_for_processing(
        PanoramaId("pano-a"),
        make_view_spec(str(image_path), "view-hash"),
    )
    assert claim is not None
    completed_view = view_service.mark_view_complete(
        claim.id,
        image_path=str(image_path),
        image_hash="view-hash",
        image_bytes=image_path.stat().st_size,
    )
    model_spec = make_model_spec()
    embedding_claim = embedding_service.claim_embedding_for_view(
        completed_view.id,
        model_spec,
    )
    assert embedding_claim is not None
    embedding_service.mark_embedding_complete(
        embedding_claim.id,
        vector_store_kind="local_hnsw",
        vector_store_path=".local/index.bin",
        vector_id=str(embedding_claim.id),
    )
    queue = InMemoryPanoEmbeddingQueue()

    count = requeue_embedding_jobs_from_db(
        engine=engine,
        embedding_queue=queue,
        model_spec=model_spec,
        limit=10,
    )

    assert count == 0
    assert queue.messages == []


def test_requeue_embedding_jobs_from_db_publishes_missing_embeddings(
    tmp_path: Path,
) -> None:
    engine = make_engine()
    panorama_service = PanoramaService(engine)
    view_service = PanoramaViewService(engine)
    image_path = tmp_path / "view-a.jpg"
    image_path.write_bytes(b"fake")
    panorama_service.upsert_discovered_panorama(PanoramaId("pano-a"))
    claim = view_service.claim_view_for_processing(
        PanoramaId("pano-a"),
        make_view_spec(str(image_path), "view-hash"),
    )
    assert claim is not None
    completed_view = view_service.mark_view_complete(
        claim.id,
        image_path=str(image_path),
        image_hash="view-hash",
        image_bytes=image_path.stat().st_size,
    )
    queue = InMemoryPanoEmbeddingQueue()

    count = requeue_embedding_jobs_from_db(
        engine=engine,
        embedding_queue=queue,
        model_spec=make_model_spec(),
        limit=10,
    )

    assert count == 1
    assert queue.messages[0].pano_id.value == "pano-a"
    assert queue.messages[0].view_id == completed_view.id
    assert queue.messages[0].image_path == str(image_path)
