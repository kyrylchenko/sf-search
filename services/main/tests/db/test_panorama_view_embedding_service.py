from sqlalchemy import create_engine, inspect

from main_service.db.models.base import Base
from main_service.db.models.panorama_view_embedding import PanoramaViewEmbedding
from main_service.db.services.panorama_service import PanoramaService
from main_service.db.services.panorama_view_embedding_service import (
    EmbeddingModelSpec,
    PanoramaViewEmbeddingService,
)
from main_service.db.services.panorama_view_service import (
    PanoramaViewService,
    PanoramaViewSpecRecord,
)
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


def make_view_spec() -> PanoramaViewSpecRecord:
    return PanoramaViewSpecRecord(
        source_image_path=".local/panoramas/pano-a.jpg",
        source_image_hash="source-hash",
        viewset_name="candidate",
        viewset_description="test viewset",
        view_id="center",
        view_kind="center_context",
        view_spec_json={
            "id": "center",
            "relative_heading": 0,
            "pitch": 10,
            "fov": 77,
            "output_width": 512,
            "output_height": 512,
        },
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
        embedding_dimension=1152,
        embedding_dtype="float16",
        embedding_normalized=True,
    )


def create_completed_view() -> tuple[int, PanoramaViewEmbeddingService]:
    panorama_service, view_service, embedding_service = make_services()
    panorama_service.upsert_discovered_panorama(PanoramaId("pano-a"))
    claim = view_service.claim_view_for_processing(PanoramaId("pano-a"), make_view_spec())
    assert claim is not None
    completed = view_service.mark_view_complete(
        claim.id,
        image_path=".local/panorama-views/pano-a/candidate/center.jpg",
        image_hash="view-image-hash",
        image_bytes=1234,
    )
    return completed.id, embedding_service


def test_panorama_view_embedding_table_has_model_and_vector_metadata() -> None:
    columns = PanoramaViewEmbedding.__table__.c

    assert "panorama_view_embedding_table" in Base.metadata.tables
    assert "panorama_view_id" in columns
    assert "model_provider" in columns
    assert "model_id" in columns
    assert "model_revision" in columns
    assert "preprocess_version" in columns
    assert "source_image_hash" in columns
    assert "embedding_status" in columns
    assert "embedding_dimension" in columns
    assert "vector_store_kind" in columns
    assert "vector_store_path" in columns
    assert "vector_id" in columns


def test_metadata_can_create_panorama_view_embedding_sqlite_schema() -> None:
    engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(engine)

    assert inspect(engine).has_table("panorama_view_embedding_table") is True


def test_claim_embedding_for_view_creates_processing_row() -> None:
    view_id, embedding_service = create_completed_view()

    claim = embedding_service.claim_embedding_for_view(view_id, make_model_spec())

    assert claim is not None
    assert claim.panorama_view_id == view_id
    assert claim.model_provider == "transformers"
    assert claim.model_id == "google/siglip2-so400m-patch14-384"
    assert claim.preprocess_version == "siglip2-384-rgb-v1"
    assert claim.source_image_path == ".local/panorama-views/pano-a/candidate/center.jpg"
    assert claim.source_image_hash == "view-image-hash"
    assert claim.source_image_bytes == 1234
    assert claim.embedding_status == ProcessingStatus.PROCESSING.value
    assert claim.embedding_attempt_count == 1


def test_complete_embedding_blocks_duplicate_claim_for_same_model_and_source() -> None:
    view_id, embedding_service = create_completed_view()
    claim = embedding_service.claim_embedding_for_view(view_id, make_model_spec())
    assert claim is not None

    completed = embedding_service.mark_embedding_complete(
        claim.id,
        vector_store_kind="local_hnsw",
        vector_store_path=".local/embedding-indexes/siglip2/index.bin",
        vector_id="1",
    )
    duplicate = embedding_service.claim_embedding_for_view(view_id, make_model_spec())

    assert completed.embedding_status == ProcessingStatus.COMPLETE.value
    assert completed.vector_store_kind == "local_hnsw"
    assert completed.vector_id == "1"
    assert duplicate is None


def test_failed_embedding_can_be_claimed_again() -> None:
    view_id, embedding_service = create_completed_view()
    claim = embedding_service.claim_embedding_for_view(view_id, make_model_spec())
    assert claim is not None
    embedding_service.mark_embedding_failed(claim.id, "model failed")

    retry = embedding_service.claim_embedding_for_view(view_id, make_model_spec())

    assert retry is not None
    assert retry.id == claim.id
    assert retry.embedding_status == ProcessingStatus.PROCESSING.value
    assert retry.embedding_attempt_count == 2
    assert retry.last_error is None


def test_claim_embedding_rejects_unprocessed_view() -> None:
    _, view_service, embedding_service = make_services()
    claim = view_service.claim_view_for_processing(PanoramaId("pano-a"), make_view_spec())

    assert claim is None
    assert embedding_service.claim_embedding_for_view(999, make_model_spec()) is None
