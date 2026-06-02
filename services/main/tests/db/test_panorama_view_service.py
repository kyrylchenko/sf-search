from sqlalchemy import create_engine, inspect

from main_service.db.models.base import Base
from main_service.db.models.panorama_view import PanoramaView
from main_service.db.services.panorama_service import PanoramaService
from main_service.db.services.panorama_view_service import (
    PanoramaViewService,
    PanoramaViewSpecRecord,
)
from main_service.ingestion.types import PanoramaId, ProcessingStatus


def make_services() -> tuple[PanoramaService, PanoramaViewService]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return PanoramaService(engine), PanoramaViewService(engine)


def make_spec() -> PanoramaViewSpecRecord:
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


def test_panorama_view_table_has_processing_and_embedding_metadata() -> None:
    columns = PanoramaView.__table__.c

    assert "panorama_view_table" in Base.metadata.tables
    assert "panorama_id" in columns
    assert "viewset_name" in columns
    assert "view_id" in columns
    assert "view_spec_json" in columns
    assert "view_spec_hash" in columns
    assert "source_image_path" in columns
    assert "source_image_hash" in columns
    assert "image_path" in columns
    assert "image_hash" in columns
    assert "processing_status" in columns
    assert "embedding_status" in columns


def test_metadata_can_create_panorama_view_sqlite_schema() -> None:
    engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(engine)

    assert inspect(engine).has_table("panorama_view_table") is True


def test_claim_view_for_processing_creates_processing_row() -> None:
    panorama_service, view_service = make_services()
    pano = panorama_service.upsert_discovered_panorama(PanoramaId("pano-a"))
    spec = make_spec()

    claim = view_service.claim_view_for_processing(PanoramaId("pano-a"), spec)

    assert claim is not None
    assert claim.panorama_id == pano.id
    assert claim.viewset_name == "candidate"
    assert claim.view_id == "center"
    assert claim.processing_status == ProcessingStatus.PROCESSING.value
    assert claim.embedding_status == ProcessingStatus.PENDING.value
    assert claim.view_spec_json == spec.view_spec_json


def test_complete_view_blocks_duplicate_claim_for_same_source_and_spec() -> None:
    panorama_service, view_service = make_services()
    panorama_service.upsert_discovered_panorama(PanoramaId("pano-a"))
    spec = make_spec()
    claim = view_service.claim_view_for_processing(PanoramaId("pano-a"), spec)
    assert claim is not None

    completed = view_service.mark_view_complete(
        claim.id,
        image_path=".local/panorama-views/pano-a/candidate/center.jpg",
        image_hash="generated-hash",
        image_bytes=1234,
    )
    duplicate = view_service.claim_view_for_processing(PanoramaId("pano-a"), spec)

    assert completed.processing_status == ProcessingStatus.COMPLETE.value
    assert completed.image_path == ".local/panorama-views/pano-a/candidate/center.jpg"
    assert completed.image_hash == "generated-hash"
    assert duplicate is None


def test_complete_view_without_temp_image_path_blocks_duplicate_claim() -> None:
    panorama_service, view_service = make_services()
    panorama_service.upsert_discovered_panorama(PanoramaId("pano-a"))
    spec = make_spec()
    claim = view_service.claim_view_for_processing(PanoramaId("pano-a"), spec)
    assert claim is not None

    completed = view_service.mark_view_complete(
        claim.id,
        image_path=".local/panorama-view-tmp/pano-a/candidate/center.jpg",
        image_hash="generated-hash",
        image_bytes=1234,
    )
    cleared = view_service.clear_view_temp_image_path(
        completed.id,
        expected_path=".local/panorama-view-tmp/pano-a/candidate/center.jpg",
    )
    duplicate = view_service.claim_view_for_processing(PanoramaId("pano-a"), spec)

    assert cleared.processing_status == ProcessingStatus.COMPLETE.value
    assert cleared.image_path is None
    assert cleared.image_hash == "generated-hash"
    assert duplicate is None


def test_clear_view_temp_image_path_keeps_unexpected_path() -> None:
    panorama_service, view_service = make_services()
    panorama_service.upsert_discovered_panorama(PanoramaId("pano-a"))
    spec = make_spec()
    claim = view_service.claim_view_for_processing(PanoramaId("pano-a"), spec)
    assert claim is not None
    completed = view_service.mark_view_complete(
        claim.id,
        image_path=".local/panorama-view-tmp/pano-a/candidate/center.jpg",
        image_hash="generated-hash",
        image_bytes=1234,
    )

    cleared = view_service.clear_view_temp_image_path(
        completed.id,
        expected_path=".local/panorama-view-tmp/other.jpg",
    )

    assert cleared.image_path == ".local/panorama-view-tmp/pano-a/candidate/center.jpg"
    assert cleared.image_hash == "generated-hash"
    assert cleared.image_bytes == 1234


def test_failed_view_can_be_claimed_again() -> None:
    panorama_service, view_service = make_services()
    panorama_service.upsert_discovered_panorama(PanoramaId("pano-a"))
    spec = make_spec()
    claim = view_service.claim_view_for_processing(PanoramaId("pano-a"), spec)
    assert claim is not None
    view_service.mark_view_failed(claim.id, "render failed")

    retry = view_service.claim_view_for_processing(PanoramaId("pano-a"), spec)

    assert retry is not None
    assert retry.id == claim.id
    assert retry.processing_status == ProcessingStatus.PROCESSING.value
    assert retry.attempt_count == 2
    assert retry.last_error is None
