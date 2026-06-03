from main_service.embedding.query_ui import (
    QueryResult,
    TileRenderer,
    _extract_pano_heading,
    build_coverage_payload,
    build_search_payload,
    google_maps_street_view_url,
    render_results_page,
)
from main_service.db.models.base import Base
from main_service.db.models.panorama import Panorama
from main_service.db.models.panorama_view import PanoramaView
from main_service.db.models.panorama_view_embedding import PanoramaViewEmbedding
from main_service.ingestion.types import DownloadStatus, ProcessingStatus
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


def make_result(**overrides: object) -> QueryResult:
    values = {
        "score": 0.87,
        "view_db_id": 7,
        "pano_image_path": ".local/panoramas/pano-a.jpg",
        "pano_id": "pano-a",
        "viewset_name": "candidate",
        "view_id": "center",
        "relative_heading": 15.0,
        "pitch": 10.0,
        "fov": 77.0,
        "rendered_width": 1024,
        "rendered_height": 1024,
        "output_format": "jpeg",
        "image_quality": 95,
        "model_id": "google/siglip2-so400m-patch14-384",
        "vector_id": "42",
        "latitude": 37.1234567,
        "longitude": -122.7654321,
        "pano_heading": 100.0,
    }
    values.update(overrides)
    return QueryResult(**values)  # type: ignore[arg-type]


def test_render_results_page_uses_api_and_infinite_scroll_shell() -> None:
    html = render_results_page(
        query="u haul truck",
        limit=50,
    )

    assert "u haul truck" in html
    assert "/api/search" in html
    assert "IntersectionObserver" in html
    assert "data-query=\"u haul truck\"" in html
    assert "data-limit=\"50\"" in html
    assert ".local/panorama-views" not in html


def test_render_results_page_includes_loading_indicators() -> None:
    html = render_results_page(query="orange car", limit=50)

    assert 'id="topLoader"' in html
    assert "@keyframes tilePulse" in html
    assert "tileSkeleton" in html
    assert "article.classList.remove(\"is-loading\")" in html
    assert "setLoading(true" in html


def test_render_results_page_includes_coverage_map_tab() -> None:
    html = render_results_page(query="", limit=50, active_tab="coverage")

    assert 'href="/coverage"' in html
    assert 'id="coverageMap"' in html
    assert "https://unpkg.com/leaflet" in html
    assert "/api/coverage" in html
    assert "embedded_pano_count" in html
    assert "Embedded panos" in html


def test_build_search_payload_uses_view_tile_urls_and_next_offset() -> None:
    payload = build_search_payload(
        query="u haul truck",
        results=[make_result()],
        offset=50,
        limit=50,
        has_more=True,
    )

    assert payload["query"] == "u haul truck"
    assert payload["offset"] == 50
    assert payload["limit"] == 50
    assert payload["next_offset"] == 100
    assert payload["has_more"] is True
    assert payload["results"][0]["tile_url"] == "/tile?view_id=7"
    assert payload["results"][0]["pano_id"] == "pano-a"
    assert payload["results"][0]["maps_url"].startswith("https://www.google.com/maps/@?")


def test_build_coverage_payload_counts_total_and_embedded_panos_by_cell() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        p1 = _add_panorama(session, "pano-a", 37.1002, -122.1002, DownloadStatus.DOWNLOADED.value)
        p2 = _add_panorama(session, "pano-b", 37.1005, -122.1004, DownloadStatus.DOWNLOADED.value)
        _add_panorama(session, "pano-c", 37.2000, -122.2000, DownloadStatus.PENDING.value)
        _add_completed_embedding(session, p1)
        _add_completed_embedding(session, p2)
        session.commit()

    payload = build_coverage_payload(engine, cell_degrees=0.01)

    assert payload["stats"] == {
        "total_panos": 3,
        "panos_with_coordinates": 3,
        "downloaded_panos": 2,
        "rendered_views": 2,
        "complete_embeddings": 2,
        "embedded_panos": 2,
    }
    assert payload["bounds"] == {
        "south": 37.1002,
        "west": -122.2,
        "north": 37.2,
        "east": -122.1002,
    }
    cells = payload["cells"]
    assert len(cells) == 2
    hot_cell = max(cells, key=lambda cell: cell["embedded_pano_count"])
    assert hot_cell["pano_count"] == 2
    assert hot_cell["embedded_pano_count"] == 2
    assert hot_cell["downloaded_pano_count"] == 2
    assert hot_cell["bounds"]["south"] == 37.1
    assert hot_cell["bounds"]["west"] == -122.11


def test_google_maps_street_view_url_uses_north_based_heading() -> None:
    url = google_maps_street_view_url(
        QueryResult(
            score=0.87,
            view_db_id=7,
            pano_image_path=".local/panoramas/pano-a.jpg",
            pano_id="pano-a",
            viewset_name="candidate",
            view_id="center",
            relative_heading=15.0,
            pitch=10.0,
            fov=77.0,
            rendered_width=1024,
            rendered_height=1024,
            output_format="jpeg",
            image_quality=95,
            model_id="google/siglip2-so400m-patch14-384",
            vector_id="42",
            latitude=37.1234567,
            longitude=-122.7654321,
            pano_heading=100.0,
        )
    )

    assert url.startswith("https://www.google.com/maps/@?")
    assert "api=1" in url
    assert "map_action=pano" in url
    assert "pano=pano-a" in url
    assert "viewpoint=37.1234567%2C-122.7654321" in url
    assert "heading=115.000000" in url
    assert "pitch=10.000000" in url
    assert "fov=77.000000" in url


def _add_panorama(
    session: Session,
    orig_id: str,
    latitude: float,
    longitude: float,
    download_status: str,
) -> Panorama:
    panorama = Panorama(
        orig_id=orig_id,
        image_hash=None,
        latitude=latitude,
        longitude=longitude,
        download_status=download_status,
        metadata_status=ProcessingStatus.COMPLETE.value,
        image_path=f".local/panoramas/{orig_id}.jpg",
    )
    session.add(panorama)
    session.flush()
    return panorama


def _add_completed_embedding(session: Session, panorama: Panorama) -> None:
    view = PanoramaView(
        panorama_id=panorama.id,
        viewset_name="small-object-grid-72",
        viewset_description="test",
        view_id=f"small-{panorama.id:03d}",
        view_kind="small_object",
        view_spec_json={"id": f"small-{panorama.id:03d}"},
        view_spec_hash=f"view-hash-{panorama.id}",
        relative_heading=0.0,
        pitch=0.0,
        fov=40.0,
        output_width=512,
        output_height=512,
        render_scale=2,
        rendered_width=1024,
        rendered_height=1024,
        output_format="jpeg",
        image_quality=95,
        interpolation_mode="bicubic",
        renderer_version="py360convert.e2p",
        source_image_path=panorama.image_path or "",
        source_image_hash=f"source-hash-{panorama.id}",
        image_path=None,
        image_hash=f"view-image-hash-{panorama.id}",
        image_bytes=1024,
        processing_status=ProcessingStatus.COMPLETE.value,
        embedding_status=ProcessingStatus.COMPLETE.value,
    )
    session.add(view)
    session.flush()
    session.add(
        PanoramaViewEmbedding(
            panorama_view_id=view.id,
            model_provider="transformers",
            model_id="google/siglip2-so400m-patch14-384",
            model_revision="main",
            preprocess_version="siglip2-384-rgb-v1",
            source_image_path="",
            source_image_hash=view.image_hash or "",
            source_image_bytes=view.image_bytes,
            embedding_status=ProcessingStatus.COMPLETE.value,
            embedding_dimension=1152,
            embedding_dtype="float16",
            embedding_normalized=True,
            vector_store_kind="qdrant",
            vector_store_path="panorama_view_embeddings_siglip2",
            vector_id=str(view.id),
        )
    )


def test_extract_pano_heading_converts_streetlevel_radians_to_degrees() -> None:
    assert _extract_pano_heading({"heading": 1.5707963267948966}) == 90.0


def test_extract_pano_heading_keeps_explicit_degree_metadata() -> None:
    assert _extract_pano_heading({"heading_degrees": 100.0, "heading": 1.0}) == 100.0


def test_tile_renderer_renders_view_from_panorama(tmp_path) -> None:
    pano_path = tmp_path / "pano-a.jpg"
    Image.new("RGB", (64, 32), color=(20, 40, 80)).save(pano_path)
    renderer = TileRenderer(max_cached_panos=2)

    data, content_type = renderer.render_tile(
        make_result(
            pano_image_path=str(pano_path),
            rendered_width=32,
            rendered_height=24,
            output_format="jpeg",
            image_quality=90,
        )
    )

    assert content_type == "image/jpeg"
    assert data.startswith(b"\xff\xd8")
