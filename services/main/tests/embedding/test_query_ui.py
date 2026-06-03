from main_service.embedding.query_ui import (
    QueryResult,
    TileRenderer,
    _extract_pano_heading,
    _tile_output_size,
    build_search_payload,
    google_maps_street_view_url,
    render_results_page,
)
from PIL import Image


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
    assert "tileUrlForResult" in html
    assert "w: String(width)" in html


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
    assert payload["results"][0]["view_db_id"] == 7
    assert payload["results"][0]["tile_url"] == "/tile?view_id=7"
    assert payload["results"][0]["pano_id"] == "pano-a"
    assert payload["results"][0]["maps_url"].startswith("https://www.google.com/maps/@?")



def test_tile_output_size_defaults_to_one_third() -> None:
    assert _tile_output_size(make_result(rendered_width=1024, rendered_height=768)) == (341, 256)


def test_tile_output_size_caps_requested_size_and_preserves_aspect() -> None:
    assert _tile_output_size(
        make_result(rendered_width=1024, rendered_height=768),
        requested_width=500,
        requested_height=200,
    ) == (267, 200)

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
