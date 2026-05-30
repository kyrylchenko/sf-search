from main_service.embedding.query_ui import (
    QueryResult,
    _extract_pano_heading,
    google_maps_street_view_url,
    render_results_page,
)


def test_render_results_page_displays_local_images_and_embedding_metadata() -> None:
    html = render_results_page(
        query="u haul truck",
        results=[
            QueryResult(
                score=0.87,
                image_path=".local/panorama-views/pano-a/candidate/center.jpg",
                pano_id="pano-a",
                viewset_name="candidate",
                view_id="center",
                relative_heading=15.0,
                pitch=10.0,
                fov=77.0,
                rendered_width=1024,
                rendered_height=1024,
                model_id="google/siglip2-so400m-patch14-384",
                vector_id="42",
                latitude=37.1234567,
                longitude=-122.7654321,
                pano_heading=100.0,
            )
        ],
    )

    assert "u haul truck" in html
    assert ".local/panorama-views/pano-a/candidate/center.jpg" in html
    assert "pano-a" in html
    assert "candidate / center" in html
    assert "score 0.8700" in html
    assert "heading 15.0" in html
    assert "google/siglip2-so400m-patch14-384" in html
    assert "Open in Google Maps" in html
    assert "map_action=pano" in html
    assert "heading=115.000000" in html


def test_google_maps_street_view_url_uses_north_based_heading() -> None:
    url = google_maps_street_view_url(
        QueryResult(
            score=0.87,
            image_path=".local/panorama-views/pano-a/candidate/center.jpg",
            pano_id="pano-a",
            viewset_name="candidate",
            view_id="center",
            relative_heading=15.0,
            pitch=10.0,
            fov=77.0,
            rendered_width=1024,
            rendered_height=1024,
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
