import json
from pathlib import Path

from PIL import Image

from main_service.tools.viewset_visualizer.server import (
    build_google_embed_url,
    create_app_payload,
    parse_gpano_pose_heading,
    render_view_page,
    render_view_image,
)


def test_create_app_payload_returns_image_metadata_and_python_polygons(
    tmp_path: Path,
) -> None:
    pano_path = tmp_path / "pano.jpg"
    Image.new("RGB", (1024, 512), color="white").save(pano_path)
    viewsets_dir = tmp_path / "viewsets"
    viewsets_dir.mkdir()
    (viewsets_dir / "candidate.json").write_text(
        json.dumps(
            {
                "name": "candidate",
                "views": [
                    {
                        "id": "wide-000",
                        "relative_heading": 0,
                        "pitch": 0,
                        "fov": 90,
                        "view_kind": "wide_context",
                    }
                ],
            }
        )
    )

    payload = create_app_payload(pano_path, viewsets_dir, edge_samples=9)

    assert payload["pano"]["width"] == 1024
    assert payload["pano"]["height"] == 512
    assert payload["viewsets"][0]["name"] == "candidate"
    view = payload["viewsets"][0]["views"][0]
    assert view["relative_heading"] == 0
    assert len(view["polygons"][0]) == 32


def test_render_view_image_returns_perspective_jpeg_bytes(tmp_path: Path) -> None:
    pano_path = tmp_path / "pano.jpg"
    Image.new("RGB", (1024, 512), color="white").save(pano_path)
    viewsets_dir = tmp_path / "viewsets"
    viewsets_dir.mkdir()
    (viewsets_dir / "candidate.json").write_text(
        json.dumps(
            {
                "name": "candidate",
                "views": [
                    {
                        "id": "center",
                        "relative_heading": 0,
                        "pitch": 0,
                        "fov": 60,
                        "output_width": 320,
                        "output_height": 240,
                    }
                ],
            }
        )
    )

    body, content_type = render_view_image(
        pano_path,
        viewsets_dir,
        viewset_name="candidate",
        view_id="center",
    )

    output_path = tmp_path / "view.jpg"
    output_path.write_bytes(body)
    with Image.open(output_path) as image:
        assert image.size == (320, 240)
    assert content_type == "image/jpeg"


def test_parse_gpano_pose_heading_reads_xmp_heading() -> None:
    xmp = b'GPano:PoseHeadingDegrees="313.478946685791"'

    assert parse_gpano_pose_heading(xmp) == 313.478946685791


def test_build_google_embed_url_uses_north_based_heading() -> None:
    url = build_google_embed_url(
        api_key="test-key",
        pano_id="pano-a",
        latitude=37.1,
        longitude=-122.1,
        north_offset=313.5,
        relative_heading=60,
        pitch=-10,
        fov=40,
    )

    assert "key=test-key" in url
    assert "pano=pano-a" in url
    assert "location=37.1%2C-122.1" in url
    assert "heading=13.5" in url
    assert "pitch=-10" in url
    assert "fov=40" in url


def test_render_view_page_contains_local_and_google_toggle(tmp_path: Path) -> None:
    pano_path = tmp_path / "pano-a.jpg"
    Image.new("RGB", (1024, 512), color="white").save(pano_path)
    viewsets_dir = tmp_path / "viewsets"
    viewsets_dir.mkdir()
    (viewsets_dir / "candidate.json").write_text(
        json.dumps(
            {
                "name": "candidate",
                "views": [
                    {
                        "id": "center",
                        "relative_heading": 60,
                        "pitch": 0,
                        "fov": 60,
                    }
                ],
            }
        )
    )

    html = render_view_page(
        pano_path,
        viewsets_dir,
        viewset_name="candidate",
        view_id="center",
        google_api_key="test-key",
        north_offset=313.5,
        pano_id="pano-a",
        latitude=None,
        longitude=None,
    ).decode("utf-8")

    assert "Local 2D view" in html
    assert "Google embed" in html
    assert "/api/view-image?viewset=candidate&amp;view=center" in html
    assert "heading=13.5" in html
