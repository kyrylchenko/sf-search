import json
from pathlib import Path

from PIL import Image

from main_service.tools.viewset_visualizer.server import (
    build_google_embed_url,
    create_app_payload,
    parse_gpano_pose_heading,
    resolve_pano_gallery,
    resolve_pano_paths,
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


def test_create_app_payload_can_select_from_pano_gallery(tmp_path: Path) -> None:
    pano_a = tmp_path / "a.jpg"
    pano_b = tmp_path / "b.jpg"
    Image.new("RGB", (1024, 512), color="white").save(pano_a)
    Image.new("RGB", (2048, 1024), color="white").save(pano_b)
    viewsets_dir = tmp_path / "viewsets"
    viewsets_dir.mkdir()
    (viewsets_dir / "candidate.json").write_text(
        json.dumps(
            {
                "name": "candidate",
                "views": [
                    {"id": "center", "relative_heading": 0, "pitch": 0, "fov": 60}
                ],
            }
        )
    )

    payload = create_app_payload(
        pano_a,
        viewsets_dir,
        pano_paths=[pano_a, pano_b],
        pano_index=1,
    )

    assert payload["pano"]["filename"] == "b.jpg"
    assert payload["pano"]["width"] == 2048
    assert payload["pano"]["height"] == 1024
    assert payload["pano"]["index"] == 1
    assert payload["pano"]["count"] == 2
    assert payload["pano"]["previous_index"] == 0
    assert payload["pano"]["next_index"] == 0
    assert payload["pano"]["url"] == "/pano?pano=1"


def test_render_view_image_returns_perspective_png_bytes(tmp_path: Path) -> None:
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

    output_path = tmp_path / "view.png"
    output_path.write_bytes(body)
    with Image.open(output_path) as image:
        assert image.size == (320, 240)
    assert content_type == "image/png"


def test_render_view_image_can_scale_output_for_preview(tmp_path: Path) -> None:
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

    body, _ = render_view_image(
        pano_path,
        viewsets_dir,
        viewset_name="candidate",
        view_id="center",
        render_scale=2,
    )

    output_path = tmp_path / "view.png"
    output_path.write_bytes(body)
    with Image.open(output_path) as image:
        assert image.size == (640, 480)


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

    assert "High-res preview" in html
    assert "Processing tile" not in html
    assert "Google embed" in html
    assert "/api/view-image?viewset=candidate&amp;view=center&amp;scale=4" in html
    assert "heading=13.5" in html


def test_render_view_page_pins_local_image_to_pano_index(tmp_path: Path) -> None:
    pano_path = tmp_path / "pano-b.jpg"
    Image.new("RGB", (1024, 512), color="white").save(pano_path)
    viewsets_dir = tmp_path / "viewsets"
    viewsets_dir.mkdir()
    (viewsets_dir / "candidate.json").write_text(
        json.dumps(
            {
                "name": "candidate",
                "views": [
                    {"id": "center", "relative_heading": 0, "pitch": 0, "fov": 60}
                ],
            }
        )
    )

    html = render_view_page(
        pano_path,
        viewsets_dir,
        viewset_name="candidate",
        view_id="center",
        google_api_key=None,
        north_offset=None,
        pano_id=None,
        latitude=None,
        longitude=None,
        pano_index=3,
    ).decode("utf-8")

    assert "/api/view-image?viewset=candidate&amp;view=center&amp;pano=3&amp;scale=4" in html


def test_render_view_page_fits_preview_image_inside_viewport(tmp_path: Path) -> None:
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
                        "relative_heading": 0,
                        "pitch": 0,
                        "fov": 60,
                        "output_width": 512,
                        "output_height": 512,
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
        google_api_key=None,
        north_offset=None,
        pano_id=None,
        latitude=None,
        longitude=None,
    ).decode("utf-8")

    assert ".stage { height: calc(100vh - 58px); overflow: hidden;" in html
    assert (
        "width: 100%; height: 100%; min-width: 0; min-height: 0; "
        "max-width: 100vw; max-height: calc(100vh - 58px); object-fit: contain;"
    ) in html
    assert ".viewer.hidden, .hidden { display: none;" in html


def test_resolve_pano_paths_returns_sorted_supported_images(tmp_path: Path) -> None:
    (tmp_path / "b.jpg").write_bytes(b"fake")
    (tmp_path / "a.webp").write_bytes(b"fake")
    (tmp_path / "ignore.txt").write_text("nope")

    assert [path.name for path in resolve_pano_paths(tmp_path)] == ["a.webp", "b.jpg"]


def test_resolve_pano_gallery_for_file_uses_siblings_and_initial_index(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.jpg").write_bytes(b"fake")
    selected = tmp_path / "b.jpg"
    selected.write_bytes(b"fake")
    (tmp_path / "ignore.txt").write_text("nope")

    gallery = resolve_pano_gallery(selected)

    assert [path.name for path in gallery.paths] == ["a.jpg", "b.jpg"]
    assert gallery.initial_index == 1
