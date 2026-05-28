import json
from pathlib import Path

from PIL import Image

from main_service.tools.viewset_visualizer.server import create_app_payload


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
