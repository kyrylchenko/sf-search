import json
from pathlib import Path

import pytest

from main_service.tools.viewset_visualizer.viewsets import (
    Viewset,
    load_viewset,
    load_viewsets,
)


def test_load_viewset_parses_views_with_defaults(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    path.write_text(
        json.dumps(
            {
                "name": "candidate",
                "description": "Test preset",
                "views": [
                    {
                        "id": "wide-000",
                        "relative_heading": 0,
                        "pitch": 0,
                        "fov": 95,
                        "view_kind": "wide_context",
                    }
                ],
            }
        )
    )

    viewset = load_viewset(path)

    assert viewset == Viewset(
        name="candidate",
        description="Test preset",
        path=path,
        views=[
            viewset.views[0],
        ],
    )
    assert viewset.views[0].output_width == 512
    assert viewset.views[0].output_height == 512


def test_load_viewset_rejects_duplicate_view_ids(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "name": "bad",
                "views": [
                    {"id": "same", "relative_heading": 0, "pitch": 0, "fov": 45},
                    {"id": "same", "relative_heading": 90, "pitch": 0, "fov": 45},
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_viewset(path)


def test_load_viewsets_loads_sorted_json_files(tmp_path: Path) -> None:
    for name in ["b", "a"]:
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "views": [
                        {"id": f"{name}-view", "relative_heading": 0, "pitch": 0, "fov": 45}
                    ],
                }
            )
        )

    assert [viewset.name for viewset in load_viewsets(tmp_path)] == ["a", "b"]
