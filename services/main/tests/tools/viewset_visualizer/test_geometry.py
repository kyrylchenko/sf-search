import pytest
import numpy as np

from main_service.processing.view_rendering import PerspectiveViewSpec, render_perspective_view
from main_service.tools.viewset_visualizer.geometry import (
    ViewSpec,
    normalize_heading,
    overlay_polygons_for_view,
    validate_google_compatible_view,
)


def test_normalize_heading_wraps_to_zero_360() -> None:
    assert normalize_heading(0) == 0
    assert normalize_heading(360) == 0
    assert normalize_heading(-90) == 270
    assert normalize_heading(450) == 90


def test_validate_google_compatible_view_rejects_invalid_embed_values() -> None:
    with pytest.raises(ValueError, match="pitch"):
        validate_google_compatible_view(
            ViewSpec(id="bad-pitch", relative_heading=0, pitch=91, fov=45)
        )

    with pytest.raises(ValueError, match="fov"):
        validate_google_compatible_view(
            ViewSpec(id="bad-fov", relative_heading=0, pitch=0, fov=101)
        )


def test_overlay_polygons_are_sampled_curved_boundaries_not_four_point_boxes() -> None:
    polygons = overlay_polygons_for_view(
        ViewSpec(
            id="small-center",
            relative_heading=90,
            pitch=0,
            fov=45,
            output_width=512,
            output_height=512,
        ),
        edge_samples=17,
    )

    assert len(polygons) == 1
    assert len(polygons[0]) == 64
    assert len({round(point["y"], 6) for point in polygons[0][:17]}) > 2


def test_overlay_heading_matches_processing_renderer_heading() -> None:
    pano_width = 360
    pano_height = 180
    x_values = np.linspace(0, 1, pano_width, dtype=np.float32)
    pano = np.tile(x_values, (pano_height, 1))
    pano = np.stack([pano, pano, pano], axis=2)

    rendered = render_perspective_view(
        pano,
        PerspectiveViewSpec(
            relative_heading=60,
            pitch=0,
            fov=40,
            output_width=31,
            output_height=31,
        ),
    )
    rendered_center_x = float(rendered[15, 15, 0])
    polygons = overlay_polygons_for_view(
        ViewSpec(id="wide", relative_heading=60, pitch=0, fov=40),
        edge_samples=9,
    )
    points = polygons[0]
    overlay_center_x = sum(point["x"] for point in points) / len(points)

    assert overlay_center_x == pytest.approx(rendered_center_x, abs=0.02)


def test_center_heading_view_does_not_cross_pano_seam() -> None:
    polygons = overlay_polygons_for_view(
        ViewSpec(
            id="center",
            relative_heading=0,
            pitch=0,
            fov=95,
            output_width=512,
            output_height=512,
        ),
        edge_samples=9,
    )

    assert len(polygons) == 1
    assert all(0 <= point["x"] <= 1 for point in polygons[0])


def test_overlay_polygons_duplicate_shifted_copy_when_crossing_pano_seam() -> None:
    polygons = overlay_polygons_for_view(
        ViewSpec(
            id="seam-crossing",
            relative_heading=180,
            pitch=0,
            fov=45,
            output_width=512,
            output_height=512,
        ),
        edge_samples=9,
    )

    assert len(polygons) == 2
    assert any(point["x"] < 0 for point in polygons[0])
    assert any(point["x"] > 1 for point in polygons[1])
