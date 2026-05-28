import pytest

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
