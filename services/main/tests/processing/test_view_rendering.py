import numpy as np

from main_service.processing import view_rendering
from main_service.processing.view_rendering import (
    PerspectiveViewSpec,
    perspective_renderer_backend,
    render_perspective_view,
    vertical_fov_degrees,
)


def test_render_perspective_view_uses_requested_output_dimensions() -> None:
    pano = np.zeros((512, 1024, 3), dtype=np.uint8)
    spec = PerspectiveViewSpec(
        relative_heading=0,
        pitch=0,
        fov=60,
        output_width=320,
        output_height=240,
    )

    rendered = render_perspective_view(pano, spec)

    assert rendered.shape == (240, 320, 3)


def test_render_perspective_view_uses_bicubic_interpolation_by_default(
    monkeypatch,
) -> None:
    pano = np.zeros((512, 1024, 3), dtype=np.uint8)
    spec = PerspectiveViewSpec(
        relative_heading=0,
        pitch=0,
        fov=60,
        output_width=320,
        output_height=240,
    )
    captured: dict[str, object] = {}

    def fake_e2p(
        pano,
        fov,
        heading,
        pitch,
        output_shape,
        *,
        mode="bilinear",
    ):
        captured["mode"] = mode
        return np.zeros((output_shape[0], output_shape[1], 3), dtype=np.uint8)

    monkeypatch.setattr(view_rendering, "e2p", fake_e2p)

    render_perspective_view(pano, spec)

    assert captured["mode"] == "bicubic"


def test_vertical_fov_degrees_accounts_for_output_aspect_ratio() -> None:
    assert vertical_fov_degrees(
        horizontal_fov_degrees=90,
        output_width=512,
        output_height=512,
    ) == 90
    assert vertical_fov_degrees(
        horizontal_fov_degrees=90,
        output_width=768,
        output_height=512,
    ) < 90


def test_perspective_renderer_backend_reports_available_sampler(monkeypatch) -> None:
    monkeypatch.setattr(view_rendering.py360_utils, "cv2", object())
    assert perspective_renderer_backend() == "opencv"

    monkeypatch.setattr(view_rendering.py360_utils, "cv2", None)
    assert perspective_renderer_backend() == "scipy"
