import numpy as np

from main_service.processing.view_rendering import PerspectiveViewSpec, render_perspective_view


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
