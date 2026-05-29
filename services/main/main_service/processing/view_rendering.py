from dataclasses import dataclass
from math import atan, degrees, radians, tan

import numpy as np
from py360convert import e2p


@dataclass(frozen=True)
class PerspectiveViewSpec:
    relative_heading: float
    pitch: float
    fov: float
    output_width: int
    output_height: int


def render_perspective_view(
    pano: np.ndarray,
    spec: PerspectiveViewSpec,
) -> np.ndarray:
    return e2p(
        pano,
        (
            spec.fov,
            vertical_fov_degrees(
                horizontal_fov_degrees=spec.fov,
                output_width=spec.output_width,
                output_height=spec.output_height,
            ),
        ),
        _py360_heading(spec.relative_heading),
        spec.pitch,
        (spec.output_height, spec.output_width),
        mode="bicubic",
    )


def _py360_heading(relative_heading: float) -> float:
    heading = relative_heading % 360
    return heading if heading <= 180 else heading - 360


def vertical_fov_degrees(
    *,
    horizontal_fov_degrees: float,
    output_width: int,
    output_height: int,
) -> float:
    aspect_height_over_width = output_height / output_width
    horizontal_fov = radians(horizontal_fov_degrees)
    return degrees(2 * atan(tan(horizontal_fov / 2) * aspect_height_over_width))
