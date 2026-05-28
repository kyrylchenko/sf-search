from dataclasses import dataclass

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
        spec.fov,
        _py360_heading(spec.relative_heading),
        spec.pitch,
        (spec.output_height, spec.output_width),
    )


def _py360_heading(relative_heading: float) -> float:
    heading = relative_heading % 360
    return heading if heading <= 180 else heading - 360
