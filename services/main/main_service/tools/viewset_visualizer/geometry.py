from dataclasses import dataclass
from math import atan, degrees, radians, tan

import numpy as np
from py360convert import utils as py360_utils

Point = dict[str, float]
Polygon = list[Point]


@dataclass(frozen=True)
class ViewSpec:
    id: str
    relative_heading: float
    pitch: float
    fov: float
    view_kind: str = "custom"
    output_width: int = 512
    output_height: int = 512


def normalize_heading(heading: float) -> float:
    return heading % 360


def validate_google_compatible_view(view: ViewSpec) -> None:
    if not -90 <= view.pitch <= 90:
        raise ValueError(f"pitch must be in [-90, 90], got {view.pitch}")
    if not 10 <= view.fov <= 100:
        raise ValueError(f"fov must be in [10, 100], got {view.fov}")
    if view.output_width <= 0 or view.output_height <= 0:
        raise ValueError("output dimensions must be positive")


def overlay_polygons_for_view(
    view: ViewSpec,
    *,
    edge_samples: int = 49,
) -> list[Polygon]:
    validate_google_compatible_view(view)
    if edge_samples < 2:
        raise ValueError("edge_samples must be at least 2")

    xyz = _sample_frustum_edge_xyz(view, edge_samples)
    u, v = py360_utils.xyz2uv(xyz)
    coor_x, coor_y = py360_utils.uv2coor(u, v, h=1, w=1)
    x_values = np.asarray(coor_x, dtype=np.float64).reshape(-1) + 0.5
    y_values = np.asarray(coor_y, dtype=np.float64).reshape(-1) + 0.5
    x_values = _unwrap_x_values(x_values, center=normalize_heading(view.relative_heading) / 360)

    polygon = [
        {"x": float(x), "y": float(np.clip(y, 0, 1))}
        for x, y in zip(x_values, y_values)
    ]
    return _duplicate_for_canvas_clipping(polygon)


def view_to_api_dict(view: ViewSpec, *, edge_samples: int = 49) -> dict[str, object]:
    return {
        "id": view.id,
        "relative_heading": normalize_heading(view.relative_heading),
        "pitch": view.pitch,
        "fov": view.fov,
        "view_kind": view.view_kind,
        "output_width": view.output_width,
        "output_height": view.output_height,
        "polygons": overlay_polygons_for_view(view, edge_samples=edge_samples),
    }


def _sample_frustum_edge_xyz(view: ViewSpec, edge_samples: int) -> np.ndarray:
    h_fov = radians(view.fov)
    v_fov = _vertical_fov_radians(
        horizontal_fov_degrees=view.fov,
        output_width=view.output_width,
        output_height=view.output_height,
    )
    grid = py360_utils.xyzpers(
        h_fov,
        v_fov,
        radians(_py360_u_degrees(view.relative_heading)),
        radians(view.pitch),
        (edge_samples, edge_samples),
        0,
    )
    top = grid[0, :, :]
    right = grid[1:, -1, :]
    bottom = grid[-1, -2::-1, :]
    left = grid[-2:0:-1, 0, :]
    return np.concatenate([top, right, bottom, left], axis=0)


def _vertical_fov_radians(
    *,
    horizontal_fov_degrees: float,
    output_width: int,
    output_height: int,
) -> float:
    aspect_height_over_width = output_height / output_width
    horizontal_fov = radians(horizontal_fov_degrees)
    return 2 * atan(tan(horizontal_fov / 2) * aspect_height_over_width)


def _py360_u_degrees(relative_heading: float) -> float:
    normalized = normalize_heading(relative_heading)
    return normalized if normalized <= 180 else normalized - 360


def _unwrap_x_values(x_values: np.ndarray, *, center: float) -> np.ndarray:
    unwrapped = x_values.copy()
    unwrapped[unwrapped - center > 0.5] -= 1
    unwrapped[center - unwrapped > 0.5] += 1
    return unwrapped


def _duplicate_for_canvas_clipping(polygon: Polygon) -> list[Polygon]:
    polygons = [polygon]
    if any(point["x"] > 1 for point in polygon):
        polygons.append([{"x": point["x"] - 1, "y": point["y"]} for point in polygon])
    if any(point["x"] < 0 for point in polygon):
        polygons.append([{"x": point["x"] + 1, "y": point["y"]} for point in polygon])
    return polygons
