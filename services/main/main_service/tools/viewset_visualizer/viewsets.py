import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from main_service.tools.viewset_visualizer.geometry import (
    ViewSpec,
    validate_google_compatible_view,
)


@dataclass(frozen=True)
class Viewset:
    name: str
    description: str
    path: Path
    views: list[ViewSpec]


def load_viewsets(directory: Path) -> list[Viewset]:
    if not directory.exists():
        raise ValueError(f"viewsets directory does not exist: {directory}")
    return [load_viewset(path) for path in sorted(directory.glob("*.json"))]


def load_viewset(path: Path) -> Viewset:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"viewset must be an object: {path}")

    name = _required_str(payload, "name", fallback=path.stem)
    description = _optional_str(payload.get("description"))
    raw_views = payload.get("views")
    if not isinstance(raw_views, list) or len(raw_views) == 0:
        raise ValueError(f"viewset must contain at least one view: {path}")

    views = [_parse_view(raw_view) for raw_view in raw_views]
    ids = [view.id for view in views]
    if len(ids) != len(set(ids)):
        raise ValueError(f"viewset has duplicate view ids: {path}")

    return Viewset(name=name, description=description, path=path, views=views)


def _parse_view(raw_view: object) -> ViewSpec:
    if not isinstance(raw_view, dict):
        raise ValueError("view must be an object")

    view = ViewSpec(
        id=_required_str(raw_view, "id"),
        relative_heading=_required_float(raw_view, "relative_heading"),
        pitch=_required_float(raw_view, "pitch"),
        fov=_required_float(raw_view, "fov"),
        view_kind=_optional_str(raw_view.get("view_kind"), default="custom"),
        output_width=_optional_int(raw_view.get("output_width"), default=512),
        output_height=_optional_int(raw_view.get("output_height"), default=512),
    )
    validate_google_compatible_view(view)
    return view


def _required_str(
    payload: dict[str, Any],
    key: str,
    *,
    fallback: str | None = None,
) -> str:
    value = payload.get(key, fallback)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_str(value: object, *, default: str = "") -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError("expected string")
    return value


def _required_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _optional_int(value: object, *, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int):
        raise ValueError("expected integer")
    return value
