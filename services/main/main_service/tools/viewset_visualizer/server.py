import json
import os
import re
from dataclasses import dataclass
from io import BytesIO
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import numpy as np
from PIL import Image

from main_service.processing.view_rendering import (
    PerspectiveViewSpec,
    render_perspective_view,
)
from main_service.tools.viewset_visualizer.geometry import view_to_api_dict
from main_service.tools.viewset_visualizer.viewsets import Viewset, load_viewsets

Image.MAX_IMAGE_PIXELS = None
_POSE_HEADING_PATTERN = re.compile(rb'GPano:PoseHeadingDegrees="([^"]+)"')
_SUPPORTED_PANO_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_PREVIEW_RENDER_SCALE = 4


@dataclass(frozen=True)
class PanoGallery:
    paths: list[Path]
    initial_index: int


def create_app_payload(
    pano_path: Path,
    viewsets_dir: Path,
    *,
    edge_samples: int = 49,
    pano_paths: list[Path] | None = None,
    pano_index: int = 0,
) -> dict[str, object]:
    gallery_paths = pano_paths or [pano_path]
    selected_index = _normalize_pano_index(pano_index, gallery_paths)
    selected_pano_path = gallery_paths[selected_index]
    with Image.open(selected_pano_path) as image:
        width, height = image.size

    viewsets = []
    for viewset in load_viewsets(viewsets_dir):
        viewsets.append(
            {
                "name": viewset.name,
                "description": viewset.description,
                "file": viewset.path.name,
                "views": [
                    view_to_api_dict(view, edge_samples=edge_samples)
                    for view in viewset.views
                ],
            }
        )

    return {
        "pano": {
            "filename": selected_pano_path.name,
            "width": width,
            "height": height,
            "url": "/pano?" + urlencode({"pano": selected_index}),
            "index": selected_index,
            "count": len(gallery_paths),
            "previous_index": (selected_index - 1) % len(gallery_paths),
            "next_index": (selected_index + 1) % len(gallery_paths),
            "filenames": [path.name for path in gallery_paths],
        },
        "viewsets": viewsets,
    }


class ViewsetVisualizerHandler(SimpleHTTPRequestHandler):
    pano_paths: list[Path]
    initial_pano_index: int
    viewsets_dir: Path
    edge_samples: int
    google_api_key: str | None
    north_offset: float | None
    pano_id: str | None
    latitude: float | None
    longitude: float | None

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/api/state":
            pano_index = self._query_pano_index(query)
            self._send_json(
                create_app_payload(
                    self.pano_paths[pano_index],
                    self.viewsets_dir,
                    edge_samples=self.edge_samples,
                    pano_paths=self.pano_paths,
                    pano_index=pano_index,
                )
            )
            return

        if parsed.path == "/view":
            pano_index = self._query_pano_index(query)
            viewset_name = _single_query_value(query, "viewset")
            view_id = _single_query_value(query, "view")
            body = render_view_page(
                self.pano_paths[pano_index],
                self.viewsets_dir,
                viewset_name=viewset_name,
                view_id=view_id,
                google_api_key=self.google_api_key,
                north_offset=self.north_offset,
                pano_id=self.pano_id,
                latitude=self.latitude,
                longitude=self.longitude,
                pano_index=pano_index,
            )
            self._send_bytes(body, "text/html; charset=utf-8")
            return

        if parsed.path == "/api/view-image":
            pano_index = self._query_pano_index(query)
            viewset_name = _single_query_value(query, "viewset")
            view_id = _single_query_value(query, "view")
            render_scale = _positive_int_query_value(query, "scale", default=1)
            body, content_type = render_view_image(
                self.pano_paths[pano_index],
                self.viewsets_dir,
                viewset_name=viewset_name,
                view_id=view_id,
                render_scale=render_scale,
            )
            self._send_bytes(body, content_type)
            return

        if parsed.path == "/pano":
            self._send_file(self.pano_paths[self._query_pano_index(query)])
            return

        super().do_GET()

    def _query_pano_index(self, query: dict[str, list[str]]) -> int:
        raw_index = query.get("pano", [str(self.initial_pano_index)])[0]
        try:
            return _normalize_pano_index(int(raw_index), self.pano_paths)
        except ValueError as exc:
            raise ValueError(f"invalid pano index: {raw_index}") from exc

    def _send_json(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        content_type = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "application/octet-stream"
        body = path.read_bytes()
        self._send_bytes(body, content_type)

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def render_view_image(
    pano_path: Path,
    viewsets_dir: Path,
    *,
    viewset_name: str,
    view_id: str,
    render_scale: int = 1,
) -> tuple[bytes, str]:
    if render_scale < 1:
        raise ValueError("render_scale must be positive")
    viewset = _find_viewset(load_viewsets(viewsets_dir), viewset_name)
    view = next((candidate for candidate in viewset.views if candidate.id == view_id), None)
    if view is None:
        raise ValueError(f"view not found: {view_id}")

    with Image.open(pano_path) as image:
        pano_array = np.asarray(image.convert("RGB"))
    rendered = render_perspective_view(
        pano_array,
        PerspectiveViewSpec(
            relative_heading=view.relative_heading,
            pitch=view.pitch,
            fov=view.fov,
            output_width=view.output_width * render_scale,
            output_height=view.output_height * render_scale,
        ),
    )
    output = BytesIO()
    Image.fromarray(rendered).save(output, format="PNG")
    return output.getvalue(), "image/png"


def render_view_page(
    pano_path: Path,
    viewsets_dir: Path,
    *,
    viewset_name: str,
    view_id: str,
    google_api_key: str | None,
    north_offset: float | None,
    pano_id: str | None,
    latitude: float | None,
    longitude: float | None,
    pano_index: int | None = None,
) -> bytes:
    viewset = _find_viewset(load_viewsets(viewsets_dir), viewset_name)
    view = next((candidate for candidate in viewset.views if candidate.id == view_id), None)
    if view is None:
        raise ValueError(f"view not found: {view_id}")

    resolved_north_offset = north_offset
    if resolved_north_offset is None:
        resolved_north_offset = read_gpano_pose_heading(pano_path)

    google_url = None
    if google_api_key is not None and resolved_north_offset is not None:
        google_url = build_google_embed_url(
            api_key=google_api_key,
            pano_id=pano_id or pano_path.stem,
            latitude=latitude,
            longitude=longitude,
            north_offset=resolved_north_offset,
            relative_heading=view.relative_heading,
            pitch=view.pitch,
            fov=view.fov,
        )

    preview_params: dict[str, str | int] = {"viewset": viewset.name, "view": view.id}
    if pano_index is not None:
        preview_params["pano"] = pano_index
    preview_params["scale"] = _PREVIEW_RENDER_SCALE
    preview_url = "/api/view-image?" + urlencode(preview_params)
    return _view_page_html(
        view_id=view.id,
        preview_url=preview_url,
        preview_scale=_PREVIEW_RENDER_SCALE,
        google_url=google_url,
        north_offset=resolved_north_offset,
        google_heading=(
            (resolved_north_offset + view.relative_heading) % 360
            if resolved_north_offset is not None
            else None
        ),
    ).encode("utf-8")


def read_gpano_pose_heading(pano_path: Path) -> float | None:
    with Image.open(pano_path) as image:
        xmp = image.info.get("xmp")
    if not isinstance(xmp, bytes):
        return None
    return parse_gpano_pose_heading(xmp)


def parse_gpano_pose_heading(xmp: bytes) -> float | None:
    match = _POSE_HEADING_PATTERN.search(xmp)
    if match is None:
        return None
    return float(match.group(1))


def build_google_embed_url(
    *,
    api_key: str,
    pano_id: str | None,
    latitude: float | None,
    longitude: float | None,
    north_offset: float,
    relative_heading: float,
    pitch: float,
    fov: float,
) -> str:
    params: dict[str, str] = {
        "key": api_key,
        "heading": _format_number((north_offset + relative_heading) % 360),
        "pitch": _format_number(pitch),
        "fov": _format_number(fov),
    }
    if pano_id:
        params["pano"] = pano_id
    if latitude is not None and longitude is not None:
        params["location"] = f"{_format_number(latitude)},{_format_number(longitude)}"
    if "pano" not in params and "location" not in params:
        raise ValueError("Google embed requires pano_id or latitude/longitude")
    return "https://www.google.com/maps/embed/v1/streetview?" + urlencode(params)


def run_server(
    *,
    pano_path: Path,
    viewsets_dir: Path,
    host: str,
    port: int,
    edge_samples: int = 49,
    google_api_key: str | None = None,
    google_api_key_env: str = "GOOGLE_MAPS_EMBED_API_KEY",
    north_offset: float | None = None,
    pano_id: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> None:
    static_dir = Path(__file__).parent / "static"
    pano_gallery = resolve_pano_gallery(pano_path)

    class ConfiguredHandler(ViewsetVisualizerHandler):
        pass

    ConfiguredHandler.pano_paths = pano_gallery.paths
    ConfiguredHandler.initial_pano_index = pano_gallery.initial_index
    ConfiguredHandler.viewsets_dir = viewsets_dir
    ConfiguredHandler.edge_samples = edge_samples
    ConfiguredHandler.google_api_key = google_api_key or os.getenv(google_api_key_env)
    ConfiguredHandler.north_offset = north_offset
    ConfiguredHandler.pano_id = pano_id
    ConfiguredHandler.latitude = latitude
    ConfiguredHandler.longitude = longitude
    handler = partial(ConfiguredHandler, directory=str(static_dir))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Viewset visualizer: http://{host}:{port}")
    print(f"Panos: {len(pano_gallery.paths)} from {pano_path}")
    print(f"Initial pano: {pano_gallery.paths[pano_gallery.initial_index]}")
    print(f"Viewsets: {viewsets_dir}")
    server.serve_forever()


def resolve_pano_gallery(pano_path: Path) -> PanoGallery:
    if pano_path.is_dir():
        paths = _sorted_supported_images(pano_path)
        initial_index = 0
    else:
        sibling_paths = _sorted_supported_images(pano_path.parent)
        if pano_path in sibling_paths:
            paths = sibling_paths
            initial_index = paths.index(pano_path)
        else:
            paths = [pano_path]
            initial_index = 0
    if not paths:
        raise ValueError(f"no pano images found: {pano_path}")
    return PanoGallery(paths=paths, initial_index=initial_index)


def resolve_pano_paths(pano_path: Path) -> list[Path]:
    return resolve_pano_gallery(pano_path).paths


def _sorted_supported_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _SUPPORTED_PANO_SUFFIXES
    )


def _normalize_pano_index(index: int, pano_paths: list[Path]) -> int:
    if not pano_paths:
        raise ValueError("no pano images configured")
    return index % len(pano_paths)


def _find_viewset(viewsets: list[Viewset], name: str) -> Viewset:
    for viewset in viewsets:
        if viewset.name == name or viewset.path.name == name:
            return viewset
    raise ValueError(f"viewset not found: {name}")


def _single_query_value(query: dict[str, list[str]], key: str) -> str:
    values = query.get(key)
    if not values or values[0] == "":
        raise ValueError(f"missing query parameter: {key}")
    return values[0]


def _positive_int_query_value(
    query: dict[str, list[str]],
    key: str,
    *,
    default: int,
) -> int:
    raw_value = query.get(key, [str(default)])[0]
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{key} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _view_page_html(
    *,
    view_id: str,
    preview_url: str,
    preview_scale: int,
    google_url: str | None,
    north_offset: float | None,
    google_heading: float | None,
) -> str:
    google_panel = (
        f'<iframe class="viewer" src="{_escape_attr(google_url)}" allowfullscreen '
        'referrerpolicy="no-referrer-when-downgrade"></iframe>'
        if google_url is not None
        else '<div class="viewer empty">Google embed unavailable. Set GOOGLE_MAPS_EMBED_API_KEY and provide/parse a north offset.</div>'
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape_text(view_id)}</title>
  <style>
    body {{ margin: 0; background: #0f172a; color: #e5e7eb; font-family: Arial, sans-serif; }}
    header {{ display: flex; align-items: center; gap: 16px; padding: 12px 16px; border-bottom: 1px solid #334155; }}
    button {{ border: 1px solid #475569; border-radius: 6px; background: #1e293b; color: #f8fafc; padding: 8px 10px; cursor: pointer; }}
    button.active {{ border-color: #ef4444; }}
    .meta {{ color: #94a3b8; font-size: 13px; }}
    .stage {{ height: calc(100vh - 58px); overflow: hidden; display: grid; place-items: center; background: #020617; }}
    .viewer {{ border: 0; background: #020617; }}
    img.viewer {{ display: block; width: 100%; height: 100%; min-width: 0; min-height: 0; max-width: 100vw; max-height: calc(100vh - 58px); object-fit: contain; }}
    #googleView, iframe.viewer {{ width: 100%; height: 100%; }}
    .empty {{ display: grid; place-items: center; color: #cbd5e1; padding: 24px; text-align: center; }}
    .viewer.hidden, .hidden {{ display: none; }}
  </style>
</head>
<body>
  <header>
    <strong>{_escape_text(view_id)}</strong>
    <button id="previewBtn" class="active">High-res preview</button>
    <button id="googleBtn">Google embed</button>
    <span class="meta">preview: {_escape_text(preview_scale)}x · north offset: {_escape_text(_format_optional(north_offset))} · google heading: {_escape_text(_format_optional(google_heading))}</span>
  </header>
  <section class="stage">
    <img id="previewView" class="viewer" src="{_escape_attr(preview_url)}" alt="High-resolution rendered perspective preview">
    <div id="googleView" class="hidden">{google_panel}</div>
  </section>
  <script>
    const previewBtn = document.getElementById("previewBtn");
    const googleBtn = document.getElementById("googleBtn");
    const previewView = document.getElementById("previewView");
    const googleView = document.getElementById("googleView");
    function show(which) {{
      const google = which === "google";
      previewView.classList.toggle("hidden", google);
      googleView.classList.toggle("hidden", !google);
      previewBtn.classList.toggle("active", which === "preview");
      googleBtn.classList.toggle("active", google);
    }}
    previewBtn.addEventListener("click", () => show("preview"));
    googleBtn.addEventListener("click", () => show("google"));
  </script>
</body>
</html>"""


def _format_number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _format_optional(value: float | None) -> str:
    return "unknown" if value is None else _format_number(value)


def _escape_text(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _escape_attr(value: object) -> str:
    return _escape_text(value).replace('"', "&quot;")
