import json
from io import BytesIO
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
from py360convert import e2p
from PIL import Image

from main_service.tools.viewset_visualizer.geometry import view_to_api_dict
from main_service.tools.viewset_visualizer.viewsets import Viewset, load_viewsets

Image.MAX_IMAGE_PIXELS = None


def create_app_payload(
    pano_path: Path,
    viewsets_dir: Path,
    *,
    edge_samples: int = 49,
) -> dict[str, object]:
    with Image.open(pano_path) as image:
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
            "filename": pano_path.name,
            "width": width,
            "height": height,
            "url": "/pano",
        },
        "viewsets": viewsets,
    }


class ViewsetVisualizerHandler(SimpleHTTPRequestHandler):
    pano_path: Path
    viewsets_dir: Path
    edge_samples: int

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self._send_json(
                create_app_payload(
                    self.pano_path,
                    self.viewsets_dir,
                    edge_samples=self.edge_samples,
                )
            )
            return

        if parsed.path == "/api/view-image":
            query = parse_qs(parsed.query)
            viewset_name = _single_query_value(query, "viewset")
            view_id = _single_query_value(query, "view")
            body, content_type = render_view_image(
                self.pano_path,
                self.viewsets_dir,
                viewset_name=viewset_name,
                view_id=view_id,
            )
            self._send_bytes(body, content_type)
            return

        if parsed.path == "/pano":
            self._send_file(self.pano_path)
            return

        super().do_GET()

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
) -> tuple[bytes, str]:
    viewset = _find_viewset(load_viewsets(viewsets_dir), viewset_name)
    view = next((candidate for candidate in viewset.views if candidate.id == view_id), None)
    if view is None:
        raise ValueError(f"view not found: {view_id}")

    with Image.open(pano_path) as image:
        pano_array = np.asarray(image.convert("RGB"))
    rendered = e2p(
        pano_array,
        view.fov,
        _py360_heading(view.relative_heading),
        view.pitch,
        (view.output_height, view.output_width),
    )
    output = BytesIO()
    Image.fromarray(rendered).save(output, format="JPEG", quality=92)
    return output.getvalue(), "image/jpeg"


def run_server(
    *,
    pano_path: Path,
    viewsets_dir: Path,
    host: str,
    port: int,
    edge_samples: int = 49,
) -> None:
    static_dir = Path(__file__).parent / "static"

    class ConfiguredHandler(ViewsetVisualizerHandler):
        pass

    ConfiguredHandler.pano_path = pano_path
    ConfiguredHandler.viewsets_dir = viewsets_dir
    ConfiguredHandler.edge_samples = edge_samples
    handler = partial(ConfiguredHandler, directory=str(static_dir))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Viewset visualizer: http://{host}:{port}")
    print(f"Pano: {pano_path}")
    print(f"Viewsets: {viewsets_dir}")
    server.serve_forever()


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


def _py360_heading(relative_heading: float) -> float:
    heading = relative_heading % 360
    return heading if heading <= 180 else heading - 360
