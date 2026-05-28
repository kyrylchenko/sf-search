import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image

from main_service.tools.viewset_visualizer.geometry import view_to_api_dict
from main_service.tools.viewset_visualizer.viewsets import load_viewsets

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
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


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
