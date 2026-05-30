import argparse
import html
import json
import logging
from math import degrees, tau
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from urllib.parse import parse_qs, quote, urlencode, unquote, urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from main_service.config import CONFIG
from main_service.db.initialize_engine import initialize_engine
from main_service.db.models.panorama import Panorama
from main_service.db.models.panorama_view import PanoramaView
from main_service.db.models.panorama_view_embedding import PanoramaViewEmbedding
from main_service.embedding.model import ImageTextEmbedder, TransformersSiglipEmbedder
from main_service.embedding.vector_store import LocalHnswVectorStore, VectorStore
from main_service.logging_config import configure_cli_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryResult:
    score: float
    image_path: str
    pano_id: str
    viewset_name: str
    view_id: str
    relative_heading: float
    pitch: float
    fov: float
    rendered_width: int
    rendered_height: int
    model_id: str
    vector_id: str
    latitude: float | None = None
    longitude: float | None = None
    pano_heading: float | None = None


class LocalQueryService:
    def __init__(
        self,
        *,
        engine: Engine,
        embedder: ImageTextEmbedder,
        vector_store: VectorStore,
    ) -> None:
        self.engine = engine
        self.embedder = embedder
        self.vector_store = vector_store

    def search(self, query: str, limit: int) -> list[QueryResult]:
        started_at = perf_counter()
        logger.info("query_ui_search_start query=%r limit=%s", query, limit)
        stage_started_at = perf_counter()
        vector = self.embedder.embed_text(query)
        embed_seconds = perf_counter() - stage_started_at
        stage_started_at = perf_counter()
        hits = self.vector_store.search(vector, limit)
        vector_search_seconds = perf_counter() - stage_started_at
        stage_started_at = perf_counter()
        results = _results_for_hits(self.engine, hits)
        db_seconds = perf_counter() - stage_started_at
        total_seconds = perf_counter() - started_at
        logger.info(
            "query_ui_search_complete query=%r hits=%s results=%s",
            query,
            len(hits),
            len(results),
        )
        logger.info(
            (
                "query_ui_search_timing query=%r embed_seconds=%.3f "
                "vector_search_seconds=%.3f db_seconds=%.3f total_seconds=%.3f"
            ),
            query,
            embed_seconds,
            vector_search_seconds,
            db_seconds,
            total_seconds,
        )
        return results


def _results_for_hits(
    engine: Engine,
    hits: list[tuple[str, float]],
) -> list[QueryResult]:
    if not hits:
        return []
    scores = {vector_id: score for vector_id, score in hits}
    with Session(engine) as session:
        rows = (
            session.execute(
                select(PanoramaViewEmbedding, PanoramaView, Panorama)
                .join(
                    PanoramaView,
                    PanoramaView.id == PanoramaViewEmbedding.panorama_view_id,
                )
                .join(Panorama, Panorama.id == PanoramaView.panorama_id)
                .where(PanoramaViewEmbedding.vector_id.in_(scores.keys()))
            )
            .all()
        )
    by_vector_id = {
        embedding.vector_id: QueryResult(
            score=scores[embedding.vector_id or ""],
            image_path=view.image_path or "",
            pano_id=panorama.orig_id,
            viewset_name=view.viewset_name,
            view_id=view.view_id,
            relative_heading=view.relative_heading,
            pitch=view.pitch,
            fov=view.fov,
            rendered_width=view.rendered_width,
            rendered_height=view.rendered_height,
            model_id=embedding.model_id,
            vector_id=embedding.vector_id or "",
            latitude=panorama.latitude,
            longitude=panorama.longitude,
            pano_heading=_extract_pano_heading(panorama.metadata_json),
        )
        for embedding, view, panorama in rows
    }
    return [
        by_vector_id[vector_id]
        for vector_id, _ in hits
        if vector_id in by_vector_id
    ]


def render_results_page(*, query: str, results: list[QueryResult]) -> str:
    escaped_query = html.escape(query)
    cards = "\n".join(_render_result_card(result) for result in results)
    if not cards and query:
        cards = '<p class="empty">No matches found.</p>'
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Local Embedding Query</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101620;
      color: #ecf2f8;
    }}
    header {{
      position: sticky;
      top: 0;
      background: #101620;
      border-bottom: 1px solid #263142;
      padding: 16px 20px;
      z-index: 1;
    }}
    form {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      max-width: 980px;
    }}
    input {{
      min-width: 0;
      border: 1px solid #34445a;
      border-radius: 6px;
      background: #182232;
      color: #ecf2f8;
      padding: 10px 12px;
      font-size: 15px;
    }}
    button {{
      border: 0;
      border-radius: 6px;
      background: #5ea1ff;
      color: #07111f;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
    }}
    main {{
      padding: 20px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 16px;
    }}
    .card {{
      border: 1px solid #263142;
      border-radius: 8px;
      overflow: hidden;
      background: #161f2d;
    }}
    .card img {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      display: block;
      background: #0c111a;
    }}
    .thumb-link {{
      display: block;
      cursor: pointer;
    }}
    .meta {{
      padding: 12px;
      display: grid;
      gap: 6px;
      font-size: 13px;
      color: #b6c4d5;
    }}
    .meta strong {{
      color: #fff;
    }}
    .meta a {{
      color: #8ab8ff;
      text-decoration: none;
      font-weight: 700;
    }}
    .meta a:hover {{
      text-decoration: underline;
    }}
    .empty {{
      color: #b6c4d5;
    }}
  </style>
</head>
<body>
  <header>
    <form method="get" action="/">
      <input name="q" value="{escaped_query}" placeholder="green graffiti of a woman">
      <button type="submit">Search</button>
    </form>
  </header>
  <main>
    <div class="grid">{cards}</div>
  </main>
</body>
</html>"""


def _render_result_card(result: QueryResult) -> str:
    image_path = html.escape(result.image_path)
    image_url = "/image?path=" + quote(result.image_path)
    maps_url = google_maps_street_view_url(result)
    escaped_maps_url = html.escape(maps_url, quote=True)
    image_markup = (
        f'<a class="thumb-link" href="{escaped_maps_url}" target="_blank" '
        f'rel="noopener noreferrer"><img src="{image_url}" alt="{image_path}"></a>'
    )
    maps_link = (
        f'<a href="{escaped_maps_url}" target="_blank" '
        f'rel="noopener noreferrer">Open in Google Maps</a>'
    )
    return f"""<article class="card" data-path="{image_path}">
  {image_markup}
  <div class="meta">
    <strong>{html.escape(result.pano_id)}</strong>
    <span>{html.escape(result.viewset_name)} / {html.escape(result.view_id)}</span>
    <span>score {result.score:.4f}</span>
    <span>heading {result.relative_heading:.1f} · pitch {result.pitch:.1f} · fov {result.fov:.1f}</span>
    <span>{result.rendered_width}x{result.rendered_height}</span>
    <span>{html.escape(result.model_id)} · vector {html.escape(result.vector_id)}</span>
    <span>{maps_link}</span>
  </div>
</article>"""


def google_maps_street_view_url(result: QueryResult) -> str:
    heading = _google_heading(result)
    params: dict[str, str] = {
        "api": "1",
        "map_action": "pano",
        "pano": result.pano_id,
        "heading": f"{heading:.6f}",
        "pitch": f"{_clamp(result.pitch, -90.0, 90.0):.6f}",
        "fov": f"{_clamp(result.fov, 10.0, 100.0):.6f}",
    }
    if result.latitude is not None and result.longitude is not None:
        params["viewpoint"] = f"{result.latitude:.7f},{result.longitude:.7f}"
    return "https://www.google.com/maps/@" + "?" + urlencode(params)


def _google_heading(result: QueryResult) -> float:
    return ((result.pano_heading or 0.0) + result.relative_heading) % 360.0


def _extract_pano_heading(metadata: dict[str, object] | None) -> float | None:
    if not metadata:
        return None
    for key in ("heading_degrees", "pose_heading_degrees", "PoseHeadingDegrees"):
        value = metadata.get(key)
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed

    heading = _optional_float(metadata.get("heading"))
    if heading is not None:
        return degrees(heading) if abs(heading) <= tau else heading

    pose_heading = _optional_float(metadata.get("pose_heading"))
    if pose_heading is not None:
        return pose_heading
    return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local embedding query UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--vector-store-dir", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--log-level", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = CONFIG
    configure_cli_logging(args.log_level or settings.log_level)
    engine = initialize_engine(settings)
    model_id = args.model_id or settings.embedding_model_id
    embedder = TransformersSiglipEmbedder(
        model_id=model_id,
        revision=settings.embedding_model_revision,
        dtype=settings.embedding_dtype,
    )
    vector_store = LocalHnswVectorStore(
        root_dir=Path(args.vector_store_dir or settings.embedding_vector_store_dir),
        model_id=model_id,
        dimension=settings.embedding_dimension,
    )
    service = LocalQueryService(
        engine=engine,
        embedder=embedder,
        vector_store=vector_store,
    )
    _warm_query_service(service)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/image":
                self._serve_image(parsed.query)
                return
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0].strip()
            try:
                logger.info("query_ui_request path=%s query=%r", parsed.path, query)
                results = service.search(query, args.limit) if query else []
                html_body = render_results_page(query=query, results=results)
            except Exception as exc:
                logger.exception("query_ui_request_failed query=%r", query)
                html_body = render_results_page(query=query, results=[])
                html_body += f"\n<!-- {html.escape(str(exc))} -->"
            self._send_html(html_body)

        def _send_html(self, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _serve_image(self, query: str) -> None:
            path_value = parse_qs(query).get("path", [""])[0]
            path = Path(unquote(path_value))
            if not path.exists() or not path.is_file():
                logger.warning("query_ui_image_missing path=%s", path)
                self.send_error(404)
                return
            data = path.read_bytes()
            logger.info("query_ui_image_served path=%s bytes=%s", path, len(data))
            self.send_response(200)
            self.send_header("Content-Type", _content_type(path))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info("query_ui_start url=http://%s:%s/", args.host, args.port)
    print(json.dumps({"url": f"http://{args.host}:{args.port}/"}, sort_keys=True))
    server.serve_forever()


def _warm_query_service(service: LocalQueryService) -> None:
    started_at = perf_counter()
    logger.info("query_ui_warmup_start")
    try:
        results = service.search("street view warmup", 1)
    except Exception:
        logger.exception(
            "query_ui_warmup_failed seconds=%.3f",
            perf_counter() - started_at,
        )
        return
    logger.info(
        "query_ui_warmup_complete seconds=%.3f results=%s",
        perf_counter() - started_at,
        len(results),
    )


def _content_type(path: Path) -> str:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if path.suffix.lower() == ".png":
        return "image/png"
    return "application/octet-stream"


if __name__ == "__main__":
    main()
