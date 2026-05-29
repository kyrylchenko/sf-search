import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse
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
        vector = self.embedder.embed_text(query)
        hits = self.vector_store.search(vector, limit)
        return _results_for_hits(self.engine, hits)


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
    return f"""<article class="card" data-path="{image_path}">
  <img src="{image_url}" alt="{image_path}">
  <div class="meta">
    <strong>{html.escape(result.pano_id)}</strong>
    <span>{html.escape(result.viewset_name)} / {html.escape(result.view_id)}</span>
    <span>score {result.score:.4f}</span>
    <span>heading {result.relative_heading:.1f} · pitch {result.pitch:.1f} · fov {result.fov:.1f}</span>
    <span>{result.rendered_width}x{result.rendered_height}</span>
    <span>{html.escape(result.model_id)} · vector {html.escape(result.vector_id)}</span>
  </div>
</article>"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local embedding query UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--vector-store-dir", default=None)
    parser.add_argument("--model-id", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = CONFIG
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

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/image":
                self._serve_image(parsed.query)
                return
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0].strip()
            try:
                results = service.search(query, args.limit) if query else []
                html_body = render_results_page(query=query, results=results)
            except Exception as exc:
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
                self.send_error(404)
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", _content_type(path))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"url": f"http://{args.host}:{args.port}/"}, sort_keys=True))
    server.serve_forever()


def _content_type(path: Path) -> str:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if path.suffix.lower() == ".png":
        return "image/png"
    return "application/octet-stream"


if __name__ == "__main__":
    main()
