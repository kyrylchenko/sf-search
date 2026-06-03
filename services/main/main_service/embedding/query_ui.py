import argparse
import html
import json
import logging
from collections import OrderedDict
from io import BytesIO
from math import degrees, floor, tau
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import perf_counter
from urllib.parse import parse_qs, urlencode, urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image
import numpy as np
from sqlalchemy import Engine, distinct, func, select
from sqlalchemy.orm import Session

from main_service.config import CONFIG, Settings
from main_service.db.initialize_engine import initialize_engine
from main_service.db.models.panorama import Panorama
from main_service.db.models.panorama_view import PanoramaView
from main_service.db.models.panorama_view_embedding import PanoramaViewEmbedding
from main_service.db.services.panorama_view_embedding_service import EmbeddingModelSpec
from main_service.embedding.model import ImageTextEmbedder, TransformersSiglipEmbedder
from main_service.embedding.vector_store import VectorStore
from main_service.embedding.vector_store_factory import create_vector_store
from main_service.ingestion.types import DownloadStatus, ProcessingStatus
from main_service.logging_config import configure_cli_logging
from main_service.observability import NoopTelemetry, configure_observability
from main_service.observability.telemetry import PipelineTelemetry
from main_service.processing.view_rendering import (
    PerspectiveViewSpec,
    render_perspective_view,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryResult:
    score: float
    view_db_id: int
    pano_image_path: str
    pano_id: str
    viewset_name: str
    view_id: str
    relative_heading: float
    pitch: float
    fov: float
    rendered_width: int
    rendered_height: int
    output_format: str
    image_quality: int | None
    model_id: str
    vector_id: str
    latitude: float | None = None
    longitude: float | None = None
    pano_heading: float | None = None


@dataclass(frozen=True)
class CoverageCell:
    south: float
    west: float
    north: float
    east: float
    pano_count: int
    downloaded_pano_count: int
    embedded_pano_count: int


@dataclass(frozen=True)
class CoverageStats:
    total_panos: int
    panos_with_coordinates: int
    downloaded_panos: int
    rendered_views: int
    complete_embeddings: int
    embedded_panos: int


class LocalQueryService:
    def __init__(
        self,
        *,
        engine: Engine,
        embedder: ImageTextEmbedder,
        vector_store: VectorStore,
        telemetry: PipelineTelemetry | None = None,
    ) -> None:
        self.engine = engine
        self.embedder = embedder
        self.vector_store = vector_store
        self.telemetry = telemetry or NoopTelemetry()

    def search(self, query: str, limit: int) -> list[QueryResult]:
        results, _ = self.search_page(query=query, offset=0, limit=limit)
        return results

    def search_page(
        self,
        *,
        query: str,
        offset: int,
        limit: int,
    ) -> tuple[list[QueryResult], bool]:
        started_at = perf_counter()
        search_limit = max(0, offset) + max(1, limit) + 1
        logger.info(
            "query_ui_search_start query=%r offset=%s limit=%s search_limit=%s",
            query,
            offset,
            limit,
            search_limit,
        )
        with self.telemetry.span("query.text_embedding"):
            stage_started_at = perf_counter()
            vector = self.embedder.embed_text(query)
        embed_seconds = perf_counter() - stage_started_at
        self.telemetry.record_duration(
            "query_text_embedding",
            embed_seconds,
            {"service": "query-ui", "operation": "text_embedding"},
        )
        with self.telemetry.span("query.vector_search"):
            stage_started_at = perf_counter()
            hits = self.vector_store.search(vector, search_limit)
        vector_search_seconds = perf_counter() - stage_started_at
        self.telemetry.record_duration(
            "query_vector_search",
            vector_search_seconds,
            {"service": "query-ui", "operation": "vector_search"},
        )
        with self.telemetry.span("query.db_hydration"):
            stage_started_at = perf_counter()
            hydrated = _results_for_hits(self.engine, hits)
        db_seconds = perf_counter() - stage_started_at
        page_start = max(0, offset)
        page_end = page_start + max(1, limit)
        results = hydrated[page_start:page_end]
        has_more = len(hydrated) > page_end
        self.telemetry.record_duration(
            "query_db_hydration",
            db_seconds,
            {"service": "query-ui", "operation": "db_hydration"},
        )
        total_seconds = perf_counter() - started_at
        self.telemetry.record_duration(
            "query_total",
            total_seconds,
            {"service": "query-ui", "operation": "total"},
        )
        self.telemetry.record_event(
            "query_ui_search_complete",
            {
                "results": len(results),
                "hits": len(hits),
                "offset": offset,
                "limit": limit,
                "has_more": has_more,
            },
        )
        logger.info(
            "query_ui_search_complete query=%r hits=%s results=%s has_more=%s",
            query,
            len(hits),
            len(results),
            has_more,
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
        return results, has_more


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
            view_db_id=view.id,
            pano_image_path=panorama.image_path or "",
            pano_id=panorama.orig_id,
            viewset_name=view.viewset_name,
            view_id=view.view_id,
            relative_heading=view.relative_heading,
            pitch=view.pitch,
            fov=view.fov,
            rendered_width=view.rendered_width,
            rendered_height=view.rendered_height,
            output_format=view.output_format,
            image_quality=view.image_quality,
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


def build_coverage_payload(
    engine: Engine,
    *,
    cell_degrees: float = 0.002,
) -> dict[str, object]:
    cell_size = max(0.0001, cell_degrees)
    with Session(engine) as session:
        total_panos = int(session.scalar(select(func.count(Panorama.id))) or 0)
        downloaded_panos = int(
            session.scalar(
                select(func.count(Panorama.id)).where(
                    Panorama.download_status == DownloadStatus.DOWNLOADED.value
                )
            )
            or 0
        )
        rendered_views = int(
            session.scalar(
                select(func.count(PanoramaView.id)).where(
                    PanoramaView.processing_status == ProcessingStatus.COMPLETE.value
                )
            )
            or 0
        )
        complete_embeddings = int(
            session.scalar(
                select(func.count(PanoramaViewEmbedding.id)).where(
                    PanoramaViewEmbedding.embedding_status
                    == ProcessingStatus.COMPLETE.value
                )
            )
            or 0
        )
        embedded_pano_ids = {
            int(pano_id)
            for pano_id in session.scalars(
                select(distinct(PanoramaView.panorama_id))
                .join(
                    PanoramaViewEmbedding,
                    PanoramaViewEmbedding.panorama_view_id == PanoramaView.id,
                )
                .where(
                    PanoramaViewEmbedding.embedding_status
                    == ProcessingStatus.COMPLETE.value
                )
            )
        }
        coord_rows = (
            session.execute(
                select(
                    Panorama.id,
                    Panorama.latitude,
                    Panorama.longitude,
                    Panorama.download_status,
                )
                .where(Panorama.latitude.is_not(None))
                .where(Panorama.longitude.is_not(None))
            )
            .all()
        )

    valid_rows = [
        (
            int(pano_id),
            float(latitude),
            float(longitude),
            str(download_status),
        )
        for pano_id, latitude, longitude, download_status in coord_rows
        if latitude is not None
        and longitude is not None
        and -90.0 <= float(latitude) <= 90.0
        and -180.0 <= float(longitude) <= 180.0
    ]
    stats = CoverageStats(
        total_panos=total_panos,
        panos_with_coordinates=len(valid_rows),
        downloaded_panos=downloaded_panos,
        rendered_views=rendered_views,
        complete_embeddings=complete_embeddings,
        embedded_panos=len(embedded_pano_ids),
    )
    grouped: dict[tuple[int, int], dict[str, int]] = {}
    for pano_id, latitude, longitude, download_status in valid_rows:
        cell_key = (
            floor(latitude / cell_size),
            floor(longitude / cell_size),
        )
        cell = grouped.setdefault(
            cell_key,
            {
                "pano_count": 0,
                "downloaded_pano_count": 0,
                "embedded_pano_count": 0,
            },
        )
        cell["pano_count"] += 1
        if download_status == DownloadStatus.DOWNLOADED.value:
            cell["downloaded_pano_count"] += 1
        if pano_id in embedded_pano_ids:
            cell["embedded_pano_count"] += 1

    cells = [
        CoverageCell(
            south=_round_coord(lat_index * cell_size),
            west=_round_coord(lon_index * cell_size),
            north=_round_coord((lat_index + 1) * cell_size),
            east=_round_coord((lon_index + 1) * cell_size),
            pano_count=counts["pano_count"],
            downloaded_pano_count=counts["downloaded_pano_count"],
            embedded_pano_count=counts["embedded_pano_count"],
        )
        for (lat_index, lon_index), counts in grouped.items()
    ]
    cells.sort(
        key=lambda cell: (
            -cell.embedded_pano_count,
            -cell.pano_count,
            cell.south,
            cell.west,
        )
    )
    bounds = _coverage_bounds(valid_rows)
    return {
        "cell_degrees": cell_size,
        "stats": _coverage_stats_to_json(stats),
        "bounds": bounds,
        "max_pano_count": max((cell.pano_count for cell in cells), default=0),
        "max_embedded_pano_count": max(
            (cell.embedded_pano_count for cell in cells),
            default=0,
        ),
        "cells": [_coverage_cell_to_json(cell) for cell in cells],
    }


def _coverage_bounds(
    rows: list[tuple[int, float, float, str]],
) -> dict[str, float] | None:
    if not rows:
        return None
    latitudes = [row[1] for row in rows]
    longitudes = [row[2] for row in rows]
    return {
        "south": _round_coord(min(latitudes)),
        "west": _round_coord(min(longitudes)),
        "north": _round_coord(max(latitudes)),
        "east": _round_coord(max(longitudes)),
    }


def _coverage_stats_to_json(stats: CoverageStats) -> dict[str, int]:
    return {
        "total_panos": stats.total_panos,
        "panos_with_coordinates": stats.panos_with_coordinates,
        "downloaded_panos": stats.downloaded_panos,
        "rendered_views": stats.rendered_views,
        "complete_embeddings": stats.complete_embeddings,
        "embedded_panos": stats.embedded_panos,
    }


def _coverage_cell_to_json(cell: CoverageCell) -> dict[str, object]:
    return {
        "bounds": {
            "south": cell.south,
            "west": cell.west,
            "north": cell.north,
            "east": cell.east,
        },
        "pano_count": cell.pano_count,
        "downloaded_pano_count": cell.downloaded_pano_count,
        "embedded_pano_count": cell.embedded_pano_count,
    }


def _round_coord(value: float) -> float:
    return round(value, 7)


def render_results_page(*, query: str, limit: int, active_tab: str = "search") -> str:
    escaped_query = html.escape(query, quote=True)
    is_coverage = active_tab == "coverage"
    leaflet_assets = _leaflet_assets() if is_coverage else ""
    main_body = (
        _coverage_page_body()
        if is_coverage
        else _search_page_body(escaped_query, limit)
    )
    script_body = _coverage_page_script() if is_coverage else _search_page_script()
    search_tab_class = "active" if not is_coverage else ""
    coverage_tab_class = "active" if is_coverage else ""
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Local Embedding Query</title>
  {leaflet_assets}
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
    .tabs {{
      display: flex;
      gap: 8px;
      margin: 0 0 12px;
    }}
    .tabs a {{
      border: 1px solid #34445a;
      border-radius: 6px;
      color: #b6c4d5;
      padding: 8px 10px;
      text-decoration: none;
      font-size: 14px;
      font-weight: 700;
    }}
    .tabs a.active {{
      background: #5ea1ff;
      border-color: #5ea1ff;
      color: #07111f;
    }}
    .topLoader {{
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      overflow: hidden;
      opacity: 0;
      pointer-events: none;
      transition: opacity 160ms ease;
      z-index: 2;
      background: #263142;
    }}
    .topLoader.is-active {{
      opacity: 1;
    }}
    .topLoader::before {{
      content: "";
      position: absolute;
      inset: 0;
      transform: translateX(-70%);
      background: linear-gradient(90deg, transparent, #8ab8ff, transparent);
      animation: topLoaderSweep 1.1s ease-in-out infinite;
    }}
    @keyframes topLoaderSweep {{
      from {{
        transform: translateX(-70%);
      }}
      to {{
        transform: translateX(70%);
      }}
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
    button[hidden] {{
      display: none;
    }}
    main {{
      padding: 20px;
    }}
    .status {{
      color: #b6c4d5;
      margin: 0 0 16px;
      min-height: 20px;
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
    .thumb-link {{
      position: relative;
      display: block;
      cursor: pointer;
      background: #111927;
    }}
    .tileSkeleton {{
      position: absolute;
      inset: 0;
      display: block;
      background: #202b3b;
      animation: tilePulse 1.25s ease-in-out infinite;
    }}
    @keyframes tilePulse {{
      0%, 100% {{
        background: #1a2433;
      }}
      50% {{
        background: #34445a;
      }}
    }}
    .card img {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      display: block;
      background: #0c111a;
    }}
    .card.is-loading img {{
      opacity: 0;
    }}
    .card:not(.is-loading) .tileSkeleton {{
      display: none;
    }}
    .card.is-error .tileSkeleton {{
      animation: none;
      background: #3a2630;
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
    #sentinel {{
      height: 1px;
    }}
    .loadMore {{
      display: block;
      margin: 20px auto 0;
      min-width: 160px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin: 0 0 16px;
    }}
    .stat {{
      border: 1px solid #263142;
      border-radius: 8px;
      background: #161f2d;
      padding: 12px;
    }}
    .stat span {{
      display: block;
      color: #b6c4d5;
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .stat strong {{
      display: block;
      color: #fff;
      font-size: 24px;
      line-height: 1;
    }}
    #coverageMap {{
      width: 100%;
      height: min(72vh, 780px);
      min-height: 520px;
      border: 1px solid #263142;
      border-radius: 8px;
      background: #0c111a;
    }}
    .leaflet-container {{
      color: #101620;
    }}
  </style>
</head>
<body>
  <div id="topLoader" class="topLoader" aria-hidden="true"></div>
  <header>
    <nav class="tabs" aria-label="Query UI sections">
      <a class="{search_tab_class}" href="/">Search</a>
      <a class="{coverage_tab_class}" href="/coverage">Coverage</a>
    </nav>
    <form method="get" action="/">
      <input name="q" value="{escaped_query}" placeholder="green graffiti of a woman">
      <button type="submit">Search</button>
    </form>
  </header>
  <main>{main_body}</main>
  <script>{script_body}</script>
</body>
</html>"""


def _leaflet_assets() -> str:
    return """
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>"""


def _search_page_body(escaped_query: str, limit: int) -> str:
    return f"""
    <p id="status" class="status"></p>
    <div id="results" class="grid" data-query="{escaped_query}" data-limit="{limit}"></div>
    <div id="sentinel" aria-hidden="true"></div>
    <button id="loadMoreButton" class="loadMore" type="button" hidden>Load more</button>"""


def _coverage_page_body() -> str:
    return """
    <p id="coverageStatus" class="status">Loading coverage...</p>
    <section class="stats" aria-label="Coverage stats">
      <div class="stat"><span>Total panos</span><strong id="statTotalPanos">0</strong></div>
      <div class="stat"><span>With coordinates</span><strong id="statCoordinatePanos">0</strong></div>
      <div class="stat"><span>Downloaded panos</span><strong id="statDownloadedPanos">0</strong></div>
      <div class="stat"><span>Embedded panos</span><strong id="statEmbeddedPanos">0</strong></div>
      <div class="stat"><span>Rendered views</span><strong id="statRenderedViews">0</strong></div>
      <div class="stat"><span>Complete embeddings</span><strong id="statCompleteEmbeddings">0</strong></div>
    </section>
    <div id="coverageMap" role="img" aria-label="Panorama coverage map"></div>"""


def _search_page_script() -> str:
    return """
    const grid = document.getElementById("results");
    const statusEl = document.getElementById("status");
    const topLoader = document.getElementById("topLoader");
    const loadMoreButton = document.getElementById("loadMoreButton");
    const query = grid.dataset.query || "";
    const limit = Number(grid.dataset.limit || "50");
    let offset = 0;
    let loading = false;
    let hasMore = true;

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[char]));
    }

    function appendCard(result) {
      const article = document.createElement("article");
      article.className = "card is-loading";
      article.innerHTML = `
        <a class="thumb-link" href="${escapeHtml(result.maps_url)}" target="_blank" rel="noopener noreferrer">
          <span class="tileSkeleton" aria-hidden="true"></span>
          <img src="${escapeHtml(result.tile_url)}" alt="${escapeHtml(result.pano_id)} ${escapeHtml(result.view_id)}" loading="lazy">
        </a>
        <div class="meta">
          <strong>${escapeHtml(result.pano_id)}</strong>
          <span>${escapeHtml(result.viewset_name)} / ${escapeHtml(result.view_id)}</span>
          <span>score ${Number(result.score).toFixed(4)}</span>
          <span>heading ${Number(result.relative_heading).toFixed(1)} · pitch ${Number(result.pitch).toFixed(1)} · fov ${Number(result.fov).toFixed(1)}</span>
          <span>${result.rendered_width}x${result.rendered_height}</span>
          <span>${escapeHtml(result.model_id)} · vector ${escapeHtml(result.vector_id)}</span>
          <span><a href="${escapeHtml(result.maps_url)}" target="_blank" rel="noopener noreferrer">Open in Google Maps</a></span>
        </div>
      `;
      const image = article.querySelector("img");
      image.addEventListener("load", () => {
        article.classList.remove("is-loading");
      });
      image.addEventListener("error", () => {
        article.classList.remove("is-loading");
        article.classList.add("is-error");
      });
      grid.appendChild(article);
    }

    function setLoading(active) {
      topLoader.classList.toggle("is-active", active);
      updateLoadMoreButton();
    }

    function updateLoadMoreButton() {
      loadMoreButton.hidden = !query || loading || !hasMore || grid.children.length === 0;
    }

    async function loadNext() {
      if (!query || loading || !hasMore) return;
      loading = true;
      setLoading(true);
      statusEl.textContent = offset === 0 ? "Searching..." : "Loading more...";
      const params = new URLSearchParams({ q: query, offset: String(offset), limit: String(limit) });
      try {
        const response = await fetch(`/api/search?${params.toString()}`);
        if (!response.ok) throw new Error(`Search failed: ${response.status}`);
        const payload = await response.json();
        for (const result of payload.results) appendCard(result);
        offset = payload.next_offset ?? offset + payload.results.length;
        hasMore = Boolean(payload.has_more);
        if (grid.children.length === 0) {
          statusEl.textContent = "No matches found.";
        } else {
          statusEl.textContent = hasMore ? "" : "End of results.";
        }
      } catch (error) {
        statusEl.textContent = error.message || "Search failed.";
        hasMore = false;
      } finally {
        loading = false;
        setLoading(false);
        updateLoadMoreButton();
      }
    }

    const observer = new IntersectionObserver(() => loadNext(), { rootMargin: "800px" });
    observer.observe(document.getElementById("sentinel"));
    loadMoreButton.addEventListener("click", () => loadNext());
    updateLoadMoreButton();
    loadNext();
  """


def _coverage_page_script() -> str:
    return """
    const statusEl = document.getElementById("coverageStatus");
    const numberFormatter = new Intl.NumberFormat();
    const map = L.map("coverageMap", { preferCanvas: true });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors"
    }).addTo(map);
    map.setView([37.7749, -122.4194], 12);

    function setStat(id, value) {
      document.getElementById(id).textContent = numberFormatter.format(value || 0);
    }

    function cellColor(cell, maxEmbedded, maxPanos) {
      const embeddedWeight = maxEmbedded > 0 ? cell.embedded_pano_count / maxEmbedded : 0;
      const panoWeight = maxPanos > 0 ? cell.pano_count / maxPanos : 0;
      const intensity = Math.max(embeddedWeight, panoWeight * 0.45);
      const alpha = Math.max(0.18, Math.min(0.78, 0.18 + intensity * 0.6));
      if (cell.embedded_pano_count > 0) return `rgba(255, 177, 66, ${alpha})`;
      return `rgba(94, 161, 255, ${alpha})`;
    }

    function tooltipHtml(cell) {
      return `
        <strong>${numberFormatter.format(cell.embedded_pano_count)} embedded panos</strong><br>
        ${numberFormatter.format(cell.pano_count)} total panos<br>
        ${numberFormatter.format(cell.downloaded_pano_count)} downloaded panos
      `;
    }

    async function loadCoverage() {
      try {
        const response = await fetch("/api/coverage");
        if (!response.ok) throw new Error(`Coverage failed: ${response.status}`);
        const payload = await response.json();
        const stats = payload.stats || {};
        setStat("statTotalPanos", stats.total_panos);
        setStat("statCoordinatePanos", stats.panos_with_coordinates);
        setStat("statDownloadedPanos", stats.downloaded_panos);
        setStat("statEmbeddedPanos", stats.embedded_panos);
        setStat("statRenderedViews", stats.rendered_views);
        setStat("statCompleteEmbeddings", stats.complete_embeddings);
        const maxEmbedded = payload.max_embedded_pano_count || 0;
        const maxPanos = payload.max_pano_count || 0;
        for (const cell of payload.cells || []) {
          const b = cell.bounds;
          L.rectangle(
            [[b.south, b.west], [b.north, b.east]],
            {
              color: cell.embedded_pano_count > 0 ? "#ffb142" : "#5ea1ff",
              weight: 1,
              fillColor: cellColor(cell, maxEmbedded, maxPanos),
              fillOpacity: 1
            }
          ).bindTooltip(tooltipHtml(cell), { sticky: true }).addTo(map);
        }
        if (payload.bounds) {
          const b = payload.bounds;
          map.fitBounds([[b.south, b.west], [b.north, b.east]], { padding: [24, 24] });
        }
        statusEl.textContent = `${numberFormatter.format((payload.cells || []).length)} coverage cells loaded. Hover a cell to see embedded pano count.`;
      } catch (error) {
        statusEl.textContent = error.message || "Coverage failed.";
      }
    }

    loadCoverage();
  """


def build_search_payload(
    *,
    query: str,
    results: list[QueryResult],
    offset: int,
    limit: int,
    has_more: bool,
) -> dict[str, object]:
    return {
        "query": query,
        "offset": offset,
        "limit": limit,
        "next_offset": offset + limit,
        "has_more": has_more,
        "results": [_result_to_json(result) for result in results],
    }


def _result_to_json(result: QueryResult) -> dict[str, object]:
    return {
        "score": result.score,
        "view_db_id": result.view_db_id,
        "tile_url": f"/tile?view_id={result.view_db_id}",
        "maps_url": google_maps_street_view_url(result),
        "pano_id": result.pano_id,
        "viewset_name": result.viewset_name,
        "view_id": result.view_id,
        "relative_heading": result.relative_heading,
        "pitch": result.pitch,
        "fov": result.fov,
        "rendered_width": result.rendered_width,
        "rendered_height": result.rendered_height,
        "model_id": result.model_id,
        "vector_id": result.vector_id,
    }


class TileRenderer:
    def __init__(self, *, max_cached_panos: int = 2) -> None:
        self.max_cached_panos = max(0, max_cached_panos)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = RLock()

    def render_tile(self, result: QueryResult) -> tuple[bytes, str]:
        pano_array = self._load_pano_array(result.pano_image_path)
        rendered = render_perspective_view(
            pano_array,
            PerspectiveViewSpec(
                relative_heading=result.relative_heading,
                pitch=result.pitch,
                fov=result.fov,
                output_width=result.rendered_width,
                output_height=result.rendered_height,
            ),
        )
        image = Image.fromarray(rendered)
        buffer = BytesIO()
        output_format = result.output_format.lower()
        if output_format in {"jpeg", "jpg"}:
            image.save(
                buffer,
                format="JPEG",
                quality=result.image_quality or 95,
                subsampling=0,
            )
            return buffer.getvalue(), "image/jpeg"
        if output_format == "png":
            image.save(buffer, format="PNG")
            return buffer.getvalue(), "image/png"
        raise ValueError(f"Unsupported tile output format: {result.output_format}")

    def _load_pano_array(self, pano_image_path: str) -> np.ndarray:
        with self._lock:
            cached = self._cache.get(pano_image_path)
            if cached is not None:
                self._cache.move_to_end(pano_image_path)
                return cached
        with Image.open(pano_image_path) as image:
            pano_array = np.asarray(image.convert("RGB"))
        with self._lock:
            if self.max_cached_panos > 0:
                self._cache[pano_image_path] = pano_array
                self._cache.move_to_end(pano_image_path)
                while len(self._cache) > self.max_cached_panos:
                    self._cache.popitem(last=False)
        return pano_array


def _result_for_view_id(engine: Engine, view_id: int) -> QueryResult | None:
    with Session(engine) as session:
        row = (
            session.execute(
                select(PanoramaView, Panorama)
                .join(Panorama, Panorama.id == PanoramaView.panorama_id)
                .where(PanoramaView.id == view_id)
            )
            .first()
        )
    if row is None:
        return None
    view, panorama = row
    if not panorama.image_path:
        return None
    return QueryResult(
        score=0.0,
        view_db_id=view.id,
        pano_image_path=panorama.image_path,
        pano_id=panorama.orig_id,
        viewset_name=view.viewset_name,
        view_id=view.view_id,
        relative_heading=view.relative_heading,
        pitch=view.pitch,
        fov=view.fov,
        rendered_width=view.rendered_width,
        rendered_height=view.rendered_height,
        output_format=view.output_format,
        image_quality=view.image_quality,
        model_id="",
        vector_id="",
        latitude=panorama.latitude,
        longitude=panorama.longitude,
        pano_heading=_extract_pano_heading(panorama.metadata_json),
    )


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
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--vector-store-dir", default=None)
    parser.add_argument(
        "--vector-store-kind",
        default=None,
        choices=["qdrant", "local_hnsw"],
    )
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--qdrant-collection", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--tile-cache-size", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = CONFIG
    configure_cli_logging(args.log_level or settings.log_level)
    telemetry = configure_observability(settings, "sf-search-query-ui")
    engine = initialize_engine(settings)
    model_id = args.model_id or settings.embedding_model_id
    embedder = TransformersSiglipEmbedder(
        model_id=model_id,
        revision=settings.embedding_model_revision,
        dtype=settings.embedding_dtype,
        device=_device_or_none(args.device or settings.embedding_device),
    )
    vector_store = create_vector_store(
        settings=settings,
        model_spec=_query_model_spec(settings, model_id),
        vector_store_kind=args.vector_store_kind,
        vector_store_dir=Path(args.vector_store_dir) if args.vector_store_dir else None,
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.qdrant_collection,
    )
    service = LocalQueryService(
        engine=engine,
        embedder=embedder,
        vector_store=vector_store,
        telemetry=telemetry,
    )
    tile_renderer = TileRenderer(max_cached_panos=args.tile_cache_size)
    _warm_query_service(service)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/search":
                self._serve_search_api(parsed.query)
                return
            if parsed.path == "/api/coverage":
                self._serve_coverage_api()
                return
            if parsed.path == "/tile":
                self._serve_tile(parsed.query)
                return
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0].strip()
            active_tab = "coverage" if parsed.path == "/coverage" else "search"
            try:
                logger.info("query_ui_request path=%s query=%r", parsed.path, query)
                html_body = render_results_page(
                    query=query,
                    limit=args.limit,
                    active_tab=active_tab,
                )
            except Exception as exc:
                logger.exception("query_ui_request_failed query=%r", query)
                html_body = render_results_page(
                    query=query,
                    limit=args.limit,
                    active_tab=active_tab,
                )
                html_body += f"\n<!-- {html.escape(str(exc))} -->"
            self._send_html(html_body)

        def _serve_search_api(self, query: str) -> None:
            params = parse_qs(query)
            text = params.get("q", [""])[0].strip()
            limit = _safe_positive_int(params.get("limit", [str(args.limit)])[0], args.limit)
            offset = _safe_nonnegative_int(params.get("offset", ["0"])[0])
            limit = min(limit, 100)
            if not text:
                self._send_json(
                    build_search_payload(
                        query=text,
                        results=[],
                        offset=offset,
                        limit=limit,
                        has_more=False,
                    )
                )
                return
            try:
                results, has_more = service.search_page(
                    query=text,
                    offset=offset,
                    limit=limit,
                )
            except Exception:
                logger.exception("query_ui_api_search_failed query=%r", text)
                self.send_error(500)
                return
            self._send_json(
                build_search_payload(
                    query=text,
                    results=results,
                    offset=offset,
                    limit=limit,
                    has_more=has_more,
                )
            )

        def _serve_coverage_api(self) -> None:
            try:
                payload = build_coverage_payload(engine)
            except Exception:
                logger.exception("query_ui_api_coverage_failed")
                self.send_error(500)
                return
            self._send_json(payload)

        def _serve_tile(self, query: str) -> None:
            view_id = _safe_positive_int(parse_qs(query).get("view_id", ["0"])[0], 0)
            if view_id <= 0:
                self.send_error(400)
                return
            result = _result_for_view_id(engine, view_id)
            if result is None:
                self.send_error(404)
                return
            try:
                data, content_type = tile_renderer.render_tile(result)
            except Exception:
                logger.exception("query_ui_tile_render_failed view_id=%s", view_id)
                self.send_error(500)
                return
            logger.info(
                "query_ui_tile_served view_id=%s pano_id=%s bytes=%s",
                view_id,
                result.pano_id,
                len(data),
            )
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, payload: dict[str, object]) -> None:
            encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info("query_ui_start url=http://%s:%s/", args.host, args.port)
    print(json.dumps({"url": f"http://{args.host}:{args.port}/"}, sort_keys=True))
    try:
        server.serve_forever()
    finally:
        telemetry.shutdown()


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


def _safe_positive_int(value: str, fallback: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return fallback
    return parsed if parsed > 0 else fallback


def _safe_nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return 0
    return max(0, parsed)


def _device_or_none(device: str | None) -> str | None:
    return None if device is None or device == "auto" else device


def _query_model_spec(settings: Settings, model_id: str) -> EmbeddingModelSpec:
    return EmbeddingModelSpec(
        model_provider=settings.embedding_model_provider,
        model_id=model_id,
        model_revision=settings.embedding_model_revision,
        preprocess_version=settings.embedding_preprocess_version,
        embedding_dimension=settings.embedding_dimension,
        embedding_dtype=settings.embedding_dtype,
        embedding_normalized=True,
    )


if __name__ == "__main__":
    main()
