# Query UI Coverage Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Coverage tab to the local query UI with map-based pano coverage visualization and top-level stats.

**Architecture:** The existing Python query UI will serve a second HTML tab and a new `/api/coverage` JSON endpoint. The endpoint reads current Postgres state, aggregates panorama coordinates into coarse geographic cells, and includes distinct embedded-pano counts per cell for hover tooltips.

**Tech Stack:** Python `http.server`, SQLAlchemy, existing ORM models, Leaflet loaded from CDN for the browser map, existing pytest coverage.

---

### Task 1: Coverage Data Model And API Builder

**Files:**
- Modify: `services/main/main_service/embedding/query_ui.py`
- Test: `services/main/tests/embedding/test_query_ui.py`

- [ ] Add tests that create an in-memory DB with three panos, completed views, and completed embeddings, then assert `build_coverage_payload(engine)` returns:
  - `stats.total_panos == 3`
  - `stats.panos_with_coordinates == 3`
  - `stats.embedded_panos == 2`
  - at least one cell with `embedded_pano_count == 2`
  - valid map bounds.

- [ ] Implement `CoverageCell`, `CoverageStats`, `build_coverage_payload`, and helpers in `query_ui.py`.

- [ ] Use distinct panorama IDs joined through `PanoramaView` and `PanoramaViewEmbedding` where `embedding_status == "complete"` for embedded pano counts.

### Task 2: Coverage Tab UI

**Files:**
- Modify: `services/main/main_service/embedding/query_ui.py`
- Test: `services/main/tests/embedding/test_query_ui.py`

- [ ] Extend `render_results_page` so it can render either search or coverage tab based on the request path.

- [ ] Add tests that the coverage page contains:
  - Leaflet CSS/JS links.
  - `id="coverageMap"`.
  - `/api/coverage`.
  - Tooltip text using `embedded_pano_count`.

- [ ] Implement the Coverage tab with top stats, a full-width map, colored rectangle cells, and hover tooltips showing embedded pano count and total pano count.

### Task 3: HTTP Routing

**Files:**
- Modify: `services/main/main_service/embedding/query_ui.py`

- [ ] Add `/coverage` route to render the coverage tab.

- [ ] Add `/api/coverage` route to return `build_coverage_payload(engine)`.

- [ ] Keep existing `/`, `/api/search`, and `/tile` behavior unchanged.

### Task 4: Verification

**Files:**
- Modify: `.codex/plans/2026-06-02-query-ui-coverage-map.md`

- [ ] Run:

```bash
cd services/main
uv run python -m pytest tests/embedding/test_query_ui.py
uv run python -m pytest
```

- [ ] Run from repo root:

```bash
git diff --check
rg -n "<private-value-patterns>" services/main/main_service/embedding/query_ui.py services/main/tests/embedding/test_query_ui.py .codex/plans/2026-06-02-query-ui-coverage-map.md -S
```

- [ ] Commit locally:

```bash
git add .codex/plans/2026-06-02-query-ui-coverage-map.md services/main/main_service/embedding/query_ui.py services/main/tests/embedding/test_query_ui.py
git commit -m "feat: add query ui coverage map"
```
