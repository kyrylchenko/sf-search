# Ephemeral View Tiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop retaining rendered view tile files while keeping embedding and local query UI functional from stored panoramas.

**Architecture:** Processing still writes temporary tile files and enqueues them for embedding. Embedding reads, embeds, updates Qdrant/Postgres, deletes the temporary file, and clears the view's temporary path. Query UI generates thumbnails on demand from panorama files and view metadata, with JSON pagination and browser infinite scroll.

**Tech Stack:** Python, SQLAlchemy, Pillow, py360convert renderer, Qdrant/local vector store, `ThreadingHTTPServer`, vanilla HTML/CSS/JS.

---

### Task 1: Document And Commit The Design

**Files:**
- Create: `.codex/specs/2026-06-01-ephemeral-view-tiles.md`
- Create: `.codex/plans/2026-06-01-ephemeral-view-tiles.md`

- [ ] **Step 1: Verify docs exist**

Run: `test -f .codex/specs/2026-06-01-ephemeral-view-tiles.md && test -f .codex/plans/2026-06-01-ephemeral-view-tiles.md`

Expected: exit code `0`.

- [ ] **Step 2: Commit docs**

Run:

```bash
git add .codex/specs/2026-06-01-ephemeral-view-tiles.md .codex/plans/2026-06-01-ephemeral-view-tiles.md
git commit -m "docs: plan ephemeral view tiles"
```

Expected: local commit is created.

### Task 2: Make Completed Views Deduplicate Without Persisted Tile Paths

**Files:**
- Modify: `services/main/main_service/db/services/panorama_view_service.py`
- Test: `services/main/tests/db/test_panorama_view_service.py`

- [ ] **Step 1: Write failing tests**

Add tests that:

- A completed view with `image_hash` but cleared `image_path` blocks duplicate processing claims.
- `clear_view_temp_image_path(view_id, expected_path)` clears `image_path` only when the expected path matches.

- [ ] **Step 2: Run red tests**

Run: `cd services/main && uv run pytest tests/db/test_panorama_view_service.py -q`

Expected: fails because duplicate claim still requires `image_path` and clear helper does not exist.

- [ ] **Step 3: Implement service behavior**

Change `claim_view_for_processing` duplicate check from requiring both path and hash to requiring complete status and `image_hash`.

Add:

```python
def clear_view_temp_image_path(self, view_id: int, expected_path: str) -> PanoramaView:
    ...
```

It should set `image_path = None` only when the current path equals `expected_path`, keep `image_hash`/`image_bytes`, and return the view.

- [ ] **Step 4: Run green tests**

Run: `cd services/main && uv run pytest tests/db/test_panorama_view_service.py -q`

Expected: all tests pass.

### Task 3: Delete Temporary Tiles After Embedding

**Files:**
- Modify: `services/main/main_service/embedding/runner.py`
- Test: `services/main/tests/embedding/test_embedding_runner.py`

- [ ] **Step 1: Write failing tests**

Add tests that:

- Successful single-image embedding deletes the temporary tile file and clears the view image path.
- Failed single-image embedding also deletes the temporary tile file and clears the view image path.
- Duplicate/skipped embedding messages delete the queued temporary path if it exists.
- Batch embedding deletes all temporary tile files after successful embedding.

- [ ] **Step 2: Run red tests**

Run: `cd services/main && uv run pytest tests/embedding/test_embedding_runner.py -q`

Expected: fails because the files are still present and the DB view path remains set.

- [ ] **Step 3: Implement cleanup**

Add helper functions to `embedding/runner.py`:

```python
def _cleanup_temp_tile_file(path: Path, progress: ProgressCallback | None) -> None:
    ...

def _clear_claimed_view_temp_path(
    embedding_service: PanoramaViewEmbeddingService,
    claim: PanoramaViewEmbedding,
) -> None:
    ...
```

Call cleanup after success, failure, and skipped queue messages. Use the claimed
`source_image_path` for successful/failed jobs and the message `image_path` for
skipped jobs.

- [ ] **Step 4: Run green tests**

Run: `cd services/main && uv run pytest tests/embedding/test_embedding_runner.py -q`

Expected: all tests pass.

### Task 4: Render Query UI Tiles From Panoramas

**Files:**
- Modify: `services/main/main_service/embedding/query_ui.py`
- Test: `services/main/tests/embedding/test_query_ui.py`

- [ ] **Step 1: Write failing tests**

Add tests that:

- Hydrated query results include DB `view_db_id` and `pano_image_path`.
- Rendered result cards use `/tile?view_id=...`, not `/image?path=...`.
- A tile rendering helper loads a pano and renders bytes matching the view dimensions.
- `/api/search` response JSON returns results with `next_offset`.

- [ ] **Step 2: Run red tests**

Run: `cd services/main && uv run pytest tests/embedding/test_query_ui.py -q`

Expected: fails because UI still serves `/image?path=...` and has no API/tile rendering helpers.

- [ ] **Step 3: Implement query UI backend**

Add `view_db_id` and `pano_image_path` to `QueryResult`.

Add a `TileRenderer` class with a small in-memory pano array cache and:

```python
def render_tile(self, result: QueryResult) -> tuple[bytes, str]:
    ...
```

Use `render_perspective_view` with the DB view values.

Add HTTP handlers:

- `/api/search?q=...&offset=...&limit=...`
- `/tile?view_id=...`

The API overfetches from vector search as `offset + limit` and returns the
requested slice. The tile endpoint hydrates one view by ID and renders from the
stored panorama file.

- [ ] **Step 4: Implement browser infinite scroll**

Change the page to render an initial shell and JavaScript that fetches
`/api/search`, appends cards, and uses `IntersectionObserver` to load the next
batch when the sentinel enters the viewport.

- [ ] **Step 5: Run green tests**

Run: `cd services/main && uv run pytest tests/embedding/test_query_ui.py -q`

Expected: all tests pass.

### Task 5: Update Runtime Defaults And Docs

**Files:**
- Modify: `services/main/main_service/config.py`
- Modify: `services/main/README.md` or root `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Change temporary tile default**

Set `pano_view_storage_dir` default to `.local/panorama-view-tmp`.

- [ ] **Step 2: Document behavior**

Document that panos are durable, rendered tiles are temporary handoff files, and
query UI renders tiles on demand from panos.

- [ ] **Step 3: Run focused tests**

Run:

```bash
cd services/main
uv run pytest tests/db/test_panorama_view_service.py tests/embedding/test_embedding_runner.py tests/embedding/test_query_ui.py -q
```

Expected: all focused tests pass.

### Task 6: Final Verification And Commit

**Files:**
- All changed files.

- [ ] **Step 1: Full test suite**

Run: `cd services/main && uv run pytest`

Expected: all tests pass.

- [ ] **Step 2: Static diff checks**

Run: `git diff --check`

Expected: no output, exit code `0`.

- [ ] **Step 3: Public repo secret scan**

Run:

```bash
rg -n "PRIVATE_HOST_PATTERN|API[_ -]?KEY|secret|token|password|Users/" .codex AGENTS.md README.md services/main/README.md services/main/main_service services/main/tests -S
```

Expected: no new private values. Existing placeholder/config field hits may remain.

- [ ] **Step 4: Commit**

Run:

```bash
git add .codex AGENTS.md README.md services/main
git commit -m "feat: make view tiles ephemeral"
```

Expected: local commit is created. Do not push.
