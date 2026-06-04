# Query UI Image Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add image-based semantic search to the local query UI with upload/paste/drop input and result-tile similarity pivots.

**Architecture:** Extend the existing `query_ui.py` local HTTP server. Image search uses the same embedder/vector-store hydration path as text search, with uploaded images normalized into temporary JPEG files and tile similarity searches rendered through the existing `TileRenderer`.

**Tech Stack:** Python stdlib HTTP server, SQLAlchemy, PIL, temporary files, existing SigLIP embedder, existing vector store abstraction, vanilla browser JavaScript.

---

### Task 1: Tests

**Files:**
- Modify: `services/main/tests/embedding/test_query_ui.py`

- [x] Add tests for image tab markup, result `similar_url`, multipart image extraction, and `LocalQueryService.search_image_path`.

### Task 2: Backend Image Search

**Files:**
- Modify: `services/main/main_service/embedding/query_ui.py`

- [x] Add `LocalQueryService.search_image_path`.
- [x] Add shared vector search hydration helper.
- [x] Add multipart image extraction and temporary normalized image handling.
- [x] Add `/api/image-search` GET by `view_id` and POST multipart upload.

### Task 3: Frontend Image Tab

**Files:**
- Modify: `services/main/main_service/embedding/query_ui.py`

- [x] Add `Image Search` tab route `/image`.
- [x] Add paste, drag/drop, file picker, preview, search button, result grid,
  infinite scroll, and load-more fallback.
- [x] Add `Similar` button to result cards.

### Task 4: Verification And Commit

**Files:**
- Modify: `.codex/plans/2026-06-02-query-ui-image-search.md`

- [x] Run:

```bash
cd services/main
uv run python -m pytest tests/embedding/test_query_ui.py
uv run python -m pytest
cd ../..
git diff --check
rg -n "<private-value-patterns>" .codex/specs/2026-06-02-query-ui-image-search.md .codex/plans/2026-06-02-query-ui-image-search.md services/main/main_service/embedding/query_ui.py services/main/tests/embedding/test_query_ui.py -S
```

- [ ] Commit locally:

```bash
git add .codex/specs/2026-06-02-query-ui-image-search.md .codex/plans/2026-06-02-query-ui-image-search.md services/main/main_service/embedding/query_ui.py services/main/tests/embedding/test_query_ui.py
git commit -m "feat: add query ui image search"
```

Verification completed:

- `cd services/main && uv run python -m pytest tests/embedding/test_query_ui.py`: 15 passed.
- `cd services/main && uv run python -m pytest`: 209 passed.
- `git diff --check`: no output.
