# Query UI Compact Tile Grid Plan

## Goal

Make query result tiles more image-dense and exploration-oriented:

- Tile metadata footer is hidden by default and appears on hover/focus.
- Clicking the tile image starts visual similarity search.
- A small top-right hover action opens the tile in Google Street View.
- Cards have sharper edges and only a couple of pixels of grid gap.

## Non-goals

- Do not change embedding, vector search, Qdrant, or tile rendering backends.
- Do not add new persistent data.
- Do not change map coverage UI.

## Files Likely To Change

- `services/main/main_service/embedding/query_ui.py`
- `services/main/tests/embedding/test_query_ui.py`
- `.codex/plans/2026-06-03-query-ui-compact-tile-grid.md`

## Implementation Approach

1. Add focused tests for compact-grid CSS and result-card behavior.
2. Update shared CSS in `render_results_page`:
   - Use tighter grid columns and `gap: 2px`.
   - Use smaller border radius.
   - Position card metadata as a hover/focus overlay.
   - Add a top-right Google Maps action.
3. Update both text-search and image-search card builders:
   - Make `.thumb-link` point to `result.similar_url`.
   - Keep Google Maps behind a `.mapsButton` overlay.
   - Preserve image loading skeleton and responsive tile URL behavior.
4. Run focused query UI tests, then full service tests.

## Why This Approach

The current UI already has all required URLs in the search payload. This change
can stay frontend-only by reusing `similar_url`, `maps_url`, and the existing
tile endpoint. Keeping the backend unchanged reduces risk for the long-running
pipeline.

## Verification Commands

```bash
cd services/main
uv run python -m pytest tests/embedding/test_query_ui.py
uv run python -m pytest
cd ../..
git diff --check
```

Verification completed:

- `cd services/main && uv run python -m pytest tests/embedding/test_query_ui.py`: 17 passed.
- `cd services/main && uv run python -m pytest`: 211 passed.
- `git diff --check`: no output.
- Changed-file secret-pattern scan: no matches.

## Risks And Open Questions

- Hover-only metadata can be hard to reach on touch screens, so the CSS should
  also reveal metadata on focus.
- Tile-click similarity in the text-search tab navigates to `/image?view_id=...`;
  in the image-search tab it should continue using in-page similarity search.
