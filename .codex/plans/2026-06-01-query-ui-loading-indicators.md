# Query UI Loading Indicators Plan

## Goal

Add clear loading feedback to the local query UI while searches and tile images
are loading.

## Non-Goals

- Do not redesign the query UI.
- Do not change search, vector, or tile-rendering semantics.
- Do not add frontend dependencies.

## Files

- `services/main/main_service/embedding/query_ui.py`
- `services/main/tests/embedding/test_query_ui.py`

## Approach

1. Add tests that assert the rendered page includes a batch loading indicator,
   tile skeleton markup, and pulse animation CSS.
2. Update the query UI CSS with a small indeterminate top loading bar and a
   pulsing gray tile placeholder.
3. Update the browser JavaScript so batch loading toggles the page loading
   state, each card starts with a loading class, and image `load`/`error` events
   clear or mark the tile state.
4. Run focused query UI tests and the full service test suite.

## Verification

```bash
cd services/main
uv run pytest tests/embedding/test_query_ui.py -q
uv run pytest
git diff --check
```

## Risks

The UI is server-rendered Python plus inline JavaScript, so tests should assert
stable substrings instead of trying to execute browser behavior.
