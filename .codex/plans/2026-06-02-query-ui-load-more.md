# Query UI Load More Fallback Plan

## Goal

Make the local query UI load 50 results per batch by default and add a manual
bottom fallback button for large screens where infinite scroll does not trigger
reliably.

## Non-Goals

- Do not change the embedding search algorithm.
- Do not change coverage map behavior.
- Do not add new external dependencies.

## Files

- Modify: `services/main/main_service/embedding/query_ui.py`
- Modify: `services/main/tests/embedding/test_query_ui.py`

## Approach

1. Change the query UI CLI parser default `--limit` from `20` to `50`.
2. Add a `Load more` button after the infinite-scroll sentinel.
3. Keep the `IntersectionObserver` autoload behavior with its existing large
   root margin so normal scrolling should still prefetch before the user needs
   the button.
4. Wire the button to the same `loadNext()` function and only show it when a
   query has more results and the UI is not currently loading.
5. Add tests for default batch size and button markup/script.

## Verification

```bash
cd services/main
uv run python -m pytest tests/embedding/test_query_ui.py
uv run python -m pytest
cd ../..
git diff --check
```

## Risks

- The button is a fallback, not a replacement for infinite scroll. If the
  vector store reports `has_more=false`, the button must stay hidden.
