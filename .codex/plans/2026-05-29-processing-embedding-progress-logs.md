# Processing and Embedding Progress Logs Plan

## Goal

Make long processing and embedding runs visibly active while preserving final
JSON output on stdout.

## Approach

- Add lightweight progress callbacks to processing and embedding runners.
- Have CLIs print progress events to stderr as single-line JSON.
- Log before queue fetches, after fetches, per pano/view or embedding job, pause
  decisions, failures, and model load start/finish.
- Keep tests independent of real model downloads by asserting runner callbacks.

## Verification

- Add focused tests for progress callbacks.
- Run `uv run pytest tests/processing/test_processing_runner.py tests/embedding -q`.
- Run full `uv run pytest -q`.

