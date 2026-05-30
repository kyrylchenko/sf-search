# Processing View Parallelism Plan

## Goal

Move processing concurrency from panorama jobs to generated views, because
`py360convert.e2p` rendering is the bottleneck and each panorama contains many
independent views.

## Evidence

- A local benchmark on a downloaded 8192x16384 panorama showed one 1024x1024
  perspective render takes about 8-10 seconds.
- JPEG writing for the same view takes about 0.006 seconds.
- Current processing logs show views inside a single pano rendering one by one.
- The async runner calls synchronous CPU work, so job-level asyncio concurrency
  does not help the per-pano view bottleneck.

## Approach

- Keep processing one panorama at a time to avoid loading many huge panoramas
  into memory.
- Reuse the existing `--concurrency` value as per-pano view render concurrency.
- Cap effective view render concurrency with `--max-view-concurrency` /
  `PANO_VIEW_MAX_RENDER_CONCURRENCY`, defaulting to 4. A local process launched
  with `--concurrency 100` grew past a 20 GB physical footprint while rendering
  full-resolution panoramas, so unsafe manual values must be capped rather than
  trusted.
- Claim/skips remain sequential through Postgres for deterministic dedupe.
- Render claimed views with a `ThreadPoolExecutor`.
- Mark completed/failed DB rows and enqueue embedding jobs after render results
  return.
- Lower duplicate view skip logs from warning to info/debug because duplicate
  processing queue messages are normal restart behavior.

## Verification

- Add a test with a fake renderer proving multiple views render concurrently.
- Run focused processing tests.
- Run full `uv run pytest -q`.
