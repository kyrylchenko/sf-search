# Discovery Resume Requeue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make discovery restart-safe by replaying already-discovered pano IDs from the database instead of re-fetching Google coverage for completed map tiles.

**Architecture:** The database remains the source of truth. Discovery checks each tile's persisted `discovery_status` before making a coverage request. If the tile is already `complete`, discovery loads pano IDs linked to that tile and re-enqueues downloader jobs for non-terminal download statuses; duplicates are allowed because downloader workers must deduplicate/idempotently claim work.

**Tech Stack:** Python 3.14, SQLAlchemy, pytest, existing `PanoDownloadQueue` abstraction.

---

## Behavior

- Before starting each tile, keep the existing downstream queue backpressure check.
- If a map tile is already `complete`, do not call `CoverageClient.get_pano_ids_for_tile`.
- For complete tiles, load linked pano IDs from the DB.
- Re-push linked pano IDs to the downloader queue when `download_status` is not `downloaded` and not `skipped`.
- Mark replayed panorama rows as `queued` after enqueue.
- Count replayed queue messages in `DiscoveryResult.enqueued_downloads`.
- Keep duplicate messages acceptable; downstream downloader workers are expected to deduplicate/idempotently process.

## File Map

- Modify `services/main/main_service/db/services/panorama_service.py`: add linked-pano lookup for a map tile.
- Modify `services/main/main_service/ingestion/discovery.py`: skip coverage for complete tiles and replay linked pano IDs.
- Modify `services/main/tests/db/test_discovery_service.py`: service tests for linked pano lookup and terminal-status filtering.
- Modify `services/main/tests/ingestion/test_discovery.py`: restart/resume tests proving no coverage call for complete tiles.
- Modify `AGENTS.md`: document restart/requeue invariant for future agents.

## Tasks

- [x] Add failing DB service tests for loading pano IDs linked to a map tile, excluding terminal download statuses.
- [x] Implement the DB service lookup.
- [x] Add failing discovery tests for completed-tile replay without coverage calls.
- [x] Implement completed-tile replay in discovery orchestration.
- [x] Update `AGENTS.md` with the restart/requeue invariant.
- [x] Run targeted tests.
- [x] Run full test suite.
- [x] Commit with Conventional Commits.

## Verification Commands

```bash
cd services/main
uv run pytest tests/db/test_discovery_service.py tests/ingestion/test_discovery.py -q
uv run pytest -q
```

From repo root:

```bash
git diff --check
git status --short
```

## Risks

- A replayed complete tile can add duplicate downloader messages. This is intentional and safer than losing work after restart.
- `downloading` rows are replayed because the worker that owned them may have died. A future downloader claim protocol can add leases/heartbeats to make this more precise.
