# Panorama Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the downloaded-pano preprocessing stage that renders configured perspective views, stores them locally, and records durable DB metadata.

**Architecture:** Add a dedicated `panorama_view_table`, a focused DB service for claiming/completing view rows, a NATS job source for `pano.processing.requested`, and a processing runner/CLI under `main_service.processing`. The runner loads one pano once, renders all configured viewsets through the shared projection function, writes files atomically, and acks the queue message after DB state is updated.

**Tech Stack:** Python 3.14, SQLAlchemy, NATS JetStream, Pillow, py360convert, pytest.

---

## Files

- Create `services/main/main_service/db/models/panorama_view.py`
- Modify `services/main/main_service/db/models/__init__.py`
- Modify `services/main/main_service/db/models/panorama.py`
- Modify `services/main/main_service/db/initialize_engine.py`
- Create `services/main/main_service/db/services/panorama_view_service.py`
- Modify `services/main/main_service/downloader/storage.py`
- Create `services/main/main_service/processing/nats_source.py`
- Create `services/main/main_service/processing/storage.py`
- Create `services/main/main_service/processing/runner.py`
- Create `services/main/main_service/processing/__main__.py`
- Modify `services/main/main_service/config.py`
- Modify `services/main/README.md`
- Add tests under `services/main/tests/db/` and `services/main/tests/processing/`

## Tasks

- [x] Write spec and implementation plan.
- [x] Add failing model/service tests for `panorama_view_table` and idempotent
  view claiming.
- [x] Implement the panorama view model and service.
- [x] Add failing tests for processing NATS message parsing/source ack behavior.
- [x] Implement the processing NATS source.
- [x] Add failing runner tests for generated views, duplicate skips, failures,
  and job acking.
- [x] Implement processing storage helpers and runner.
- [x] Add CLI/config tests and implement `python -m main_service.processing`.
- [x] Update README with run command and storage behavior.
- [x] Run targeted tests, then `uv run pytest -q`.
- [x] Commit with Conventional Commit format.

## Implementation Notes

- Default viewsets path: `../../docs/data/viewsets` when running from
  `services/main`, configurable through env/CLI.
- Default output path: `.local/panorama-views`.
- Default render scale: `2`.
- Default output format: JPEG, quality `95`, no chroma subsampling.
- Unique key includes `view_spec_hash`, so viewset edits produce new rows.
- Duplicate processing queue messages are expected and should be cheap.

## Verification Commands

```bash
cd services/main
uv run pytest tests/db tests/processing -q
uv run pytest -q
```
