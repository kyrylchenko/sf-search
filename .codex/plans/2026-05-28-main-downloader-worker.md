# Main Downloader Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a restart-safe downloader runner inside `services/main` that consumes a bounded batch of pano IDs from NATS, downloads images locally with configurable concurrency, updates Postgres, and emits processing jobs.

**Architecture:** Keep this as simple code in `services/main`, not a separate service or deployment unit. Postgres is the source of truth for pano download status and metadata, and it is the dedupe gate for duplicate NATS messages. NATS JetStream provides durable downloader input and processing-output streams; the runner acks input messages after DB state is safely updated.

**Tech Stack:** Python 3.14, SQLAlchemy, Postgres, NATS JetStream via `nats-py`, `streetlevel`, `aiohttp`, pytest, Docker Compose.

---

## File Map

- Modify `services/main/main_service/config.py`: downloader and processing queue settings.
- Modify `services/main/main_service/db/models/panorama.py`: image path, metadata JSON, downloaded timestamp.
- Modify `services/main/main_service/db/services/panorama_service.py`: claim, success, failure, and terminal-state helpers.
- Modify `services/main/main_service/ingestion/download_queue.py`: shared NATS stream helper and processing queue publisher.
- Create `services/main/main_service/downloader/__init__.py`: package marker.
- Create `services/main/main_service/downloader/types.py`: downloader job/result DTOs.
- Create `services/main/main_service/downloader/storage.py`: local image path, temp file, hash helpers.
- Create `services/main/main_service/downloader/streetview_client.py`: Street View protocol and real async implementation.
- Create `services/main/main_service/downloader/runner.py`: core bounded async download runner.
- Create `services/main/main_service/downloader/__main__.py`: runnable entry point.
- Create tests under `services/main/tests/downloader/`.
- Modify `.env.example`: downloader defaults.
- Modify `README.md`: local downloader run and inspection commands.
- Modify `.codex/specs/2026-05-28-main-downloader-worker.md`: update if implementation changes the design.

## Task 1: Add Downloader Config

- [ ] **Step 1: Write failing config tests**

Create `services/main/tests/downloader/test_downloader_config.py`:

```python
from main_service.config import Settings


def test_downloader_defaults_are_configured_for_local_dev() -> None:
    settings = Settings(
        _env_file=None,
        db_user="user",
        db_password="password",
        db_host="localhost",
        db_port=5432,
        db_name="sf_search",
    )

    assert settings.pano_download_concurrency == 5
    assert settings.pano_download_storage_dir == ".local/panoramas"
    assert settings.nats_url == "nats://localhost:4222"
    assert settings.pano_processing_stream == "PANO_PROCESSING"
    assert settings.pano_processing_subject == "pano.processing.requested"
```

- [ ] **Step 2: Run red**

```bash
cd services/main
uv run pytest tests/downloader/test_downloader_config.py -q
```

- [ ] **Step 3: Implement settings**

Add fields:

```python
pano_download_concurrency: int = Field(default=5)
pano_download_storage_dir: str = Field(default=".local/panoramas")
nats_url: str = Field(default="nats://localhost:4222")
pano_download_stream: str = Field(default="PANO_DOWNLOADS")
pano_download_subject: str = Field(default="pano.download.requested")
pano_processing_stream: str = Field(default="PANO_PROCESSING")
pano_processing_subject: str = Field(default="pano.processing.requested")
pano_downloader_consumer: str = Field(default="pano-downloader")
```

- [ ] **Step 4: Run green and commit**

```bash
uv run pytest tests/downloader/test_downloader_config.py tests/test_config.py -q
git add services/main/main_service/config.py services/main/tests/downloader/test_downloader_config.py
git commit -m "feat: add downloader configuration"
```

## Task 2: Extend Download Persistence Schema

- [ ] **Step 1: Write failing model tests**

Extend `services/main/tests/db/test_discovery_models.py` to assert `image_path`, `metadata_json`, and `downloaded_at` exist on `Panorama`.

- [ ] **Step 2: Implement columns**

In `Panorama` add:

```python
from datetime import datetime
from sqlalchemy import JSON

image_path: Mapped[str | None] = mapped_column(nullable=True)
metadata_json: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
downloaded_at: Mapped[datetime | None] = mapped_column(nullable=True)
```

- [ ] **Step 3: Run green and commit**

```bash
uv run pytest tests/db/test_discovery_models.py tests/test_models.py -q
git add services/main/main_service/db/models/panorama.py services/main/tests/db/test_discovery_models.py
git commit -m "feat: store pano download outputs"
```

## Task 3: Add Download Status and Dedupe Service Methods

- [ ] **Step 1: Write failing service tests**

Add tests to `services/main/tests/db/test_discovery_service.py` for:

- claiming a queued pano marks it `downloading`;
- downloaded pano with image path/hash is not claimable and is reported as an
  already-complete duplicate;
- marking success stores image path, hash, metadata, lat/lon, timestamp;
- marking failure increments attempts and stores `last_error`;
- marking failure at max attempts sets `skipped`.

- [ ] **Step 2: Implement service methods**

Add:

```python
claim_panorama_for_download(pano_id: PanoramaId) -> Panorama | None
is_panorama_download_complete(pano_id: PanoramaId) -> bool
mark_panorama_downloaded(...)
mark_panorama_download_failed(...)
```

- [ ] **Step 3: Run green and commit**

```bash
uv run pytest tests/db/test_discovery_service.py -q
git add services/main/main_service/db/services/panorama_service.py services/main/tests/db/test_discovery_service.py
git commit -m "feat: add pano download state transitions"
```

## Task 4: Add Processing Queue Publisher

- [ ] **Step 1: Write failing queue tests**

Add tests for `PanoProcessingMessage` and `NatsJetStreamPanoProcessingQueue`.

- [ ] **Step 2: Implement publisher**

Message shape:

```json
{
  "pano_id": "example-pano-id",
  "image_path": ".local/panoramas/example-pano-id.jpg",
  "source": "pano_downloader"
}
```

- [ ] **Step 3: Run green and commit**

```bash
uv run pytest tests/ingestion/test_download_queue.py -q
git add services/main/main_service/ingestion/download_queue.py services/main/tests/ingestion/test_download_queue.py
git commit -m "feat: add pano processing queue"
```

## Task 5: Add Local Download Storage Helpers

- [ ] **Step 1: Write failing storage tests**

Create `services/main/tests/downloader/test_storage.py` for safe file paths, temp file paths, atomic finalize, and SHA-256 hashing.

- [ ] **Step 2: Implement storage helpers**

Create `services/main/main_service/downloader/storage.py`.

- [ ] **Step 3: Run green and commit**

```bash
uv run pytest tests/downloader/test_storage.py -q
git add services/main/main_service/downloader/storage.py services/main/tests/downloader/test_storage.py
git commit -m "feat: add pano download storage"
```

## Task 6: Add Street View Download Client Boundary

- [ ] **Step 1: Write offline tests**

Create tests for metadata extraction from fake pano objects.

- [ ] **Step 2: Implement client protocol and real adapter**

Create `services/main/main_service/downloader/streetview_client.py`.

The real adapter should call:

```python
streetview.find_panorama_by_id_async(pano_id, session)
streetview.download_panorama_async(pano, str(output_path), session)
```

- [ ] **Step 3: Run green and commit**

```bash
uv run pytest tests/downloader/test_streetview_client.py -q
git add services/main/main_service/downloader/streetview_client.py services/main/tests/downloader/test_streetview_client.py
git commit -m "feat: add streetview downloader client"
```

## Task 7: Add Downloader Runner Core

- [ ] **Step 1: Write runner tests with fakes**

Create `services/main/tests/downloader/test_runner.py` covering:

- successful job downloads, updates DB, publishes processing message, acks input;
- duplicate/already-downloaded pano acks and skips by checking Postgres, without
  calling Street View and without publishing another processing job;
- failed download records DB failure and acks;
- concurrency limit is honored with `pano_download_concurrency`.

- [ ] **Step 2: Implement runner**

Create `services/main/main_service/downloader/runner.py`.

Keep queue-consuming code behind protocols so tests do not require NATS.

- [ ] **Step 3: Run green and commit**

```bash
uv run pytest tests/downloader/test_runner.py -q
git add services/main/main_service/downloader services/main/tests/downloader/test_runner.py
git commit -m "feat: add pano downloader runner"
```

## Task 8: Add Runnable Entry Point and Docs

- [ ] **Step 1: Add `python -m main_service.downloader` entry point**

Support safe manual options:

```bash
uv run python -m main_service.downloader --limit 5
```

- [ ] **Step 2: Update README and `.env.example`**

Document:

- local storage path `.local/panoramas`;
- how to run discovery;
- how to run downloader with a limit;
- how to inspect Postgres rows and processing queue depth.

- [ ] **Step 3: Commit**

```bash
git add README.md .env.example services/main/main_service/downloader/__main__.py
git commit -m "feat: add downloader entrypoint"
```

## Task 9: Runtime Verification

- [ ] **Step 1: Start infra**

```bash
docker compose up -d postgres nats
```

- [ ] **Step 2: Ensure discovery messages exist**

Run the existing one-tile discovery if needed.

- [ ] **Step 3: Download a small bounded batch**

```bash
cd services/main
uv run python -m main_service.downloader --limit 5
```

- [ ] **Step 4: Verify outputs**

Check:

- image files exist under `.local/panoramas`;
- Postgres rows show `download_status='downloaded'`;
- `image_path`, `image_hash`, `downloaded_at` are populated;
- processing stream message count increases.

- [ ] **Step 5: Full tests and final checks**

```bash
uv run pytest -q
git diff --check
git status --short
```

- [ ] **Step 6: Commit verification record**

```bash
git add .codex/plans/2026-05-28-main-downloader-worker.md
git commit -m "docs: mark downloader runner verified"
```

## Self-Review

- Scope stays inside `services/main`; no separate downloader service.
- Queue payloads remain small and contain only IDs/references.
- Local images remain under ignored `.local/`.
- Downloader runner is restartable: DB state plus JetStream acking controls recovery.
- Duplicates are skipped through Postgres before any Street View download call.
