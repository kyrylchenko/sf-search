# Postgres and NATS Compose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add local Postgres and NATS JetStream via Docker Compose, point the main service database engine at Postgres, provide a real NATS downloader queue adapter, and capture small live Street View coverage examples for future agents.

**Architecture:** Docker Compose owns local infrastructure only: Postgres for durable state and NATS JetStream for small downloader job messages. The Python service keeps its existing `Settings` shape, adds a configurable SQLAlchemy driver defaulting to `postgresql+psycopg`, and exposes a small URL builder so DB connection behavior can be unit-tested without starting Docker. Runtime verification starts Postgres and NATS, checks readiness, creates app tables through the real service code, publishes a real NATS downloader message, runs one live coverage discovery pass for a single tile, and saves small JSON documentation fixtures describing returned coverage data.

**Tech Stack:** Docker Compose, Postgres 16 Alpine, NATS JetStream, SQLAlchemy, psycopg 3, nats-py, Pydantic Settings, pytest, uv.

---

## Rules

- Do not commit `.env`, real credentials, private endpoints, logs, database volumes, or generated data.
- Use only local git commits in Conventional Commits format.
- Do not run remote/origin git operations.
- Use NATS JetStream rather than plain core NATS because discovery needs durable queue depth for backpressure.
- Keep this slice scoped to local Postgres/NATS infrastructure, DB URL wiring, queue publishing, and small discovery verification fixtures.
- Do not commit downloaded panorama images, large response dumps, or generated private runtime data.

## File Map

- Create `docker-compose.yml`: local Postgres and NATS services, named volumes, healthchecks.
- Create `.env.example`: public-safe local development placeholders for app DB and NATS config.
- Modify `services/main/pyproject.toml`: replace MySQL driver with psycopg.
- Modify `services/main/uv.lock`: update dependency lock after `uv sync`.
- Modify `services/main/main_service/config.py`: add `db_driver` default for Postgres.
- Modify `services/main/main_service/db/initialize_engine.py`: build Postgres SQLAlchemy URL from settings.
- Create `services/main/tests/db/test_initialize_engine.py`: offline tests for URL generation.
- Modify `services/main/main_service/ingestion/download_queue.py`: add NATS JetStream queue adapter that publishes JSON messages and reads stream depth.
- Modify `services/main/tests/ingestion/test_download_queue.py`: fake-client tests for NATS message publishing and queue depth.
- Modify `README.md`: document local Postgres startup and verification commands.
- Create `docs/data/streetview-coverage-tile-example.json`: small live sample of one coverage tile response.
- Create `docs/streetview-coverage-data.md`: notes about the coverage tile response shape and verification commands.

## Task 1: Document Local Postgres Plan

- [x] **Step 1: Create this plan**

Create `.codex/plans/2026-05-28-postgres-compose.md`.

- [x] **Step 2: Commit the plan**

```bash
git add .codex/plans/2026-05-28-postgres-compose.md
git commit -m "docs: plan postgres local database"
```

## Task 2: Add Postgres URL Tests

- [x] **Step 1: Write failing tests**

Create `services/main/tests/db/test_initialize_engine.py`:

```python
from main_service.config import Settings
from main_service.db.initialize_engine import build_database_url


def test_build_database_url_defaults_to_postgres_psycopg() -> None:
    settings = Settings(
        _env_file=None,
        db_user="user",
        db_password="password",
        db_host="localhost",
        db_port=5432,
        db_name="sf_search",
    )

    url = build_database_url(settings)

    assert url.drivername == "postgresql+psycopg"
    assert url.username == "user"
    assert url.password == "password"
    assert url.host == "localhost"
    assert url.port == 5432
    assert url.database == "sf_search"


def test_build_database_url_allows_driver_override() -> None:
    settings = Settings(
        _env_file=None,
        db_user="user",
        db_password="password",
        db_host="db",
        db_port=5432,
        db_name="sf_search",
        db_driver="postgresql+psycopg",
    )

    assert build_database_url(settings).drivername == "postgresql+psycopg"
```

- [x] **Step 2: Run tests and verify red**

```bash
cd services/main
uv run pytest tests/db/test_initialize_engine.py -q
```

Expected: import fails because `build_database_url` does not exist.

## Task 3: Switch Main Service DB Wiring to Postgres

- [x] **Step 1: Add config field and URL builder**

In `services/main/main_service/config.py`, add:

```python
    db_driver: str = Field(default="postgresql+psycopg")
```

In `services/main/main_service/db/initialize_engine.py`, add:

```python
from main_service.config import CONFIG, Settings


def build_database_url(settings: Settings) -> URL:
    return URL.create(
        settings.db_driver,
        username=settings.db_user,
        password=settings.db_password,
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
    )
```

Update `initialize_engine()` to call `create_engine(build_database_url(CONFIG))`.

- [x] **Step 2: Replace DB driver dependency**

In `services/main/pyproject.toml`, remove:

```toml
"pymysql>=1.1.2",
```

Add:

```toml
"psycopg[binary]>=3.2.13",
```

Then run:

```bash
cd services/main
uv sync
```

- [x] **Step 3: Run tests and verify green**

```bash
cd services/main
uv run pytest tests/db/test_initialize_engine.py tests/test_config.py -q
```

Expected: all selected tests pass.

- [x] **Step 4: Commit**

```bash
git add services/main/main_service/config.py services/main/main_service/db/initialize_engine.py services/main/pyproject.toml services/main/uv.lock services/main/tests/db/test_initialize_engine.py
git commit -m "feat: use postgres database driver"
```

## Task 4: Add NATS JetStream Queue Adapter

- [x] **Step 1: Write failing tests**

Extend `services/main/tests/ingestion/test_download_queue.py`:

```python
import json

from main_service.ingestion.download_queue import NatsJetStreamPanoDownloadQueue


class FakeStreamState:
    messages = 7


class FakeStreamInfo:
    state = FakeStreamState()


class FakeJetStream:
    def __init__(self) -> None:
        self.stream_names: list[str] = []
        self.published: list[dict[str, object]] = []

    async def stream_info(self, stream_name: str) -> FakeStreamInfo:
        self.stream_names.append(stream_name)
        return FakeStreamInfo()

    async def publish(self, subject: str, payload: bytes) -> None:
        self.published.append({"subject": subject, "payload": payload})


def test_nats_queue_reads_pending_count_from_stream_state() -> None:
    jetstream = FakeJetStream()
    queue = NatsJetStreamPanoDownloadQueue(
        jetstream=jetstream,
        stream_name="PANO_DOWNLOADS",
        subject="pano.download.requested",
    )

    assert queue.pending_count() == 7
    assert jetstream.stream_names == ["PANO_DOWNLOADS"]


def test_nats_queue_publishes_json_download_message() -> None:
    jetstream = FakeJetStream()
    queue = NatsJetStreamPanoDownloadQueue(
        jetstream=jetstream,
        stream_name="PANO_DOWNLOADS",
        subject="pano.download.requested",
    )
    message = PanoDownloadMessage(
        pano_id=PanoramaId("pano-a"),
        source_tile=MapTileKey(x=1, y=2, z=17),
    )

    queue.enqueue(message)

    published = jetstream.published[0]
    assert published["subject"] == "pano.download.requested"
    assert json.loads(published["payload"]) == message.to_dict()
```

- [x] **Step 2: Run tests and verify red**

```bash
cd services/main
uv run pytest tests/ingestion/test_download_queue.py -q
```

Expected: fail because `NatsJetStreamPanoDownloadQueue` does not exist.

- [x] **Step 3: Implement adapter**

Add `NatsJetStreamPanoDownloadQueue` in `services/main/main_service/ingestion/download_queue.py`.

It should:

- Accept a NATS JetStream context, stream name, and subject.
- Provide `connect(servers: str | list[str], stream_name: str, subject: str)` for runtime use.
- Ensure the stream exists with the download subject.
- Return pending depth from `stream_info(...).state.messages`.
- Publish JSON `PanoDownloadMessage.to_dict()` bodies to the subject.

- [x] **Step 4: Run tests and verify green**

```bash
cd services/main
uv run pytest tests/ingestion/test_download_queue.py -q
```

Expected: all queue tests pass.

- [x] **Step 5: Commit**

```bash
git add services/main/main_service/ingestion/download_queue.py services/main/tests/ingestion/test_download_queue.py
git commit -m "feat: add nats download queue"
```

## Task 5: Add Docker Compose Local Infrastructure and Docs Folder

- [ ] **Step 1: Add Compose and example env**

Create `docker-compose.yml` with:

- `postgres` service using `postgres:16-alpine`, local-only defaults, a named `postgres-data` volume, and a `pg_isready` healthcheck.
- `nats` service using `nats:2-alpine`, JetStream enabled, local-only client and monitoring ports, a named `nats-data` volume, and an HTTP healthcheck against the monitoring endpoint.

Create `.env.example` with public-safe local defaults:

```text
DB_USER=sf_search
DB_PASSWORD=sf_search_dev_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=sf_search
NATS_URL=nats://localhost:4222
PANO_DOWNLOAD_STREAM=PANO_DOWNLOADS
PANO_DOWNLOAD_SUBJECT=pano.download.requested
```

- [ ] **Step 2: Document usage**

Update `README.md` with:

```bash
docker compose up -d postgres
docker compose exec postgres pg_isready -U sf_search -d sf_search
docker compose exec postgres psql -U sf_search -d sf_search -c "select version();"
docker compose up -d nats
docker compose exec nats wget -qO- http://127.0.0.1:8222/healthz
```

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml .env.example README.md
git commit -m "feat: add local postgres and nats compose"
```

## Task 6: Capture Street View Coverage Example

- [ ] **Step 1: Add fixture writer**

Run a one-off Python command from `services/main` that:

1. Loads `target.geojson`.
2. Picks the first sorted map tile at zoom 17.
3. Calls `streetview.get_coverage_tile(tile.x, tile.y)`.
4. Writes a small sample JSON file with the source tile, total returned objects, and the first five objects represented by public scalar fields only.

Output path:

```text
docs/data/streetview-coverage-tile-example.json
```

- [ ] **Step 2: Add data documentation**

Create `docs/streetview-coverage-data.md` describing:

- The command used to capture the fixture.
- The tile coordinates used.
- The observed returned field names.
- The fact that downloader queues should carry IDs/references only, not panorama bytes.

- [ ] **Step 3: Commit**

```bash
git add docs/data/streetview-coverage-tile-example.json docs/streetview-coverage-data.md
git commit -m "docs: capture streetview coverage sample"
```

## Task 7: Verify Runtime Postgres, NATS, Discovery Code, and Full Tests

- [ ] **Step 1: Start local infrastructure**

```bash
docker compose up -d postgres nats
```

Expected: Postgres and NATS containers start.

- [ ] **Step 2: Verify Postgres readiness and query**

```bash
docker compose exec postgres pg_isready -U sf_search -d sf_search
docker compose exec postgres psql -U sf_search -d sf_search -c "select current_database(), current_user;"
```

Expected: readiness succeeds and query returns `sf_search`.

- [ ] **Step 3: Verify NATS readiness**

```bash
docker compose exec nats wget -qO- http://127.0.0.1:8222/healthz
```

Expected: health endpoint returns a healthy response.

- [ ] **Step 4: Verify app DB initialization against Postgres**

Run a one-off Python command from `services/main` that constructs `Settings` with local Compose credentials, calls `initialize_engine(settings)`, and checks that the discovery tables exist through SQLAlchemy inspection.

Expected: table check prints the discovery table names.

- [ ] **Step 5: Verify NATS queue adapter publishes a downloader job**

Run a one-off Python command from `services/main` that constructs `NatsJetStreamPanoDownloadQueue.connect(...)` for a verification stream/subject, enqueues one `PanoDownloadMessage`, and checks `pending_count() >= 1`.

Expected: pending count reaches one after publish.

- [ ] **Step 6: Verify one live discovery pass against Postgres**

Run a one-off Python command from `services/main` that constructs `Settings` with local Compose credentials, initializes the engine, creates `PanoramaService`, loads the first boundary tile, calls `discover_panos_for_tiles(...)` with `StreetLevelCoverageClient` and `NatsJetStreamPanoDownloadQueue`, and prints the resulting `DiscoveryResult`.

Expected: one tile is processed, discovered pano IDs are persisted, and queued downloader messages stay as small ID/reference payloads.

- [ ] **Step 7: Run all tests**

```bash
cd services/main
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 8: Check repo state**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors and no uncommitted changes after commits, except Docker runtime state outside git.

## Self-Review

- Scope is limited to local Postgres/NATS infra, main-service DB URL wiring, NATS queue publishing, live coverage shape docs, and one small discovery verification.
- No real credentials or private endpoints are introduced; committed values are local placeholders.
- Tests do not require Docker or a live database.
- Runtime verification uses Docker Compose, NATS, Postgres, and the real main-service discovery modules explicitly.
