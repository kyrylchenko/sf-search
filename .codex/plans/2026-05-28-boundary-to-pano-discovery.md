# Boundary to Pano Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first useful ingestion deliverable: read a boundary, generate all coverage tiles, query panorama IDs for each tile, persist map tiles/panoramas/tile-to-panorama links with resumable statuses, and enqueue discovered pano IDs for downstream downloading with queue backpressure.

**Architecture:** Keep the milestone single-process and testable without live network calls. Use small ingestion modules for domain types, boundary loading, Street View coverage client abstraction, downloader queue abstraction, and discovery orchestration. Use SQLAlchemy models/service methods for idempotent persistence so the job can resume and later continue into metadata fetching, downloading, view generation, and embeddings. RabbitMQ is the intended real queue, but unit tests use an in-memory fake.

**Tech Stack:** Python 3.14, uv, pytest, SQLAlchemy, Pydantic Settings, mercantile, shapely, streetlevel, pika/RabbitMQ.

---

## Git Rules

- Commit coherent local changes as work progresses.
- Use Conventional Commits format for every commit.
- Never run remote/origin operations: no `git push`, `git pull`, `git fetch`, commands targeting `origin`, or remote configuration changes.
- This is a public repo. Never commit real API keys, tokens, credentials, private endpoints, private filesystem paths, personal names, private organization names, generated private artifacts, downloaded panoramas, vectors, or indexes.

## Milestone Result

After this milestone, the project can run a discovery pass that:

1. Loads `target.geojson` or configured boundary GeoJSON.
2. Generates sorted map tile keys at configured zoom.
3. Persists every map tile with `discovery_status`.
4. Queries a coverage client for pano IDs in each tile.
5. Persists every discovered pano ID once.
6. Persists the many-to-many relationship between map tiles and pano IDs.
7. Enqueues each discovered pano ID as a small downloader job message.
8. Marks each tile as complete or failed with attempt counts and last errors.
9. Leaves every enqueued panorama in `download_status="queued"` until a downloader service claims it.

No panorama images are downloaded in this milestone.

## Queue and Backpressure Rule

RabbitMQ is the intended production queue. Messages must stay small and contain identifiers and references only. Do not put panorama bytes, generated view bytes, or 5-10 MB image payloads into RabbitMQ.

Initial downloader queue:

```text
pano.download.requested
```

Message shape:

```json
{
  "pano_id": "example-pano-id",
  "source": "coverage_discovery",
  "discovered_from_tile": {"x": 1, "y": 2, "z": 17}
}
```

Backpressure behavior:

- Before starting a new coverage tile, check downstream downloader queue depth.
- If downloader queue depth is greater than or equal to `max_downloader_queue_depth`, pause discovery before starting that next tile.
- If the current tile returns more IDs than the threshold, finish the current tile anyway. Persist all IDs, enqueue all downloader jobs, mark that tile complete, then pause before the next tile.
- The pause is represented in code as a returned `DiscoveryResult(paused=True, pause_reason=...)`, not as a tile DB status. The tile remains `pending` until a later run processes it.

## Status Model

Use two status enums:

- `ProcessingStatus`: `pending`, `processing`, `complete`, `failed`, `skipped`
- `DownloadStatus`: `pending`, `queued`, `downloading`, `downloaded`, `failed`, `skipped`

Map tiles use `ProcessingStatus` for coverage discovery.

Panoramas use:

- `metadata_status: ProcessingStatus`
- `download_status: DownloadStatus`
- future stages can add view/embedding statuses without changing this milestone.

## File Map

- Create `services/main/main_service/ingestion/__init__.py`: package marker.
- Create `services/main/main_service/ingestion/types.py`: status enums and DTOs.
- Create `services/main/main_service/ingestion/boundary_loader.py`: GeoJSON boundary to `MapTileKey` list.
- Create `services/main/main_service/ingestion/coverage_client.py`: protocol plus `streetlevel` implementation for pano ID coverage lookup.
- Create `services/main/main_service/ingestion/download_queue.py`: queue protocol, in-memory fake, and RabbitMQ-oriented message types.
- Create `services/main/main_service/ingestion/discovery.py`: orchestration for persisting tiles and discovered pano IDs.
- Modify `services/main/main_service/config.py`: ingestion settings.
- Create `services/main/main_service/db/models/map_tile.py`: map tile state.
- Create `services/main/main_service/db/models/map_tile_panorama.py`: tile/pano discovery link.
- Modify `services/main/main_service/db/models/panorama.py`: add status columns for metadata/download stages.
- Modify `services/main/main_service/db/models/__init__.py`: import all model modules.
- Modify `services/main/main_service/db/initialize_engine.py`: ensure new models are included.
- Modify `services/main/main_service/db/services/panorama_service.py`: idempotent persistence and state transitions.
- Create `services/main/tests/ingestion/`: offline ingestion tests.
- Create `services/main/tests/db/`: SQLite persistence tests.

## Task 1: Add Ingestion Domain Types

**Files:**

- Create: `services/main/main_service/ingestion/__init__.py`
- Create: `services/main/main_service/ingestion/types.py`
- Create: `services/main/tests/ingestion/test_types.py`

- [x] **Step 1: Write failing tests**

Create `services/main/tests/ingestion/test_types.py`:

```python
from main_service.ingestion.types import (
    DownloadStatus,
    MapTileKey,
    PanoramaId,
    ProcessingStatus,
)


def test_map_tile_key_tuple_round_trip() -> None:
    key = MapTileKey(x=1, y=2, z=17)

    assert key.as_tuple() == (1, 2, 17)


def test_processing_status_values_are_stable() -> None:
    assert ProcessingStatus.PENDING.value == "pending"
    assert ProcessingStatus.PROCESSING.value == "processing"
    assert ProcessingStatus.COMPLETE.value == "complete"
    assert ProcessingStatus.FAILED.value == "failed"
    assert ProcessingStatus.SKIPPED.value == "skipped"


def test_download_status_values_are_stable() -> None:
    assert DownloadStatus.PENDING.value == "pending"
    assert DownloadStatus.QUEUED.value == "queued"
    assert DownloadStatus.DOWNLOADING.value == "downloading"
    assert DownloadStatus.DOWNLOADED.value == "downloaded"
    assert DownloadStatus.FAILED.value == "failed"
    assert DownloadStatus.SKIPPED.value == "skipped"


def test_panorama_id_wraps_public_safe_example_value() -> None:
    pano_id = PanoramaId(value="example-pano-id")

    assert pano_id.value == "example-pano-id"
```

- [x] **Step 2: Run tests and verify red**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_types.py -q
```

Expected: fail during import because `main_service.ingestion` does not exist.

- [x] **Step 3: Implement domain types**

Create `services/main/main_service/ingestion/__init__.py` as an empty file.

Create `services/main/main_service/ingestion/types.py`:

```python
from dataclasses import dataclass
from enum import StrEnum


class ProcessingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


class DownloadStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class MapTileKey:
    x: int
    y: int
    z: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class PanoramaId:
    value: str
```

- [x] **Step 4: Run tests and verify green**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_types.py -q
```

Expected: `4 passed`.

- [x] **Step 5: Commit**

Run:

```bash
git add services/main/main_service/ingestion services/main/tests/ingestion/test_types.py
git commit -m "feat: add ingestion discovery types"
```

## Task 2: Add Boundary Loader and Ingestion Config

**Files:**

- Modify: `services/main/main_service/config.py`
- Create: `services/main/main_service/ingestion/boundary_loader.py`
- Create: `services/main/tests/ingestion/test_boundary_loader.py`

- [x] **Step 1: Write failing tests**

Create `services/main/tests/ingestion/test_boundary_loader.py`:

```python
from pathlib import Path

from main_service.config import Settings
from main_service.ingestion.boundary_loader import load_map_tiles_from_geojson


def test_load_map_tiles_from_geojson_uses_existing_geo_logic() -> None:
    tiles = load_map_tiles_from_geojson(Path("target.geojson"), zoom=17)

    assert len(tiles) == 598
    assert tiles[0].z == 17
    assert tiles == sorted(tiles, key=lambda tile: (tile.z, tile.x, tile.y))


def test_settings_exposes_discovery_defaults() -> None:
    settings = Settings(
        _env_file=None,
        db_user="user",
        db_password="password",
        db_host="localhost",
        db_port=3306,
        db_name="sf_search",
    )

    assert settings.map_tiles_zoom == 17
    assert settings.discovery_concurrency == 20
    assert settings.max_attempts == 5
```

- [x] **Step 2: Run tests and verify red**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_boundary_loader.py -q
```

Expected: fail because `main_service.ingestion.boundary_loader` does not exist.

- [x] **Step 3: Implement boundary loader**

Create `services/main/main_service/ingestion/boundary_loader.py`:

```python
import json
from pathlib import Path

from main_service.geo import generate_tiles_given_geojson
from main_service.ingestion.types import MapTileKey


def load_map_tiles_from_geojson(path: Path, zoom: int) -> list[MapTileKey]:
    geojson_data = json.loads(path.read_text())
    tiles = generate_tiles_given_geojson(geojson_data, zoom)
    return [
        MapTileKey(x=tile.x, y=tile.y, z=tile.z)
        for tile in sorted(tiles, key=lambda tile: (tile.z, tile.x, tile.y))
    ]
```

- [x] **Step 4: Add config fields**

In `services/main/main_service/config.py`, add:

```python
    map_tiles_zoom: int = Field(default=17)
    discovery_concurrency: int = Field(default=20)
    max_attempts: int = Field(default=5)
```

- [x] **Step 5: Run tests and verify green**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_boundary_loader.py tests/test_geo.py tests/test_config.py -q
```

Expected: all tests pass.

- [x] **Step 6: Commit**

Run:

```bash
git add services/main/main_service/config.py services/main/main_service/ingestion/boundary_loader.py services/main/tests/ingestion/test_boundary_loader.py
git commit -m "feat: load discovery tiles from boundary"
```

## Task 3: Add Discovery Persistence Models

**Files:**

- Create: `services/main/main_service/db/models/map_tile.py`
- Create: `services/main/main_service/db/models/map_tile_panorama.py`
- Modify: `services/main/main_service/db/models/panorama.py`
- Modify: `services/main/main_service/db/models/__init__.py`
- Modify: `services/main/main_service/db/initialize_engine.py`
- Create: `services/main/tests/db/test_discovery_models.py`

- [x] **Step 1: Write failing metadata tests**

Create `services/main/tests/db/test_discovery_models.py`:

```python
from sqlalchemy import create_engine, inspect

from main_service.db.models.base import Base
from main_service.db.models.map_tile import MapTile
from main_service.db.models.map_tile_panorama import MapTilePanorama
from main_service.db.models.panorama import Panorama


def test_discovery_tables_exist_in_metadata() -> None:
    assert "map_tile_table" in Base.metadata.tables
    assert "map_tile_panorama_table" in Base.metadata.tables


def test_map_tile_has_unique_tile_key_and_status_columns() -> None:
    columns = MapTile.__table__.c
    constraint_columns = {
        tuple(column.name for column in constraint.columns)
        for constraint in MapTile.__table__.constraints
        if constraint.name == "uq_map_tile_xyz"
    }

    assert ("x", "y", "z") in constraint_columns
    assert "discovery_status" in columns
    assert "attempt_count" in columns
    assert "last_error" in columns


def test_map_tile_panorama_has_unique_link_key() -> None:
    constraint_columns = {
        tuple(column.name for column in constraint.columns)
        for constraint in MapTilePanorama.__table__.constraints
        if constraint.name == "uq_map_tile_panorama"
    }

    assert ("map_tile_id", "panorama_id") in constraint_columns


def test_panorama_has_discovery_and_later_stage_status_columns() -> None:
    columns = Panorama.__table__.c

    assert "metadata_status" in columns
    assert "download_status" in columns
    assert "discovered_at_tile_count" in columns
    assert "attempt_count" in columns
    assert "last_error" in columns


def test_metadata_can_create_sqlite_schema() -> None:
    engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(engine)

    assert inspect(engine).has_table("map_tile_table") is True
    assert inspect(engine).has_table("map_tile_panorama_table") is True
```

- [x] **Step 2: Run tests and verify red**

Run:

```bash
cd services/main
uv run pytest tests/db/test_discovery_models.py -q
```

Expected: fail because `map_tile` models do not exist.

- [x] **Step 3: Implement `MapTile`**

Create `services/main/main_service/db/models/map_tile.py`:

```python
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from main_service.ingestion.types import ProcessingStatus

from .base import Base


class MapTile(Base):
    __tablename__ = "map_tile_table"
    __table_args__ = (UniqueConstraint("x", "y", "z", name="uq_map_tile_xyz"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    x: Mapped[int]
    y: Mapped[int]
    z: Mapped[int]
    discovery_status: Mapped[str] = mapped_column(
        default=ProcessingStatus.PENDING.value
    )
    attempt_count: Mapped[int] = mapped_column(default=0)
    last_error: Mapped[str | None]
```

- [x] **Step 4: Implement `MapTilePanorama`**

Create `services/main/main_service/db/models/map_tile_panorama.py`:

```python
from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class MapTilePanorama(Base):
    __tablename__ = "map_tile_panorama_table"
    __table_args__ = (
        UniqueConstraint(
            "map_tile_id",
            "panorama_id",
            name="uq_map_tile_panorama",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    map_tile_id: Mapped[int] = mapped_column(ForeignKey("map_tile_table.id"))
    panorama_id: Mapped[int] = mapped_column(ForeignKey("panorama_table.id"))
```

- [x] **Step 5: Extend `Panorama`**

In `services/main/main_service/db/models/panorama.py`, add:

```python
from main_service.ingestion.types import DownloadStatus, ProcessingStatus
```

Add fields:

```python
    metadata_status: Mapped[str] = mapped_column(default=ProcessingStatus.PENDING.value)
    download_status: Mapped[str] = mapped_column(default=DownloadStatus.PENDING.value)
    discovered_at_tile_count: Mapped[int] = mapped_column(default=0)
    attempt_count: Mapped[int] = mapped_column(default=0)
    last_error: Mapped[str | None]
```

- [x] **Step 6: Import new models**

Update `services/main/main_service/db/models/__init__.py`:

```python
from . import embedding, map_tile, map_tile_panorama, panorama, tile

__all__ = [
    "embedding",
    "map_tile",
    "map_tile_panorama",
    "panorama",
    "tile",
]
```

Update `services/main/main_service/db/initialize_engine.py` import:

```python
from main_service.db.models import embedding, map_tile, map_tile_panorama, panorama, tile
```

- [x] **Step 7: Run tests and verify green**

Run:

```bash
cd services/main
uv run pytest tests/db/test_discovery_models.py tests/test_models.py -q
```

Expected: all tests pass.

- [x] **Step 8: Commit**

Run:

```bash
git add services/main/main_service/db services/main/tests/db/test_discovery_models.py
git commit -m "feat: add coverage discovery schema"
```

## Task 4: Add Idempotent Discovery Persistence Service

**Files:**

- Modify: `services/main/main_service/db/services/panorama_service.py`
- Create: `services/main/tests/db/test_discovery_service.py`

- [x] **Step 1: Write failing service tests**

Create `services/main/tests/db/test_discovery_service.py`:

```python
from sqlalchemy import create_engine

from main_service.db.models import embedding, map_tile, map_tile_panorama, panorama, tile
from main_service.db.models.base import Base
from main_service.db.services.panorama_service import PanoramaService
from main_service.ingestion.types import (
    DownloadStatus,
    MapTileKey,
    PanoramaId,
    ProcessingStatus,
)


def make_service() -> PanoramaService:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return PanoramaService(engine)


def test_upsert_map_tile_is_idempotent() -> None:
    service = make_service()
    key = MapTileKey(x=1, y=2, z=17)

    first = service.upsert_map_tile(key)
    second = service.upsert_map_tile(key)

    assert first.id == second.id
    assert first.discovery_status == ProcessingStatus.PENDING.value


def test_upsert_discovered_panorama_is_idempotent() -> None:
    service = make_service()
    pano_id = PanoramaId(value="example-pano-id")

    first = service.upsert_discovered_panorama(pano_id)
    second = service.upsert_discovered_panorama(pano_id)

    assert first.id == second.id
    assert first.orig_id == "example-pano-id"
    assert first.download_status == DownloadStatus.PENDING.value


def test_link_map_tile_to_panorama_is_idempotent() -> None:
    service = make_service()
    tile_row = service.upsert_map_tile(MapTileKey(x=1, y=2, z=17))
    pano_row = service.upsert_discovered_panorama(PanoramaId(value="example-pano-id"))

    service.link_map_tile_to_panorama(tile_row.id, pano_row.id)
    service.link_map_tile_to_panorama(tile_row.id, pano_row.id)

    assert service.count_map_tile_panorama_links() == 1


def test_mark_map_tile_discovery_complete_updates_status() -> None:
    service = make_service()
    tile_row = service.upsert_map_tile(MapTileKey(x=1, y=2, z=17))

    updated = service.mark_map_tile_discovery_complete(tile_row.id)

    assert updated.discovery_status == ProcessingStatus.COMPLETE.value


def test_mark_panorama_download_queued_updates_status() -> None:
    service = make_service()
    pano_row = service.upsert_discovered_panorama(PanoramaId(value="example-pano-id"))

    updated = service.mark_panorama_download_queued(pano_row.id)

    assert updated.download_status == DownloadStatus.QUEUED.value
```

- [x] **Step 2: Run tests and verify red**

Run:

```bash
cd services/main
uv run pytest tests/db/test_discovery_service.py -q
```

Expected: fail because the new service methods do not exist.

- [x] **Step 3: Implement service methods**

Add methods to `services/main/main_service/db/services/panorama_service.py`:

```python
    def upsert_map_tile(self, key: MapTileKey) -> MapTile:
        ...

    def upsert_discovered_panorama(self, pano_id: PanoramaId) -> Panorama:
        ...

    def link_map_tile_to_panorama(self, map_tile_id: int, panorama_id: int) -> None:
        ...

    def count_map_tile_panorama_links(self) -> int:
        ...

    def mark_map_tile_discovery_complete(self, map_tile_id: int) -> MapTile:
        ...

    def mark_map_tile_discovery_failed(self, map_tile_id: int, error: str) -> MapTile:
        ...
```

Implementation requirements:

- Use `select(...)` before insert so repeated calls are idempotent.
- Return detached ORM objects by calling `session.expunge(...)` before returning.
- `upsert_discovered_panorama` must create panoramas with `download_status=DownloadStatus.PENDING.value`.
- `mark_map_tile_discovery_failed` must increment `attempt_count` by one and set `last_error`.

- [x] **Step 6: Add queued status transition**

Add a service method:

```python
    def mark_panorama_download_queued(self, panorama_id: int) -> Panorama:
        ...
```

It must set `download_status=DownloadStatus.QUEUED.value` and return a detached `Panorama`.

- [x] **Step 7: Run tests and verify green**

Run:

```bash
cd services/main
uv run pytest tests/db/test_discovery_service.py -q
```

Expected: `5 passed`.

- [x] **Step 8: Commit**

Run:

```bash
git add services/main/main_service/db/services/panorama_service.py services/main/tests/db/test_discovery_service.py
git commit -m "feat: add coverage discovery persistence"
```

## Task 5: Add Coverage Client Abstraction

**Files:**

- Create: `services/main/main_service/ingestion/coverage_client.py`
- Create: `services/main/tests/ingestion/test_coverage_client.py`

- [x] **Step 1: Write tests for pano ID extraction**

Create `services/main/tests/ingestion/test_coverage_client.py`:

```python
from types import SimpleNamespace

from main_service.ingestion.coverage_client import pano_ids_from_coverage_objects


def test_pano_ids_from_coverage_objects_deduplicates_and_sorts() -> None:
    coverage_objects = [
        SimpleNamespace(id="pano-b"),
        SimpleNamespace(id="pano-a"),
        SimpleNamespace(id="pano-b"),
        SimpleNamespace(id=None),
    ]

    pano_ids = pano_ids_from_coverage_objects(coverage_objects)

    assert [pano_id.value for pano_id in pano_ids] == ["pano-a", "pano-b"]
```

- [x] **Step 2: Run tests and verify red**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_coverage_client.py -q
```

Expected: fail because `coverage_client.py` does not exist.

- [x] **Step 3: Implement coverage client abstraction**

Create `services/main/main_service/ingestion/coverage_client.py`:

```python
from typing import Protocol

from streetlevel import streetview

from main_service.ingestion.types import MapTileKey, PanoramaId


class CoverageClient(Protocol):
    def get_pano_ids_for_tile(self, tile: MapTileKey) -> list[PanoramaId]:
        ...


def pano_ids_from_coverage_objects(coverage_objects: list[object]) -> list[PanoramaId]:
    ids = {
        str(raw_id)
        for item in coverage_objects
        if (raw_id := getattr(item, "id", None)) is not None
    }
    return [PanoramaId(value=raw_id) for raw_id in sorted(ids)]


class StreetLevelCoverageClient:
    def get_pano_ids_for_tile(self, tile: MapTileKey) -> list[PanoramaId]:
        coverage_objects = streetview.get_coverage_tile(tile.x, tile.y)
        return pano_ids_from_coverage_objects(coverage_objects)
```

- [x] **Step 4: Run tests and verify green**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_coverage_client.py -q
```

Expected: `1 passed`.

- [x] **Step 5: Commit**

Run:

```bash
git add services/main/main_service/ingestion/coverage_client.py services/main/tests/ingestion/test_coverage_client.py
git commit -m "feat: add coverage client abstraction"
```

## Task 6: Add Downloader Queue Abstraction

**Files:**

- Create: `services/main/main_service/ingestion/download_queue.py`
- Create: `services/main/tests/ingestion/test_download_queue.py`

- [x] **Step 1: Write tests for queue messages and in-memory fake**

Create `services/main/tests/ingestion/test_download_queue.py`:

```python
from main_service.ingestion.download_queue import (
    InMemoryPanoDownloadQueue,
    PanoDownloadMessage,
)
from main_service.ingestion.types import MapTileKey, PanoramaId


def test_in_memory_queue_tracks_pending_count() -> None:
    queue = InMemoryPanoDownloadQueue()
    message = PanoDownloadMessage(
        pano_id=PanoramaId("pano-a"),
        source_tile=MapTileKey(x=1, y=2, z=17),
    )

    queue.enqueue(message)

    assert queue.pending_count() == 1
    assert queue.messages == [message]


def test_download_message_serializes_to_public_safe_payload() -> None:
    message = PanoDownloadMessage(
        pano_id=PanoramaId("pano-a"),
        source_tile=MapTileKey(x=1, y=2, z=17),
    )

    assert message.to_dict() == {
        "pano_id": "pano-a",
        "source": "coverage_discovery",
        "discovered_from_tile": {"x": 1, "y": 2, "z": 17},
    }
```

- [x] **Step 2: Run tests and verify red**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_download_queue.py -q
```

Expected: fail because `download_queue.py` does not exist.

- [x] **Step 3: Implement queue abstraction**

Create `services/main/main_service/ingestion/download_queue.py`:

```python
from dataclasses import dataclass
from typing import Protocol

from main_service.ingestion.types import MapTileKey, PanoramaId


@dataclass(frozen=True)
class PanoDownloadMessage:
    pano_id: PanoramaId
    source_tile: MapTileKey

    def to_dict(self) -> dict[str, object]:
        return {
            "pano_id": self.pano_id.value,
            "source": "coverage_discovery",
            "discovered_from_tile": {
                "x": self.source_tile.x,
                "y": self.source_tile.y,
                "z": self.source_tile.z,
            },
        }


class PanoDownloadQueue(Protocol):
    def pending_count(self) -> int:
        ...

    def enqueue(self, message: PanoDownloadMessage) -> None:
        ...


class InMemoryPanoDownloadQueue:
    def __init__(self) -> None:
        self.messages: list[PanoDownloadMessage] = []

    def pending_count(self) -> int:
        return len(self.messages)

    def enqueue(self, message: PanoDownloadMessage) -> None:
        self.messages.append(message)
```

Do not implement the RabbitMQ class in this milestone; keep the interface ready for it.

- [x] **Step 4: Run tests and verify green**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_download_queue.py -q
```

Expected: `2 passed`.

- [x] **Step 5: Commit**

Run:

```bash
git add services/main/main_service/ingestion/download_queue.py services/main/tests/ingestion/test_download_queue.py
git commit -m "feat: add pano download queue abstraction"
```

## Task 7: Add Boundary-to-Pano Discovery Orchestration

**Files:**

- Create: `services/main/main_service/ingestion/discovery.py`
- Create: `services/main/tests/ingestion/test_discovery.py`

- [ ] **Step 1: Write tests with fake coverage client**

Create `services/main/tests/ingestion/test_discovery.py`:

```python
from sqlalchemy import create_engine

from main_service.db.models import embedding, map_tile, map_tile_panorama, panorama, tile
from main_service.db.models.base import Base
from main_service.db.services.panorama_service import PanoramaService
from main_service.ingestion.discovery import discover_panos_for_tiles
from main_service.ingestion.download_queue import InMemoryPanoDownloadQueue
from main_service.ingestion.types import MapTileKey, PanoramaId


class FakeCoverageClient:
    def __init__(self) -> None:
        self.calls: list[MapTileKey] = []

    def get_pano_ids_for_tile(self, tile: MapTileKey) -> list[PanoramaId]:
        self.calls.append(tile)
        if tile.x == 1:
            return [PanoramaId("pano-a"), PanoramaId("pano-b")]
        return [PanoramaId("pano-b"), PanoramaId("pano-c")]


def make_service() -> PanoramaService:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return PanoramaService(engine)


def test_discover_panos_for_tiles_persists_tiles_panos_and_links() -> None:
    service = make_service()
    coverage_client = FakeCoverageClient()
    download_queue = InMemoryPanoDownloadQueue()
    tiles = [MapTileKey(x=1, y=10, z=17), MapTileKey(x=2, y=10, z=17)]

    result = discover_panos_for_tiles(
        service,
        coverage_client,
        download_queue,
        tiles,
        max_downloader_queue_depth=1000,
    )

    assert result.tiles_processed == 2
    assert result.unique_panos_discovered == 3
    assert result.tile_pano_links_created == 4
    assert result.enqueued_downloads == 4
    assert result.paused is False
    assert coverage_client.calls == tiles
    assert download_queue.pending_count() == 4


def test_discovery_pauses_before_next_tile_when_downloader_queue_is_full() -> None:
    service = make_service()
    coverage_client = FakeCoverageClient()
    download_queue = InMemoryPanoDownloadQueue()
    tiles = [MapTileKey(x=1, y=10, z=17), MapTileKey(x=2, y=10, z=17)]

    result = discover_panos_for_tiles(
        service,
        coverage_client,
        download_queue,
        tiles,
        max_downloader_queue_depth=1,
    )

    assert result.tiles_processed == 1
    assert result.paused is True
    assert result.pause_reason == "downloader_queue_full"
    assert coverage_client.calls == [tiles[0]]
    assert download_queue.pending_count() == 2
```

- [ ] **Step 2: Run tests and verify red**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_discovery.py -q
```

Expected: fail because `discovery.py` does not exist.

- [ ] **Step 3: Implement discovery orchestration**

Create `services/main/main_service/ingestion/discovery.py`:

```python
from dataclasses import dataclass
from typing import Sequence

from main_service.db.services.panorama_service import PanoramaService
from main_service.ingestion.coverage_client import CoverageClient
from main_service.ingestion.download_queue import PanoDownloadMessage, PanoDownloadQueue
from main_service.ingestion.types import MapTileKey


@dataclass(frozen=True)
class DiscoveryResult:
    tiles_processed: int
    unique_panos_discovered: int
    tile_pano_links_created: int
    enqueued_downloads: int
    paused: bool = False
    pause_reason: str | None = None


def discover_panos_for_tiles(
    pano_service: PanoramaService,
    coverage_client: CoverageClient,
    download_queue: PanoDownloadQueue,
    tiles: Sequence[MapTileKey],
    max_downloader_queue_depth: int,
) -> DiscoveryResult:
    unique_pano_ids: set[str] = set()
    link_count = 0
    enqueued_count = 0
    tiles_processed = 0

    for tile in tiles:
        if download_queue.pending_count() >= max_downloader_queue_depth:
            return DiscoveryResult(
                tiles_processed=tiles_processed,
                unique_panos_discovered=len(unique_pano_ids),
                tile_pano_links_created=link_count,
                enqueued_downloads=enqueued_count,
                paused=True,
                pause_reason="downloader_queue_full",
            )

        tile_row = pano_service.upsert_map_tile(tile)
        pano_ids = coverage_client.get_pano_ids_for_tile(tile)
        for pano_id in pano_ids:
            pano_row = pano_service.upsert_discovered_panorama(pano_id)
            created = pano_service.link_map_tile_to_panorama(tile_row.id, pano_row.id)
            unique_pano_ids.add(pano_id.value)
            if created:
                link_count += 1
            download_queue.enqueue(PanoDownloadMessage(pano_id=pano_id, source_tile=tile))
            pano_service.mark_panorama_download_queued(pano_row.id)
            enqueued_count += 1
        pano_service.mark_map_tile_discovery_complete(tile_row.id)
        tiles_processed += 1

    return DiscoveryResult(
        tiles_processed=tiles_processed,
        unique_panos_discovered=len(unique_pano_ids),
        tile_pano_links_created=link_count,
        enqueued_downloads=enqueued_count,
    )
```

Adjust `PanoramaService.link_map_tile_to_panorama` to return `bool`: `True` when it creates a new row, `False` when the link already exists.

- [ ] **Step 4: Run tests and verify green**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_discovery.py tests/db/test_discovery_service.py tests/ingestion/test_download_queue.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add services/main/main_service/ingestion/discovery.py services/main/main_service/db/services/panorama_service.py services/main/tests/ingestion/test_discovery.py services/main/tests/db/test_discovery_service.py
git commit -m "feat: discover pano ids for map tiles"
```

## Task 8: Milestone Verification

**Files:**

- No direct source changes expected.

- [ ] **Step 1: Run all tests**

Run:

```bash
cd services/main
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run import smoke check**

Run:

```bash
cd services/main
.venv/bin/python -B -c "import main_service.ingestion.boundary_loader; import main_service.ingestion.coverage_client; import main_service.ingestion.download_queue; import main_service.ingestion.discovery; print('coverage discovery imports ok')"
```

Expected output:

```text
coverage discovery imports ok
```

- [ ] **Step 3: Check whitespace and local status**

Run from repo root:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors and no uncommitted changes after task commits.

## Self-Review

- Spec coverage: this plan covers boundary loading, coverage-tile generation, pano ID discovery per tile, DB persistence, statuses, idempotent links, queue handoff, and backpressure before starting the next tile.
- Placeholder scan: no `TBD` or private values are present; examples use public-safe placeholder strings.
- Type consistency: `MapTileKey`, `PanoramaId`, `ProcessingStatus`, and `DownloadStatus` are introduced before use.
- Scope check: downloads, metadata enrichment, view generation, embeddings, HNSW, and website work are intentionally deferred.
