# Ingestion State Foundation Implementation Plan

> Superseded for immediate execution by `.codex/plans/2026-05-28-boundary-to-pano-discovery.md`. Keep this document as background for the smaller state-foundation slice, but use the boundary-to-pano discovery plan as the next milestone because it better matches the current product deliverable.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the durable ingestion foundation needed before live Street View discovery or panorama downloading: domain types, boundary loading, deterministic image paths, and relational state for map tiles and panoramas.

**Architecture:** Keep this milestone offline and deterministic. Add small ingestion modules with pure functions/classes, then add SQLAlchemy models and service methods that can be tested with in-memory SQLite. Live Google calls, image downloads, view generation, embeddings, HNSW, and website work are outside this milestone.

**Tech Stack:** Python 3.14, uv, pytest, SQLAlchemy, Pydantic Settings, mercantile, shapely.

---

## Git Rules

- Commit coherent local changes as work progresses.
- Use Conventional Commits format for every commit.
- Never run remote/origin operations: no `git push`, `git pull`, `git fetch`, commands targeting `origin`, or remote configuration changes.
- This is a public repo. Never commit real API keys, tokens, credentials, private endpoints, private filesystem paths, personal names, private organization names, generated private artifacts, downloaded panoramas, vectors, or indexes.

## Scope

In scope:

- Ingestion domain types and stable status values.
- Config fields for ingestion defaults.
- Loading map tile keys from a GeoJSON boundary using existing `main_service.geo`.
- Deterministic local relative paths for future panorama and generated-view files.
- SQLAlchemy persistence for map tiles, panoramas, and map-tile-to-panorama discovery links.
- Service methods for idempotent upserts and link creation.
- Tests that run without live network calls.

Out of scope:

- `streetlevel` coverage calls.
- Panorama metadata fetching from Google.
- Panorama image downloads.
- Panorama view generation.
- Embeddings, HNSW, querying, public APIs, and website UI.
- Multi-machine queues or distributed workers.

## File Map

- Create `services/main/main_service/ingestion/__init__.py`: package marker.
- Create `services/main/main_service/ingestion/types.py`: `ProcessingStatus`, `MapTileKey`, `PanoramaMetadata`, and `StoredImagePath`.
- Create `services/main/main_service/ingestion/boundary_loader.py`: reads GeoJSON and returns sorted `MapTileKey` values.
- Create `services/main/main_service/ingestion/image_store.py`: produces deterministic relative paths under a configured data root.
- Modify `services/main/main_service/config.py`: add ingestion defaults.
- Create `services/main/main_service/db/models/map_tile.py`: coverage tile state table.
- Create `services/main/main_service/db/models/map_tile_panorama.py`: many-to-many discovery link table.
- Modify `services/main/main_service/db/models/panorama.py`: add ingestion state fields.
- Modify `services/main/main_service/db/models/__init__.py`: import all model modules.
- Modify `services/main/main_service/db/initialize_engine.py`: include new models in metadata.
- Modify `services/main/main_service/db/services/panorama_service.py`: add idempotent ingestion service methods.
- Create `services/main/tests/ingestion/`: offline ingestion module tests.
- Create `services/main/tests/db/`: in-memory SQLite persistence tests.

## Task 1: Add Ingestion Domain Types

**Files:**

- Create: `services/main/main_service/ingestion/__init__.py`
- Create: `services/main/main_service/ingestion/types.py`
- Create: `services/main/tests/ingestion/test_types.py`

- [ ] **Step 1: Write failing tests**

Create `services/main/tests/ingestion/test_types.py`:

```python
from main_service.ingestion.types import (
    MapTileKey,
    PanoramaMetadata,
    ProcessingStatus,
    StoredImagePath,
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


def test_panorama_metadata_allows_missing_capture_date() -> None:
    metadata = PanoramaMetadata(
        orig_id="example-pano-id",
        latitude=37.0,
        longitude=-122.0,
        capture_year=None,
        capture_month=None,
    )

    assert metadata.capture_year is None
    assert metadata.capture_month is None


def test_stored_image_path_is_relative_string() -> None:
    stored_path = StoredImagePath(value="data/panoramas/example-pano-id.jpg")

    assert stored_path.value == "data/panoramas/example-pano-id.jpg"
```

- [ ] **Step 2: Run tests and verify red**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_types.py -q
```

Expected: fail during import because `main_service.ingestion` does not exist.

- [ ] **Step 3: Implement domain types**

Create `services/main/main_service/ingestion/__init__.py` as an empty file.

Create `services/main/main_service/ingestion/types.py`:

```python
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional


class ProcessingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
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
class PanoramaMetadata:
    orig_id: str
    latitude: float
    longitude: float
    capture_year: Optional[int]
    capture_month: Optional[int]


@dataclass(frozen=True)
class StoredImagePath:
    value: str
```

- [ ] **Step 4: Run tests and verify green**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_types.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add services/main/main_service/ingestion services/main/tests/ingestion/test_types.py
git commit -m "feat: add ingestion domain types"
```

## Task 2: Add Boundary Loader and Ingestion Config

**Files:**

- Modify: `services/main/main_service/config.py`
- Create: `services/main/main_service/ingestion/boundary_loader.py`
- Create: `services/main/tests/ingestion/test_boundary_loader.py`

- [ ] **Step 1: Write failing tests**

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


def test_settings_exposes_ingestion_defaults() -> None:
    settings = Settings(
        _env_file=None,
        db_user="user",
        db_password="password",
        db_host="localhost",
        db_port=3306,
        db_name="sf_search",
    )

    assert settings.map_tiles_zoom == 17
    assert settings.ingestion_data_dir == "data"
    assert settings.coverage_concurrency == 20
    assert settings.download_concurrency == 5
    assert settings.max_attempts == 5
```

- [ ] **Step 2: Run tests and verify red**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_boundary_loader.py -q
```

Expected: fail because `main_service.ingestion.boundary_loader` does not exist.

- [ ] **Step 3: Implement boundary loader**

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

- [ ] **Step 4: Add ingestion config fields**

In `services/main/main_service/config.py`, add these fields to `Settings`:

```python
    map_tiles_zoom: int = Field(default=17)
    ingestion_data_dir: str = Field(default="data")
    coverage_concurrency: int = Field(default=20)
    download_concurrency: int = Field(default=5)
    max_attempts: int = Field(default=5)
```

- [ ] **Step 5: Run tests and verify green**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_boundary_loader.py tests/test_geo.py tests/test_config.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add services/main/main_service/config.py services/main/main_service/ingestion/boundary_loader.py services/main/tests/ingestion/test_boundary_loader.py
git commit -m "feat: load ingestion tiles from boundary"
```

## Task 3: Add Deterministic Image Store Paths

**Files:**

- Create: `services/main/main_service/ingestion/image_store.py`
- Create: `services/main/tests/ingestion/test_image_store.py`

- [ ] **Step 1: Write failing tests**

Create `services/main/tests/ingestion/test_image_store.py`:

```python
from pathlib import Path

from main_service.ingestion.image_store import ImageStore
from main_service.processing.tiling import TileSpec


def test_panorama_path_is_deterministic_relative_path() -> None:
    store = ImageStore(Path("data"))

    path = store.panorama_path("example-pano-id")

    assert path == Path("data/panoramas/example-pano-id.jpg")


def test_view_path_includes_view_spec() -> None:
    store = ImageStore(Path("data"))
    spec = TileSpec(yaw=90, pitch=10, roll=0, fov=70, width=1000, height=1000)

    path = store.view_path("example-pano-id", spec)

    assert path == Path(
        "data/views/example-pano-id/"
        "yaw_90_pitch_10_roll_0_fov_70_w_1000_h_1000.jpg"
    )
```

- [ ] **Step 2: Run tests and verify red**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_image_store.py -q
```

Expected: fail because `main_service.ingestion.image_store` does not exist.

- [ ] **Step 3: Implement image store**

Create `services/main/main_service/ingestion/image_store.py`:

```python
from pathlib import Path

from main_service.processing.tiling import TileSpec


class ImageStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def panorama_path(self, orig_id: str) -> Path:
        return self._root / "panoramas" / f"{orig_id}.jpg"

    def view_path(self, orig_id: str, spec: TileSpec) -> Path:
        filename = (
            f"yaw_{spec.yaw:g}_pitch_{spec.pitch:g}_roll_{spec.roll:g}_"
            f"fov_{spec.fov:g}_w_{spec.width}_h_{spec.height}.jpg"
        )
        return self._root / "views" / orig_id / filename
```

- [ ] **Step 4: Run tests and verify green**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_image_store.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add services/main/main_service/ingestion/image_store.py services/main/tests/ingestion/test_image_store.py
git commit -m "feat: add ingestion image store paths"
```

## Task 4: Add Ingestion Persistence Models

**Files:**

- Create: `services/main/main_service/db/models/map_tile.py`
- Create: `services/main/main_service/db/models/map_tile_panorama.py`
- Modify: `services/main/main_service/db/models/__init__.py`
- Modify: `services/main/main_service/db/models/panorama.py`
- Modify: `services/main/main_service/db/initialize_engine.py`
- Create: `services/main/tests/db/test_ingestion_models.py`

- [ ] **Step 1: Write failing model metadata tests**

Create `services/main/tests/db/test_ingestion_models.py`:

```python
from sqlalchemy import create_engine, inspect

from main_service.db.models.base import Base
from main_service.db.models.map_tile import MapTile
from main_service.db.models.map_tile_panorama import MapTilePanorama
from main_service.db.models.panorama import Panorama
from main_service.ingestion.types import ProcessingStatus


def test_ingestion_tables_exist_in_metadata() -> None:
    assert "map_tile_table" in Base.metadata.tables
    assert "map_tile_panorama_table" in Base.metadata.tables


def test_map_tile_has_unique_tile_key() -> None:
    constraint_columns = {
        tuple(column.name for column in constraint.columns)
        for constraint in MapTile.__table__.constraints
        if constraint.name == "uq_map_tile_xyz"
    }

    assert ("x", "y", "z") in constraint_columns


def test_map_tile_panorama_has_unique_link_key() -> None:
    constraint_columns = {
        tuple(column.name for column in constraint.columns)
        for constraint in MapTilePanorama.__table__.constraints
        if constraint.name == "uq_map_tile_panorama"
    }

    assert ("map_tile_id", "panorama_id") in constraint_columns


def test_panorama_has_ingestion_state_columns() -> None:
    columns = Panorama.__table__.c

    assert "metadata_status" in columns
    assert "download_status" in columns
    assert "image_path" in columns
    assert "capture_year" in columns
    assert "capture_month" in columns


def test_metadata_can_create_sqlite_schema() -> None:
    engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(engine)

    assert inspect(engine).has_table("map_tile_table") is True
```

- [ ] **Step 2: Run tests and verify red**

Run:

```bash
cd services/main
uv run pytest tests/db/test_ingestion_models.py -q
```

Expected: fail because `map_tile` models do not exist.

- [ ] **Step 3: Implement map tile model**

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
    status: Mapped[str] = mapped_column(default=ProcessingStatus.PENDING.value)
    attempt_count: Mapped[int] = mapped_column(default=0)
    last_error: Mapped[str | None]
```

- [ ] **Step 4: Implement discovery link model**

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

- [ ] **Step 5: Extend panorama model**

In `services/main/main_service/db/models/panorama.py`, add:

```python
from main_service.ingestion.types import ProcessingStatus
```

and these mapped fields inside `Panorama`:

```python
    capture_year: Mapped[int | None]
    capture_month: Mapped[int | None]
    metadata_status: Mapped[str] = mapped_column(default=ProcessingStatus.PENDING.value)
    download_status: Mapped[str] = mapped_column(default=ProcessingStatus.PENDING.value)
    image_path: Mapped[str | None]
    attempt_count: Mapped[int] = mapped_column(default=0)
    last_error: Mapped[str | None]
```

- [ ] **Step 6: Import model modules**

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

Update `services/main/main_service/db/initialize_engine.py` so the model import line is:

```python
from main_service.db.models import embedding, map_tile, map_tile_panorama, panorama, tile
```

- [ ] **Step 7: Run tests and verify green**

Run:

```bash
cd services/main
uv run pytest tests/db/test_ingestion_models.py tests/test_models.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

Run:

```bash
git add services/main/main_service/db services/main/tests/db/test_ingestion_models.py
git commit -m "feat: add ingestion persistence models"
```

## Task 5: Add Idempotent Persistence Service Methods

**Files:**

- Modify: `services/main/main_service/db/services/panorama_service.py`
- Create: `services/main/tests/db/test_ingestion_service.py`

- [ ] **Step 1: Write failing service tests**

Create `services/main/tests/db/test_ingestion_service.py`:

```python
from sqlalchemy import create_engine

from main_service.db.models import embedding, map_tile, map_tile_panorama, panorama, tile
from main_service.db.models.base import Base
from main_service.db.services.panorama_service import PanoramaService
from main_service.ingestion.types import MapTileKey, PanoramaMetadata, ProcessingStatus


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
    assert first.status == ProcessingStatus.PENDING.value


def test_upsert_panorama_metadata_is_idempotent() -> None:
    service = make_service()
    metadata = PanoramaMetadata(
        orig_id="example-pano-id",
        latitude=37.0,
        longitude=-122.0,
        capture_year=2024,
        capture_month=5,
    )

    first = service.upsert_panorama_metadata(metadata)
    second = service.upsert_panorama_metadata(metadata)

    assert first.id == second.id
    assert first.orig_id == "example-pano-id"
    assert first.capture_year == 2024
    assert first.metadata_status == ProcessingStatus.COMPLETE.value


def test_link_map_tile_to_panorama_is_idempotent() -> None:
    service = make_service()
    tile_row = service.upsert_map_tile(MapTileKey(x=1, y=2, z=17))
    pano_row = service.upsert_panorama_metadata(
        PanoramaMetadata(
            orig_id="example-pano-id",
            latitude=37.0,
            longitude=-122.0,
            capture_year=None,
            capture_month=None,
        )
    )

    service.link_map_tile_to_panorama(tile_row.id, pano_row.id)
    service.link_map_tile_to_panorama(tile_row.id, pano_row.id)

    assert service.count_map_tile_panorama_links() == 1
```

- [ ] **Step 2: Run tests and verify red**

Run:

```bash
cd services/main
uv run pytest tests/db/test_ingestion_service.py -q
```

Expected: fail because the service methods do not exist.

- [ ] **Step 3: Implement service methods**

Add imports to `services/main/main_service/db/services/panorama_service.py`:

```python
from main_service.db.models.map_tile import MapTile
from main_service.db.models.map_tile_panorama import MapTilePanorama
from main_service.ingestion.types import MapTileKey, PanoramaMetadata, ProcessingStatus
```

Add these methods to `PanoramaService`:

```python
    def upsert_map_tile(self, key: MapTileKey) -> MapTile:
        with Session(self.engine) as session:
            found = session.execute(
                select(MapTile).filter_by(x=key.x, y=key.y, z=key.z)
            ).scalar_one_or_none()
            if found is not None:
                session.expunge(found)
                return found

            created = MapTile(x=key.x, y=key.y, z=key.z)
            session.add(created)
            session.commit()
            session.refresh(created)
            session.expunge(created)
            return created

    def upsert_panorama_metadata(self, metadata: PanoramaMetadata) -> Panorama:
        with Session(self.engine) as session:
            found = session.execute(
                select(Panorama).filter_by(orig_id=metadata.orig_id)
            ).scalar_one_or_none()
            if found is None:
                found = Panorama(
                    orig_id=metadata.orig_id,
                    image_hash=None,
                    latitude=metadata.latitude,
                    longitude=metadata.longitude,
                    capture_year=metadata.capture_year,
                    capture_month=metadata.capture_month,
                    metadata_status=ProcessingStatus.COMPLETE.value,
                )
                session.add(found)
            else:
                found.latitude = metadata.latitude
                found.longitude = metadata.longitude
                found.capture_year = metadata.capture_year
                found.capture_month = metadata.capture_month
                found.metadata_status = ProcessingStatus.COMPLETE.value

            session.commit()
            session.refresh(found)
            session.expunge(found)
            return found

    def link_map_tile_to_panorama(self, map_tile_id: int, panorama_id: int) -> None:
        with Session(self.engine) as session:
            found = session.execute(
                select(MapTilePanorama).filter_by(
                    map_tile_id=map_tile_id,
                    panorama_id=panorama_id,
                )
            ).scalar_one_or_none()
            if found is not None:
                return

            session.add(
                MapTilePanorama(
                    map_tile_id=map_tile_id,
                    panorama_id=panorama_id,
                )
            )
            session.commit()

    def count_map_tile_panorama_links(self) -> int:
        with Session(self.engine) as session:
            return len(session.execute(select(MapTilePanorama)).scalars().all())
```

- [ ] **Step 4: Run tests and verify green**

Run:

```bash
cd services/main
uv run pytest tests/db/test_ingestion_service.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add services/main/main_service/db/services/panorama_service.py services/main/tests/db/test_ingestion_service.py
git commit -m "feat: add ingestion persistence service"
```

## Task 6: Full Milestone Verification

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
.venv/bin/python -B -c "import main_service.ingestion.boundary_loader; import main_service.ingestion.image_store; import main_service.db.models.map_tile; import main_service.db.models.map_tile_panorama; print('ingestion foundation imports ok')"
```

Expected output:

```text
ingestion foundation imports ok
```

- [ ] **Step 3: Check whitespace and worktree**

Run from repo root:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors and no uncommitted changes after task commits.

## Self-Review

- Spec coverage: this milestone covers the storage/state foundation needed for the ingestion spec through map tile and panorama discovery state. Live network discovery and downloads are intentionally deferred.
- Placeholder scan: no steps include incomplete placeholders or require private values.
- Type consistency: `MapTileKey`, `PanoramaMetadata`, `StoredImagePath`, and `ProcessingStatus` are introduced before being used by boundary loading and persistence tasks.
- Scope check: this plan is small enough to produce working, testable software on its own without live Google calls.
