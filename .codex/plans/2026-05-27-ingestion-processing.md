# Ingestion and Panorama Processing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first resumable backend pipeline that turns a GeoJSON boundary into discovered panorama IDs, downloaded panorama files, and generated panorama view metadata.

**Architecture:** Keep the first version as a single long-running worker with relational state and clean internal boundaries. Street View access sits behind an adapter, persistence sits behind service methods, and orchestration moves rows through explicit statuses. Embeddings, HNSW, querying, and website work are outside this plan.

**Tech Stack:** Python 3.14, uv, pytest, SQLAlchemy, Pydantic Settings, mercantile, shapely, streetlevel, aiohttp, Pillow or OpenCV, py360convert, shared logging.

---

## Git Rules

- Commit coherent local changes as work progresses.
- Use Conventional Commits format for every commit, such as `feat: add ingestion domain types` or `test: add boundary loader coverage`.
- Never run remote/origin operations: no `git push`, `git pull`, `git fetch`, commands targeting `origin`, or remote configuration changes.
- This is a public repo. Never commit real API keys, tokens, credentials, private endpoints, private filesystem paths, personal names, private organization names, generated private artifacts, downloaded panoramas, vectors, or indexes.

## Scope

In scope:

- Boundary to map-tile persistence.
- Street View coverage discovery.
- Panorama metadata/download state.
- Panorama image download.
- Panorama view generation and view metadata persistence.
- Resume/retry behavior.
- Unit tests with fake Street View clients and synthetic images.

Out of scope:

- CLIP/OpenCLIP embeddings.
- HNSW index creation or querying.
- Website, public API, and UI.
- Multi-machine queue infrastructure.

## File Map

- `services/main/main_service/geo.py`: keep pure boundary-to-map-tile generation.
- `services/main/main_service/config.py`: add ingestion settings such as zoom, data directory, concurrency, and retry limits.
- `services/main/main_service/ingestion/types.py`: shared enums and dataclasses for statuses, map tile keys, panorama metadata, and download results.
- `services/main/main_service/ingestion/streetview_client.py`: adapter around `streetlevel.streetview` coverage, metadata, and download calls.
- `services/main/main_service/ingestion/image_store.py`: deterministic filesystem paths for panorama images and generated views.
- `services/main/main_service/ingestion/boundary_loader.py`: load configured GeoJSON and generate map tile keys.
- `services/main/main_service/ingestion/orchestrator.py`: run discovery, download, and view-generation loops.
- `services/main/main_service/processing/tiling.py`: generate perspective views from equirectangular panoramas.
- `services/main/main_service/db/models/map_tile.py`: map coverage tile model.
- `services/main/main_service/db/models/panorama.py`: extend existing panorama model with metadata/download state.
- `services/main/main_service/db/models/tile.py`: either rename later or repurpose carefully as panorama view metadata.
- `services/main/main_service/db/models/map_tile_panorama.py`: many-to-many discovery link.
- `services/main/main_service/db/services/panorama_service.py`: add upsert and state-transition methods.
- `services/main/tests/`: add tests for each boundary without live Google calls.

## Task 1: Prepare Repo Baseline

**Files:**

- Follow: `.codex/plans/2026-05-27-repo-cleanup-baseline.md`

- [ ] **Step 1: Complete the cleanup baseline plan**

Run through `.codex/plans/2026-05-27-repo-cleanup-baseline.md` before implementing this plan.

Expected: `services/main` imports cleanly, basic tests exist, and `PipelineManager` no longer has syntax errors.

- [ ] **Step 2: Verify baseline**

Run:

```bash
cd services/main
uv run pytest -q
```

Expected: all baseline tests pass.

## Task 2: Add Ingestion Types

**Files:**

- Create: `services/main/main_service/ingestion/__init__.py`
- Create: `services/main/main_service/ingestion/types.py`
- Create: `services/main/tests/ingestion/test_types.py`

- [ ] **Step 1: Add status and DTO tests**

Create `services/main/tests/ingestion/test_types.py`:

```python
from main_service.ingestion.types import (
    DownloadResult,
    MapTileKey,
    PanoramaMetadata,
    ProcessingStatus,
)


def test_map_tile_key_tuple_round_trip() -> None:
    key = MapTileKey(x=21003, y=50607, z=17)

    assert key.as_tuple() == (21003, 50607, 17)


def test_processing_status_values_are_stable() -> None:
    assert ProcessingStatus.PENDING.value == "pending"
    assert ProcessingStatus.PROCESSING.value == "processing"
    assert ProcessingStatus.COMPLETE.value == "complete"
    assert ProcessingStatus.FAILED.value == "failed"
    assert ProcessingStatus.SKIPPED.value == "skipped"


def test_panorama_metadata_allows_missing_capture_date() -> None:
    metadata = PanoramaMetadata(
        orig_id="Ai1Eh2D14bkcmF0rTDtStA",
        latitude=37.78,
        longitude=-122.41,
        capture_year=None,
        capture_month=None,
    )

    assert metadata.capture_year is None
    assert metadata.capture_month is None


def test_download_result_records_output_path() -> None:
    result = DownloadResult(
        orig_id="Ai1Eh2D14bkcmF0rTDtStA",
        image_path="data/panoramas/Ai1Eh2D14bkcmF0rTDtStA.jpg",
    )

    assert result.image_path.endswith(".jpg")
```

- [ ] **Step 2: Implement ingestion types**

Create `services/main/main_service/ingestion/__init__.py` as an empty package marker.

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
class DownloadResult:
    orig_id: str
    image_path: str
```

- [ ] **Step 3: Run tests**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_types.py -q
```

Expected: `4 passed`.

- [ ] **Step 4: Commit**

Run:

```bash
git add services/main/main_service/ingestion services/main/tests/ingestion/test_types.py
git commit -m "feat: add ingestion domain types"
```

## Task 3: Add Boundary Loader

**Files:**

- Modify: `services/main/main_service/config.py`
- Create: `services/main/main_service/ingestion/boundary_loader.py`
- Create: `services/main/tests/ingestion/test_boundary_loader.py`

- [ ] **Step 1: Add tests for loading configured boundary tiles**

Create `services/main/tests/ingestion/test_boundary_loader.py`:

```python
from pathlib import Path

from main_service.ingestion.boundary_loader import load_map_tiles_from_geojson


def test_load_map_tiles_from_geojson_uses_existing_geo_logic() -> None:
    tiles = load_map_tiles_from_geojson(Path("target.geojson"), zoom=17)

    assert len(tiles) == 598
    assert tiles[0].z == 17
    assert (tiles[0].x, tiles[0].y) <= (tiles[-1].x, tiles[-1].y)
```

- [ ] **Step 2: Implement boundary loader**

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

- [ ] **Step 3: Add ingestion config fields**

In `services/main/main_service/config.py`, add fields:

```python
    map_tiles_zoom: int = Field(default=17)
    ingestion_data_dir: str = Field(default="data")
    coverage_concurrency: int = Field(default=20)
    download_concurrency: int = Field(default=5)
    max_attempts: int = Field(default=5)
```

- [ ] **Step 4: Run tests**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_boundary_loader.py tests/test_geo.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add services/main/main_service/config.py services/main/main_service/ingestion/boundary_loader.py services/main/tests/ingestion/test_boundary_loader.py
git commit -m "feat: load ingestion map tiles from boundary"
```

## Task 4: Add Street View Client Adapter

**Files:**

- Create: `services/main/main_service/ingestion/streetview_client.py`
- Create: `services/main/tests/ingestion/test_streetview_client.py`

- [ ] **Step 1: Add tests for metadata extraction**

Create `services/main/tests/ingestion/test_streetview_client.py`:

```python
from types import SimpleNamespace

from main_service.ingestion.streetview_client import panorama_metadata_from_object


def test_panorama_metadata_from_direct_lat_lon_fields() -> None:
    pano = SimpleNamespace(
        id="Ai1Eh2D14bkcmF0rTDtStA",
        lat=37.78,
        lon=-122.41,
        date=SimpleNamespace(year=2024, month=5),
    )

    metadata = panorama_metadata_from_object(pano)

    assert metadata.orig_id == "Ai1Eh2D14bkcmF0rTDtStA"
    assert metadata.latitude == 37.78
    assert metadata.longitude == -122.41
    assert metadata.capture_year == 2024
    assert metadata.capture_month == 5


def test_panorama_metadata_from_nested_location_fields() -> None:
    pano = SimpleNamespace(
        id="Ai1Eh2D14bkcmF0rTDtStA",
        location=SimpleNamespace(lat=37.78, lon=-122.41),
        date=None,
    )

    metadata = panorama_metadata_from_object(pano)

    assert metadata.latitude == 37.78
    assert metadata.longitude == -122.41
    assert metadata.capture_year is None
    assert metadata.capture_month is None
```

- [ ] **Step 2: Implement adapter and extraction helper**

Create `services/main/main_service/ingestion/streetview_client.py`:

```python
from pathlib import Path
from typing import Protocol

from aiohttp import ClientSession
from streetlevel import streetview

from main_service.ingestion.types import DownloadResult, MapTileKey, PanoramaMetadata


class StreetViewClient(Protocol):
    async def get_coverage(self, tile: MapTileKey) -> list[str]:
        ...

    async def get_metadata(self, orig_id: str) -> PanoramaMetadata:
        ...

    async def download_panorama(self, orig_id: str, output_path: Path) -> DownloadResult:
        ...


def panorama_metadata_from_object(pano: object) -> PanoramaMetadata:
    orig_id = str(getattr(pano, "id"))
    lat = getattr(pano, "lat", None)
    lon = getattr(pano, "lon", None)
    location = getattr(pano, "location", None)
    if lat is None and location is not None:
        lat = getattr(location, "lat", None)
    if lon is None and location is not None:
        lon = getattr(location, "lon", None)
    if lat is None or lon is None:
        raise ValueError(f"Panorama {orig_id!r} does not include coordinates")

    date = getattr(pano, "date", None)
    capture_year = getattr(date, "year", None) if date is not None else None
    capture_month = getattr(date, "month", None) if date is not None else None
    return PanoramaMetadata(
        orig_id=orig_id,
        latitude=float(lat),
        longitude=float(lon),
        capture_year=capture_year,
        capture_month=capture_month,
    )


class StreetLevelClient:
    def __init__(self, session: ClientSession) -> None:
        self._session = session

    async def get_coverage(self, tile: MapTileKey) -> list[str]:
        panos = streetview.get_coverage_tile(tile.x, tile.y)
        return sorted({str(pano.id) for pano in panos if getattr(pano, "id", None)})

    async def get_metadata(self, orig_id: str) -> PanoramaMetadata:
        pano = await streetview.find_panorama_by_id_async(orig_id, self._session)
        if pano is None:
            raise ValueError(f"Panorama {orig_id!r} was not found")
        return panorama_metadata_from_object(pano)

    async def download_panorama(self, orig_id: str, output_path: Path) -> DownloadResult:
        pano = await streetview.find_panorama_by_id_async(orig_id, self._session)
        if pano is None:
            raise ValueError(f"Panorama {orig_id!r} was not found")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await streetview.download_panorama_async(pano, str(output_path), self._session)
        return DownloadResult(orig_id=orig_id, image_path=str(output_path))
```

- [ ] **Step 3: Run adapter tests**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_streetview_client.py -q
```

Expected: `2 passed`.

- [ ] **Step 4: Commit**

Run:

```bash
git add services/main/main_service/ingestion/streetview_client.py services/main/tests/ingestion/test_streetview_client.py
git commit -m "feat: add street view ingestion adapter"
```

## Task 5: Add Filesystem Image Store

**Files:**

- Create: `services/main/main_service/ingestion/image_store.py`
- Create: `services/main/tests/ingestion/test_image_store.py`

- [ ] **Step 1: Add image path tests**

Create `services/main/tests/ingestion/test_image_store.py`:

```python
from pathlib import Path

from main_service.ingestion.image_store import ImageStore
from main_service.processing.tiling import TileSpec


def test_panorama_path_is_deterministic() -> None:
    store = ImageStore(Path("data"))

    path = store.panorama_path("Ai1Eh2D14bkcmF0rTDtStA")

    assert path == Path("data/panoramas/Ai1Eh2D14bkcmF0rTDtStA.jpg")


def test_view_path_includes_view_spec() -> None:
    store = ImageStore(Path("data"))
    spec = TileSpec(yaw=90, pitch=10, roll=0, fov=70, width=1000, height=1000)

    path = store.view_path("Ai1Eh2D14bkcmF0rTDtStA", spec)

    assert path == Path(
        "data/views/Ai1Eh2D14bkcmF0rTDtStA/"
        "yaw_90_pitch_10_roll_0_fov_70_w_1000_h_1000.jpg"
    )
```

- [ ] **Step 2: Implement image store**

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

- [ ] **Step 3: Run tests**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_image_store.py -q
```

Expected: `2 passed`.

- [ ] **Step 4: Commit**

Run:

```bash
git add services/main/main_service/ingestion/image_store.py services/main/tests/ingestion/test_image_store.py
git commit -m "feat: add ingestion image store paths"
```

## Task 6: Add Persistence Models and Service Methods

**Files:**

- Create: `services/main/main_service/db/models/map_tile.py`
- Create: `services/main/main_service/db/models/map_tile_panorama.py`
- Modify: `services/main/main_service/db/models/__init__.py`
- Modify: `services/main/main_service/db/models/panorama.py`
- Modify: `services/main/main_service/db/models/tile.py`
- Modify: `services/main/main_service/db/initialize_engine.py`
- Modify: `services/main/main_service/db/services/panorama_service.py`
- Create: `services/main/tests/db/test_panorama_service.py`

- [ ] **Step 1: Add persistence tests with SQLite**

Create `services/main/tests/db/test_panorama_service.py` with tests that create an in-memory SQLite engine, call `Base.metadata.create_all(engine)`, then verify:

```python
from sqlalchemy import create_engine

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
        orig_id="Ai1Eh2D14bkcmF0rTDtStA",
        latitude=37.78,
        longitude=-122.41,
        capture_year=2024,
        capture_month=5,
    )

    first = service.upsert_panorama_metadata(metadata)
    second = service.upsert_panorama_metadata(metadata)

    assert first.id == second.id
    assert first.orig_id == "Ai1Eh2D14bkcmF0rTDtStA"


def test_link_map_tile_to_panorama_is_idempotent() -> None:
    service = make_service()
    map_tile = service.upsert_map_tile(MapTileKey(x=1, y=2, z=17))
    pano = service.upsert_panorama_metadata(
        PanoramaMetadata(
            orig_id="Ai1Eh2D14bkcmF0rTDtStA",
            latitude=37.78,
            longitude=-122.41,
            capture_year=None,
            capture_month=None,
        )
    )

    service.link_map_tile_to_panorama(map_tile.id, pano.id)
    service.link_map_tile_to_panorama(map_tile.id, pano.id)

    assert service.count_map_tile_panorama_links() == 1
```

- [ ] **Step 2: Add models and service methods**

Implement enough model fields and service methods to satisfy the tests. Use unique constraints for `(x, y, z)`, panorama `orig_id`, and `(map_tile_id, panorama_id)`.

- [ ] **Step 3: Run persistence tests**

Run:

```bash
cd services/main
uv run pytest tests/db/test_panorama_service.py -q
```

Expected: `3 passed`.

- [ ] **Step 4: Commit**

Run:

```bash
git add services/main/main_service/db services/main/tests/db/test_panorama_service.py
git commit -m "feat: persist ingestion map tiles and panoramas"
```

## Task 7: Add View Generation Worker Function

**Files:**

- Modify: `services/main/main_service/processing/tiling.py`
- Create: `services/main/main_service/ingestion/view_generator.py`
- Create: `services/main/tests/ingestion/test_view_generator.py`

- [ ] **Step 1: Add tests with a synthetic panorama**

Create `services/main/tests/ingestion/test_view_generator.py` to create a small RGB image, run view generation with two `TileSpec` values, and assert two output image files plus metadata records are produced.

- [ ] **Step 2: Implement view generator**

Create `view_generator.py` with a function that:

- Loads a panorama from disk.
- Converts it to an array.
- Calls `create_tiles_for_pano`.
- Saves each generated view to `ImageStore.view_path`.
- Returns metadata with `orig_id`, `TileSpec`, and saved path.

- [ ] **Step 3: Run view-generator tests**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_view_generator.py tests/test_tiling.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

Run:

```bash
git add services/main/main_service/processing/tiling.py services/main/main_service/ingestion/view_generator.py services/main/tests/ingestion/test_view_generator.py
git commit -m "feat: generate panorama view image files"
```

## Task 8: Add Single-Process Orchestrator

**Files:**

- Create: `services/main/main_service/ingestion/orchestrator.py`
- Modify: `services/main/main_service/__main__.py`
- Create: `services/main/tests/ingestion/test_orchestrator.py`

- [ ] **Step 1: Add orchestrator tests using fake clients**

Create tests with an in-memory SQLite database, a fake `StreetViewClient`, and a temporary image directory. Verify the orchestrator:

- Upserts map tiles.
- Discovers panorama IDs.
- Deduplicates duplicate IDs across tiles.
- Downloads one image per unique panorama.
- Generates views only for downloaded panoramas.

- [ ] **Step 2: Implement orchestrator**

Implement a class that accepts:

- `PanoramaService`
- `StreetViewClient`
- `ImageStore`
- `tile_specs`
- concurrency/retry settings

Keep the first version straightforward and sequential by stage. Add bounded concurrency after the sequential tests pass.

- [ ] **Step 3: Wire CLI entry point without auto-running destructive work**

Update `__main__.py` so a normal import stays safe. Require an explicit command such as:

```bash
python -m main_service ingest
```

to run ingestion.

- [ ] **Step 4: Run orchestrator tests**

Run:

```bash
cd services/main
uv run pytest tests/ingestion/test_orchestrator.py -q
```

Expected: orchestrator tests pass without live network calls.

- [ ] **Step 5: Commit**

Run:

```bash
git add services/main/main_service/ingestion/orchestrator.py services/main/main_service/__main__.py services/main/tests/ingestion/test_orchestrator.py
git commit -m "feat: orchestrate ingestion and panorama processing"
```

## Task 9: Verification

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
.venv/bin/python -B -c "import main_service.ingestion.orchestrator; import main_service.ingestion.streetview_client; import main_service.processing.tiling; print('ingestion imports ok')"
```

Expected output:

```text
ingestion imports ok
```

- [ ] **Step 3: Check git diff**

Run from repo root:

```bash
git status --short
git diff --check
```

Expected: no unstaged changes except intentional docs if the plan checklist was updated during execution, and no whitespace errors.

## Deferred Work

- CLIP/OpenCLIP embedding worker.
- Vector persistence and HNSW index building.
- Text query embedding and search APIs.
- Website result UI.
- Multi-server queue infrastructure.

## Self-Review

- Spec coverage: this plan covers boundary loading, coverage discovery, pano dedupe, download, view generation, persistence, and resumability scaffolding.
- Placeholder scan: no step asks for website/querying work; no step relies on a live Google call in tests.
- Type consistency: `MapTileKey`, `PanoramaMetadata`, `DownloadResult`, `ImageStore`, `StreetViewClient`, and `ProcessingStatus` are introduced before later tasks use them.
- Scope check: embeddings, HNSW, querying, and website work are intentionally deferred.
