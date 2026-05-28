# Repo Cleanup Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current repo importable, documented, and testable before adding crawler, downloader, tiling, embedding, or indexing features.

**Architecture:** Keep the current service layout and make the smallest cleanup pass that removes broken scaffolding and duplicate code. Add tests around existing pure functions first, then make runtime entry points safe and side-effect-light. Defer all production pipeline behavior to later specs/plans.

**Tech Stack:** Python 3.14, uv, pytest, SQLAlchemy, Pydantic Settings, mercantile, shapely, py360convert, shared local package.

---

## Git Rules

- Commit coherent local changes as work progresses.
- Use Conventional Commits format for every commit, such as `docs: add planning context`, `test: add cleanup baseline`, or `fix: make pipeline importable`.
- Never run remote/origin operations: no `git push`, `git pull`, `git fetch`, commands targeting `origin`, or remote configuration changes.

## Scope

This plan covers repository cleanup only.

In scope:

- Keep `.codex/` tracked and use it for future plans/specs.
- Keep `AGENTS.md` as the main context handoff for future agents.
- Make `services/main/main_service/pipeline_manager.py` importable.
- Remove duplicated GeoJSON tile-generation code from `pano_retrieval.py`.
- Stop `__main__.py` from inserting empty database rows during a basic run.
- Add a focused pytest baseline for geospatial tile generation, panorama-view tiling, and imports.
- Fix obvious model/config paper cuts discovered during analysis.

Out of scope:

- No Google Street View crawling.
- No panorama downloading.
- No CLIP/OpenCLIP model integration.
- No HNSW index building.
- No web UI.
- No production queue architecture.

## File Map

- `.codex/plans/README.md`: explains where implementation plans live.
- `.codex/specs/README.md`: explains where design specs live.
- `AGENTS.md`: project context, workflow rules, and product direction for future agents.
- `services/main/pyproject.toml`: add pytest as a development dependency if it is not already declared.
- `services/main/main_service/__main__.py`: keep logging setup, remove accidental DB writes, and expose a safe baseline entry point.
- `services/main/main_service/config.py`: align area GeoJSON env naming with the existing `.env` convention.
- `services/main/main_service/geo.py`: keep as the only module for GeoJSON-to-mercantile tile generation.
- `services/main/main_service/pano_retrieval.py`: reduce to retrieval-focused scaffolding; do not duplicate `geo.py`.
- `services/main/main_service/pipeline_manager.py`: replace incomplete syntax with an importable explicit scaffold.
- `services/main/main_service/processing/tiling.py`: keep view-generation helper and tighten types where needed.
- `services/main/main_service/db/models/embedding.py`: fix `__repr__` and relationship shape.
- `services/main/main_service/db/models/panorama.py`: widen pano ID columns and use float coordinates.
- `services/main/main_service/db/models/tile.py`: use float view angles and clarify one-to-one embedding relationship.
- `services/main/tests/`: new focused tests.

## Task 1: Commit Planning Context

**Files:**

- Add: `.codex/plans/README.md`
- Add: `.codex/specs/README.md`
- Add: `.codex/plans/2026-05-27-repo-cleanup-baseline.md`
- Add: `AGENTS.md`

- [ ] **Step 1: Verify docs are not ignored**

Run:

```bash
git check-ignore -v .codex .codex/plans/README.md .codex/specs/README.md AGENTS.md
```

Expected: exit code `1` with no output, meaning these paths are not ignored.

- [ ] **Step 2: Check markdown whitespace**

Run:

```bash
git diff --check
```

Expected: exit code `0` with no output.

- [ ] **Step 3: Commit documentation scaffolding**

Run:

```bash
git add .codex AGENTS.md
git commit -m "docs: add agent planning context"
```

Expected: a commit that includes `.codex/`, `AGENTS.md`, and this plan.

## Task 2: Add Test Baseline

**Files:**

- Modify: `services/main/pyproject.toml`
- Create: `services/main/tests/test_geo.py`
- Create: `services/main/tests/test_tiling.py`
- Create: `services/main/tests/test_imports.py`

- [ ] **Step 1: Declare pytest as a development dependency**

In `services/main/pyproject.toml`, add:

```toml
[dependency-groups]
dev = [
    "pytest>=9.0.0",
]
```

Then run:

```bash
cd services/main
uv sync --group dev
```

Expected: `uv.lock` updates if pytest was not already locked for this service.

- [ ] **Step 2: Add GeoJSON tile-generation tests**

Create `services/main/tests/test_geo.py`:

```python
import json
from pathlib import Path

import pytest

from main_service.geo import generate_tiles_given_geojson


def test_generate_tiles_for_target_geojson_returns_expected_count() -> None:
    data = json.loads(Path("target.geojson").read_text())

    tiles = generate_tiles_given_geojson(data, 17)

    assert len(tiles) == 598
    assert all(tile.z == 17 for tile in tiles)


def test_generate_tiles_rejects_empty_feature_collection() -> None:
    with pytest.raises(ValueError, match="features"):
        generate_tiles_given_geojson({"type": "FeatureCollection", "features": []}, 17)


def test_generate_tiles_rejects_multiple_features() -> None:
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-122.0, 37.0],
                    [-122.0, 37.1],
                    [-121.9, 37.1],
                    [-121.9, 37.0],
                    [-122.0, 37.0],
                ]
            ],
        },
    }
    data = {"type": "FeatureCollection", "features": [feature, feature]}

    with pytest.raises(ValueError, match="must be 1"):
        generate_tiles_given_geojson(data, 17)
```

- [ ] **Step 3: Add panorama-view tiling tests**

Create `services/main/tests/test_tiling.py`:

```python
import numpy as np

from main_service.processing.tiling import TileSpec, create_tiles_for_pano


def test_create_tiles_for_pano_returns_one_tile_per_spec() -> None:
    panorama = np.zeros((128, 256, 3), dtype=np.uint8)
    specs = [
        TileSpec(yaw=0, pitch=0, roll=0, fov=70, width=64, height=32),
        TileSpec(yaw=90, pitch=10, roll=0, fov=60, width=48, height=48),
    ]

    tiles = create_tiles_for_pano(panorama, specs)

    assert [spec for spec, _ in tiles] == specs
    assert tiles[0][1].shape == (32, 64, 3)
    assert tiles[1][1].shape == (48, 48, 3)
```

- [ ] **Step 4: Add import tests for currently broken modules**

Create `services/main/tests/test_imports.py`:

```python
def test_pipeline_manager_imports() -> None:
    import main_service.pipeline_manager  # noqa: F401


def test_pano_retrieval_imports() -> None:
    import main_service.pano_retrieval  # noqa: F401
```

- [ ] **Step 5: Run tests and confirm the expected failure**

Run:

```bash
cd services/main
uv run pytest -q
```

Expected before implementation: at least `test_pipeline_manager_imports` fails with a `SyntaxError` from `pipeline_manager.py`.

- [ ] **Step 6: Commit failing tests**

Run:

```bash
git add services/main/pyproject.toml services/main/uv.lock services/main/tests
git commit -m "test: add main service cleanup baseline"
```

Expected: a commit containing test files and any dependency metadata changes.

## Task 3: Make Pipeline Scaffolding Importable

**Files:**

- Modify: `services/main/main_service/pipeline_manager.py`

- [ ] **Step 1: Replace incomplete syntax with explicit scaffold**

Replace `services/main/main_service/pipeline_manager.py` with:

```python
from collections import deque
from typing import Deque, Optional, Sequence, Final

from main_service.db.models.panorama import Panorama
from main_service.db.services.panorama_service import PanoramaService
from main_service.processing.tiling import TileSpec


EMBEDDING_QUEUE_THRESHOLD = 100


class PipelineManager:
    def __init__(
        self,
        panos_to_process: Sequence[str],
        tile_specs: Sequence[TileSpec],
        pano_service: PanoramaService,
    ) -> None:
        self._panos_to_process: Deque[str] = deque(panos_to_process)
        self._tile_specs: Sequence[TileSpec] = tile_specs
        self._pano_service = pano_service

    def start(self) -> None:
        raise NotImplementedError(
            "Pipeline orchestration is not implemented yet. "
            "Write a spec and plan before adding crawler/downloader/embedder behavior."
        )

    def _refill_queue_if_needed(self, current_queue_size: int) -> None:
        if current_queue_size >= EMBEDDING_QUEUE_THRESHOLD:
            return
        if not self._panos_to_process:
            return

        pano_id_to_process: Final[str] = self._panos_to_process.pop()
        found_pano: Final[Optional[Panorama]] = (
            self._pano_service.find_panorama_by_orig_id(pano_id_to_process)
        )
        if found_pano is not None:
            return

        self._process_pano(pano_id_to_process)

    def _process_pano(self, pano_id: str) -> None:
        found_pano: Final[Optional[Panorama]] = (
            self._pano_service.find_panorama_by_orig_id(pano_id)
        )
        if found_pano is not None:
            return

        raise NotImplementedError(
            f"Panorama processing for {pano_id!r} is not implemented yet."
        )
```

- [ ] **Step 2: Run targeted import test**

Run:

```bash
cd services/main
uv run pytest tests/test_imports.py::test_pipeline_manager_imports -q
```

Expected: `1 passed`.

- [ ] **Step 3: Commit pipeline scaffold cleanup**

Run:

```bash
git add services/main/main_service/pipeline_manager.py
git commit -m "fix: make pipeline manager scaffold importable"
```

Expected: a commit containing only the pipeline manager cleanup.

## Task 4: Remove Duplicate Geo Logic From Retrieval Module

**Files:**

- Modify: `services/main/main_service/pano_retrieval.py`

- [ ] **Step 1: Replace duplicate implementation with retrieval placeholder**

Replace `services/main/main_service/pano_retrieval.py` with:

```python
"""Panorama retrieval helpers.

This module will own Google Street View coverage lookup and panorama download
logic. GeoJSON-to-map-tile conversion belongs in main_service.geo.
"""
```

- [ ] **Step 2: Run retrieval import test**

Run:

```bash
cd services/main
uv run pytest tests/test_imports.py::test_pano_retrieval_imports -q
```

Expected: `1 passed`.

- [ ] **Step 3: Run GeoJSON tests to confirm behavior stayed in `geo.py`**

Run:

```bash
cd services/main
uv run pytest tests/test_geo.py -q
```

Expected: `3 passed`.

- [ ] **Step 4: Commit duplicate-code cleanup**

Run:

```bash
git add services/main/main_service/pano_retrieval.py
git commit -m "refactor: reserve pano retrieval module for retrieval logic"
```

Expected: a commit containing only `pano_retrieval.py`.

## Task 5: Make the Main Entry Point Safe

**Files:**

- Modify: `services/main/main_service/__main__.py`

- [ ] **Step 1: Remove accidental database writes from `main()`**

Replace `services/main/main_service/__main__.py` with:

```python
import logging
import sys
from types import TracebackType

import shared.logger


def configure_exception_logging(logger: logging.Logger) -> None:
    def exception_hook(
        exception_type: type[BaseException],
        exception: BaseException,
        exception_traceback: TracebackType,
    ) -> None:
        logger.critical(
            "Unhandled exception",
            exc_info=(exception_type, exception, exception_traceback),
        )

    sys.excepthook = exception_hook


def main() -> None:
    shared.logger.configure_root_logger()
    logger = logging.getLogger(__package__)
    configure_exception_logging(logger)
    logger.info("main service scaffold started")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add an import/run smoke test**

Append to `services/main/tests/test_imports.py`:

```python


def test_main_module_imports() -> None:
    import main_service.__main__  # noqa: F401
```

- [ ] **Step 3: Run import tests**

Run:

```bash
cd services/main
uv run pytest tests/test_imports.py -q
```

Expected: `3 passed`.

- [ ] **Step 4: Commit safe entry point**

Run:

```bash
git add services/main/main_service/__main__.py services/main/tests/test_imports.py
git commit -m "fix: make main service entry point side-effect safe"
```

Expected: a commit containing the entry point and import smoke test.

## Task 6: Fix Config Naming and Model Paper Cuts

**Files:**

- Modify: `services/main/main_service/config.py`
- Modify: `services/main/main_service/db/models/embedding.py`
- Modify: `services/main/main_service/db/models/panorama.py`
- Modify: `services/main/main_service/db/models/tile.py`
- Create: `services/main/tests/test_models.py`

- [ ] **Step 1: Align GeoJSON config env names**

In `services/main/main_service/config.py`, use:

```python
from typing import Final

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    area_to_process_geojson_filepath: str = Field(
        default="target.geojson",
        validation_alias=AliasChoices(
            "AREA_TO_PROCESS_GEOJSON_FILEPATH",
            "AREA_TO_PROCESS_GEOJSON_FILENAME",
            "area_to_process_geojson_filepath",
        ),
    )
    db_user: str = Field()
    db_password: str = Field()
    db_host: str = Field()
    db_port: int = Field()
    db_name: str = Field()


CONFIG: Final[Settings] = Settings()  # type: ignore
```

- [ ] **Step 2: Fix embedding repr and one-to-one relationship**

In `services/main/main_service/db/models/embedding.py`, use:

```python
from typing import TYPE_CHECKING

from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .tile import Tile


class Embedding(Base):
    __tablename__ = "embedding_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    tile: Mapped["Tile"] = relationship(back_populates="embedding", uselist=False)

    def __repr__(self) -> str:
        return f"Embedding(id={self.id!r})"
```

- [ ] **Step 3: Widen panorama IDs and use float coordinates**

In `services/main/main_service/db/models/panorama.py`, use these field definitions:

```python
    orig_id: Mapped[str] = mapped_column(VARCHAR(64), unique=True)
    image_hash: Mapped[str | None] = mapped_column(VARCHAR(128), unique=True)
    latitude: Mapped[float]
    longitude: Mapped[float]
```

Keep the existing class name, table name, relationship, and `__repr__`.

- [ ] **Step 4: Use float view angles in tiles**

In `services/main/main_service/db/models/tile.py`, use:

```python
    pitch: Mapped[float]
    yaw: Mapped[float]
    roll: Mapped[float]
    fov: Mapped[float]
    google_pitch: Mapped[float]
    google_yaw: Mapped[float]
    google_roll: Mapped[float]
    google_fov: Mapped[float]
```

Keep the existing foreign keys and relationship names.

- [ ] **Step 5: Add model metadata tests**

Create `services/main/tests/test_models.py`:

```python
from main_service.db.models.embedding import Embedding
from main_service.db.models.panorama import Panorama


def test_embedding_repr_is_closed() -> None:
    embedding = Embedding(id=123)

    assert repr(embedding) == "Embedding(id=123)"


def test_panorama_orig_id_column_accepts_real_google_id_length() -> None:
    orig_id_column = Panorama.__table__.c.orig_id

    assert orig_id_column.type.length >= 22
```

- [ ] **Step 6: Run model tests**

Run:

```bash
cd services/main
uv run pytest tests/test_models.py -q
```

Expected: `2 passed`.

- [ ] **Step 7: Commit model/config cleanup**

Run:

```bash
git add services/main/main_service/config.py services/main/main_service/db/models services/main/tests/test_models.py
git commit -m "fix: clean up config and model metadata"
```

Expected: a commit containing config/model cleanup and tests.

## Task 7: Run Full Baseline Verification

**Files:**

- No direct source changes expected.

- [ ] **Step 1: Run all main-service tests**

Run:

```bash
cd services/main
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run direct import smoke check**

Run:

```bash
cd services/main
.venv/bin/python -B -c "import main_service.geo; import main_service.pano_retrieval; import main_service.pipeline_manager; import main_service.processing.tiling; print('imports ok')"
```

Expected output:

```text
imports ok
```

- [ ] **Step 3: Run whitespace check**

Run from repo root:

```bash
git diff --check
```

Expected: exit code `0` with no output.

- [ ] **Step 4: Inspect git status**

Run from repo root:

```bash
git status --short
```

Expected: no uncommitted changes after the previous task commits.

## Risks and Notes

- Existing local MySQL schema may not match the cleaned-up SQLAlchemy model metadata. Do not run destructive migrations in this cleanup pass.
- The current service has no production pipeline yet. `PipelineManager.start()` should intentionally raise `NotImplementedError` until a crawler/downloader/embedder spec exists.
- `py360convert` can be slow on large images; the tiling test uses a small synthetic panorama.
- The exact `598` target tile count is tied to the current `services/main/target.geojson`. If that file changes, update the test and document why in the same plan or a new plan.

## Self-Review

- Spec coverage: this plan covers the requested planning/documentation workflow and the immediate cleanup baseline needed before feature work.
- Placeholder scan: no implementation step uses `TBD`, `TODO`, or an undefined function name.
- Type consistency: `area_to_process_geojson_filepath`, `TileSpec`, `PipelineManager`, and model names are used consistently across tasks.
- Scope check: crawler, downloader, embedding, indexing, and website work are intentionally left for later specs/plans.
