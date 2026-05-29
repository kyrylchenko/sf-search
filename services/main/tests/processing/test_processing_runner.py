import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sqlalchemy import create_engine

from main_service.db.models.base import Base
from main_service.db.services.panorama_service import PanoramaService
from main_service.db.services.panorama_view_service import PanoramaViewService
from main_service.downloader.storage import sha256_file
from main_service.ingestion.types import DownloadStatus, PanoramaId, ProcessingStatus
from main_service.processing.nats_source import (
    PanoProcessingJob,
    ReceivedPanoProcessingJob,
)
from main_service.processing.runner import ProcessingRunResult, run_processing_batch


def make_services() -> tuple[PanoramaService, PanoramaViewService]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return PanoramaService(engine), PanoramaViewService(engine)


def write_test_pano(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.linspace(0, 255, 32, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, 64, dtype=np.uint8)[None, :]
    red = np.broadcast_to(x, (32, 64))
    green = np.broadcast_to(y, (32, 64))
    blue = np.full((32, 64), 120, dtype=np.uint8)
    Image.fromarray(np.dstack([red, green, blue]), mode="RGB").save(path)


def write_viewset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": "candidate",
                "description": "test viewset",
                "views": [
                    {
                        "id": "center",
                        "relative_heading": 0,
                        "pitch": 0,
                        "fov": 60,
                        "view_kind": "center_context",
                        "output_width": 16,
                        "output_height": 12,
                    }
                ],
            }
        )
    )


@dataclass
class FakeReceivedJob:
    job: PanoProcessingJob
    acked: bool = False

    async def ack(self) -> None:
        self.acked = True


class FakeJobSource:
    def __init__(self, jobs: list[FakeReceivedJob]) -> None:
        self.jobs = jobs
        self.fetch_calls: list[int] = []

    async def fetch(self, limit: int) -> list[ReceivedPanoProcessingJob]:
        self.fetch_calls.append(limit)
        return self.jobs[:limit]


def test_runner_generates_view_rows_files_and_acks_job(tmp_path: Path) -> None:
    panorama_service, view_service = make_services()
    pano_path = tmp_path / "panoramas" / "pano-a.jpg"
    write_test_pano(pano_path)
    panorama_service.upsert_discovered_panorama(PanoramaId("pano-a"))
    panorama_service.mark_panorama_downloaded(
        PanoramaId("pano-a"),
        image_path=str(pano_path),
        image_hash=sha256_file(pano_path),
        metadata_json={"pano_id": "pano-a"},
        latitude=37.1,
        longitude=-122.1,
    )
    viewsets_dir = tmp_path / "viewsets"
    write_viewset(viewsets_dir / "candidate.json")
    received = FakeReceivedJob(
        PanoProcessingJob(PanoramaId("pano-a"), image_path=str(pano_path))
    )

    result = asyncio.run(
        run_processing_batch(
            panorama_view_service=view_service,
            job_source=FakeJobSource([received]),
            viewsets_dir=viewsets_dir,
            storage_dir=tmp_path / "views",
            limit=5,
            concurrency=1,
            render_scale=2,
        )
    )

    rows = view_service.list_views_for_panorama(PanoramaId("pano-a"))
    assert result == ProcessingRunResult(
        processed_jobs=1,
        failed_jobs=0,
        generated_views=1,
        skipped_views=0,
        failed_views=0,
    )
    assert received.acked
    assert len(rows) == 1
    assert rows[0].processing_status == ProcessingStatus.COMPLETE.value
    assert rows[0].image_path is not None
    assert rows[0].image_hash is not None
    assert rows[0].rendered_width == 32
    assert rows[0].rendered_height == 24
    with Image.open(rows[0].image_path) as image:
        assert image.size == (32, 24)


def test_runner_skips_duplicate_completed_view(tmp_path: Path) -> None:
    panorama_service, view_service = make_services()
    pano_path = tmp_path / "panoramas" / "pano-a.jpg"
    write_test_pano(pano_path)
    panorama_service.upsert_discovered_panorama(PanoramaId("pano-a"))
    panorama_service.mark_panorama_downloaded(
        PanoramaId("pano-a"),
        image_path=str(pano_path),
        image_hash=sha256_file(pano_path),
        metadata_json={"pano_id": "pano-a"},
        latitude=None,
        longitude=None,
    )
    viewsets_dir = tmp_path / "viewsets"
    write_viewset(viewsets_dir / "candidate.json")

    for _ in range(2):
        received = FakeReceivedJob(
            PanoProcessingJob(PanoramaId("pano-a"), image_path=str(pano_path))
        )
        result = asyncio.run(
            run_processing_batch(
                panorama_view_service=view_service,
                job_source=FakeJobSource([received]),
                viewsets_dir=viewsets_dir,
                storage_dir=tmp_path / "views",
                limit=5,
                concurrency=1,
                render_scale=2,
            )
        )

    rows = view_service.list_views_for_panorama(PanoramaId("pano-a"))
    assert result.generated_views == 0
    assert result.skipped_views == 1
    assert received.acked
    assert len(rows) == 1


def test_runner_acks_missing_source_image_as_failed_job(tmp_path: Path) -> None:
    _, view_service = make_services()
    viewsets_dir = tmp_path / "viewsets"
    write_viewset(viewsets_dir / "candidate.json")
    received = FakeReceivedJob(
        PanoProcessingJob(PanoramaId("pano-a"), image_path=str(tmp_path / "missing.jpg"))
    )

    result = asyncio.run(
        run_processing_batch(
            panorama_view_service=view_service,
            job_source=FakeJobSource([received]),
            viewsets_dir=viewsets_dir,
            storage_dir=tmp_path / "views",
            limit=5,
            concurrency=1,
            render_scale=2,
        )
    )

    assert result.failed_jobs == 1
    assert received.acked


def test_runner_skips_pano_that_is_not_marked_downloaded(tmp_path: Path) -> None:
    panorama_service, view_service = make_services()
    panorama_service.upsert_discovered_panorama(PanoramaId("pano-a"))
    viewsets_dir = tmp_path / "viewsets"
    write_viewset(viewsets_dir / "candidate.json")
    received = FakeReceivedJob(
        PanoProcessingJob(PanoramaId("pano-a"), image_path=str(tmp_path / "pano-a.jpg"))
    )

    result = asyncio.run(
        run_processing_batch(
            panorama_view_service=view_service,
            job_source=FakeJobSource([received]),
            viewsets_dir=viewsets_dir,
            storage_dir=tmp_path / "views",
            limit=5,
            concurrency=1,
            render_scale=2,
        )
    )

    row = panorama_service.find_panorama_by_orig_id("pano-a")
    assert result.failed_jobs == 1
    assert received.acked
    assert row is not None
    assert row.download_status == DownloadStatus.PENDING.value
