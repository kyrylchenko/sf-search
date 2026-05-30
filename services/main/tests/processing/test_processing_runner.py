import asyncio
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sqlalchemy import create_engine

from main_service.db.models.base import Base
from main_service.db.services.panorama_service import PanoramaService
from main_service.db.services.panorama_view_service import PanoramaViewService
from main_service.downloader.storage import sha256_file
from main_service.ingestion.download_queue import PanoEmbeddingMessage
from main_service.ingestion.types import DownloadStatus, PanoramaId, ProcessingStatus
from main_service.processing.nats_source import (
    PanoProcessingJob,
    ReceivedPanoProcessingJob,
)
from main_service.processing.runner import ProcessingRunResult, run_processing_batch
from main_service.processing.runner import _bounded_view_concurrency


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


def write_multi_viewset(path: Path, view_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": "candidate",
                "description": "test viewset",
                "views": [
                    {
                        "id": f"view-{index:03d}",
                        "relative_heading": index * 10,
                        "pitch": 0,
                        "fov": 60,
                        "view_kind": "small_object",
                        "output_width": 16,
                        "output_height": 12,
                    }
                    for index in range(view_count)
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


class FakeEmbeddingQueue:
    def __init__(self, pending: int = 0) -> None:
        self.pending = pending
        self.messages: list[PanoEmbeddingMessage] = []

    def pending_count(self) -> int:
        return self.pending

    def enqueue(self, message: PanoEmbeddingMessage) -> None:
        self.messages.append(message)
        self.pending += 1


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


def test_runner_enqueues_embedding_job_for_completed_view(tmp_path: Path) -> None:
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
    embedding_queue = FakeEmbeddingQueue()

    result = asyncio.run(
        run_processing_batch(
            panorama_view_service=view_service,
            job_source=FakeJobSource([received]),
            viewsets_dir=viewsets_dir,
            storage_dir=tmp_path / "views",
            limit=5,
            concurrency=1,
            render_scale=2,
            embedding_queue=embedding_queue,
            max_embedding_queue_depth=100,
        )
    )

    rows = view_service.list_views_for_panorama(PanoramaId("pano-a"))
    assert result.queued_embeddings == 1
    assert embedding_queue.messages == [
        PanoEmbeddingMessage(
            pano_id=PanoramaId("pano-a"),
            view_id=rows[0].id,
            image_path=rows[0].image_path or "",
        )
    ]


def test_runner_renders_views_in_parallel_within_one_panorama(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from main_service.processing import runner

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
    write_multi_viewset(viewsets_dir / "candidate.json", view_count=4)
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_render_and_store_view(**kwargs: object) -> None:
        nonlocal active, max_active
        output_path = kwargs["output_path"]
        assert isinstance(output_path, Path)
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.05)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color=(20, 40, 80)).save(output_path)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(runner, "_render_and_store_view", fake_render_and_store_view)

    result = asyncio.run(
        run_processing_batch(
            panorama_view_service=view_service,
            job_source=FakeJobSource(
                [
                    FakeReceivedJob(
                        PanoProcessingJob(PanoramaId("pano-a"), image_path=str(pano_path))
                    )
                ]
            ),
            viewsets_dir=viewsets_dir,
            storage_dir=tmp_path / "views",
            limit=5,
            concurrency=4,
            render_scale=2,
        )
    )

    assert result.generated_views == 4
    assert max_active > 1


def test_bounded_view_concurrency_caps_unsafe_user_values() -> None:
    assert _bounded_view_concurrency(requested=100, maximum=4) == 4
    assert _bounded_view_concurrency(requested=0, maximum=4) == 1
    assert _bounded_view_concurrency(requested=4, maximum=0) == 1


def test_runner_reports_capped_view_concurrency(tmp_path: Path) -> None:
    _, view_service = make_services()
    viewsets_dir = tmp_path / "viewsets"
    write_viewset(viewsets_dir / "candidate.json")
    events: list[tuple[str, dict[str, object]]] = []

    asyncio.run(
        run_processing_batch(
            panorama_view_service=view_service,
            job_source=FakeJobSource([]),
            viewsets_dir=viewsets_dir,
            storage_dir=tmp_path / "views",
            limit=5,
            concurrency=100,
            max_view_concurrency=4,
            render_scale=2,
            progress=lambda event, payload: events.append((event, payload)),
        )
    )

    assert (
        "processing_view_concurrency_capped",
        {"requested": 100, "maximum": 4, "effective": 4},
    ) in events


def test_runner_pauses_before_fetching_when_embedding_queue_is_backed_up(
    tmp_path: Path,
) -> None:
    _, view_service = make_services()
    viewsets_dir = tmp_path / "viewsets"
    write_viewset(viewsets_dir / "candidate.json")
    source = FakeJobSource(
        [
            FakeReceivedJob(
                PanoProcessingJob(
                    PanoramaId("pano-a"),
                    image_path=str(tmp_path / "panoramas" / "pano-a.jpg"),
                )
            )
        ]
    )

    result = asyncio.run(
        run_processing_batch(
            panorama_view_service=view_service,
            job_source=source,
            viewsets_dir=viewsets_dir,
            storage_dir=tmp_path / "views",
            limit=5,
            concurrency=1,
            render_scale=2,
            embedding_queue=FakeEmbeddingQueue(pending=100),
            max_embedding_queue_depth=100,
        )
    )

    assert result == ProcessingRunResult(
        processed_jobs=0,
        failed_jobs=0,
        generated_views=0,
        skipped_views=0,
        failed_views=0,
        queued_embeddings=0,
        paused=True,
        pause_reason="embedding_queue_backlog",
    )
    assert source.fetch_calls == []


def test_runner_reports_progress_events(tmp_path: Path) -> None:
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
    events: list[tuple[str, dict[str, object]]] = []

    asyncio.run(
        run_processing_batch(
            panorama_view_service=view_service,
            job_source=FakeJobSource(
                [FakeReceivedJob(PanoProcessingJob(PanoramaId("pano-a"), str(pano_path)))]
            ),
            viewsets_dir=viewsets_dir,
            storage_dir=tmp_path / "views",
            limit=5,
            concurrency=1,
            render_scale=1,
            progress=lambda event, payload: events.append((event, payload)),
        )
    )

    event_names = [name for name, _ in events]
    assert "processing_fetch_start" in event_names
    assert "processing_renderer_backend" in event_names
    assert "processing_fetch_complete" in event_names
    assert "processing_job_start" in event_names
    assert "processing_view_complete" in event_names
    assert "processing_job_complete" in event_names


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
