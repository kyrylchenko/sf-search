import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine

from main_service.db.models import embedding, map_tile, map_tile_panorama, panorama, tile
from main_service.db.models.base import Base
from main_service.db.services.panorama_service import PanoramaService
from main_service.downloader.runner import (
    DownloadRunResult,
    PanoDownloadJob,
    ReceivedPanoDownloadJob,
    run_downloader_batch,
)
from main_service.downloader.streetview_client import ResolvedPanorama
from main_service.ingestion.download_queue import PanoProcessingMessage
from main_service.ingestion.types import DownloadStatus, PanoramaId


def make_service() -> PanoramaService:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return PanoramaService(engine)


@dataclass
class FakeReceivedJob:
    job: PanoDownloadJob
    acked: bool = False

    async def ack(self) -> None:
        self.acked = True


class FakeJobSource:
    def __init__(self, jobs: list[FakeReceivedJob]) -> None:
        self.jobs = jobs
        self.fetch_calls: list[int] = []

    async def fetch(self, limit: int) -> list[ReceivedPanoDownloadJob]:
        self.fetch_calls.append(limit)
        return self.jobs[:limit]


class FakeProcessingQueue:
    def __init__(self, pending: int = 0) -> None:
        self.pending = pending
        self.messages: list[PanoProcessingMessage] = []

    def pending_count(self) -> int:
        return self.pending

    def enqueue(self, message: PanoProcessingMessage) -> None:
        self.messages.append(message)


class FakeStreetViewClient:
    def __init__(self) -> None:
        self.resolve_calls: list[PanoramaId] = []
        self.download_calls: list[Path] = []

    async def resolve(
        self,
        pano_id: PanoramaId,
        session: object,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> ResolvedPanorama | None:
        self.resolve_calls.append(pano_id)
        return ResolvedPanorama(
            requested_pano_id=pano_id,
            resolved_pano_id=f"{pano_id.value}-resolved",
            panorama=SimpleNamespace(id=pano_id.value),
            latitude=latitude,
            longitude=longitude,
            metadata_json={"pano_id": pano_id.value},
        )

    async def download(
        self,
        panorama: object,
        output_path: Path,
        session: object,
        *,
        zoom: int,
    ) -> None:
        self.download_calls.append(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"image-bytes")


class NullSession:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *args: object) -> None:
        return None


def test_runner_pauses_before_fetching_when_processing_queue_is_backed_up(
    tmp_path: Path,
) -> None:
    source = FakeJobSource([FakeReceivedJob(PanoDownloadJob(PanoramaId("pano-a")))])
    result = asyncio.run(
        run_downloader_batch(
            panorama_service=make_service(),
            job_source=source,
            processing_queue=FakeProcessingQueue(pending=50),
            streetview_client=FakeStreetViewClient(),
            storage_dir=tmp_path,
            limit=5,
            concurrency=5,
            max_processing_queue_depth=50,
            session_factory=NullSession,
        )
    )

    assert result == DownloadRunResult(
        downloaded=0,
        skipped=0,
        failed=0,
        paused=True,
        pause_reason="processing_queue_backlog",
    )
    assert source.fetch_calls == []


def test_runner_downloads_job_updates_db_publishes_processing_and_acks(
    tmp_path: Path,
) -> None:
    service = make_service()
    service.upsert_discovered_panorama(PanoramaId("pano-a"))
    received = FakeReceivedJob(PanoDownloadJob(PanoramaId("pano-a")))
    processing_queue = FakeProcessingQueue()
    client = FakeStreetViewClient()

    result = asyncio.run(
        run_downloader_batch(
            panorama_service=service,
            job_source=FakeJobSource([received]),
            processing_queue=processing_queue,
            streetview_client=client,
            storage_dir=tmp_path,
            limit=5,
            concurrency=5,
            max_processing_queue_depth=50,
            session_factory=NullSession,
        )
    )

    updated = service.find_panorama_by_orig_id("pano-a")
    assert result.downloaded == 1
    assert received.acked
    assert updated is not None
    assert updated.download_status == DownloadStatus.DOWNLOADED.value
    assert updated.image_path == str(tmp_path / "pano-a.jpg")
    assert updated.image_hash == (
        "2c8648d103e3dd7ad87660da0f126a1443b6d21ac1bd3ec000c5e24e2373a90c"
    )
    assert processing_queue.messages == [
        PanoProcessingMessage(
            pano_id=PanoramaId("pano-a"),
            image_path=str(tmp_path / "pano-a.jpg"),
        )
    ]


def test_runner_reports_progress_events(tmp_path: Path) -> None:
    service = make_service()
    service.upsert_discovered_panorama(PanoramaId("pano-a"))
    events: list[tuple[str, dict[str, object]]] = []

    asyncio.run(
        run_downloader_batch(
            panorama_service=service,
            job_source=FakeJobSource(
                [FakeReceivedJob(PanoDownloadJob(PanoramaId("pano-a")))]
            ),
            processing_queue=FakeProcessingQueue(),
            streetview_client=FakeStreetViewClient(),
            storage_dir=tmp_path,
            limit=5,
            concurrency=5,
            max_processing_queue_depth=50,
            session_factory=NullSession,
            progress=lambda event, payload: events.append((event, payload)),
        )
    )

    event_names = [name for name, _ in events]
    assert event_names[0] == "downloader_batch_start"
    assert "downloader_fetch_start" in event_names
    assert "downloader_fetch_complete" in event_names
    assert "downloader_job_start" in event_names
    assert "downloader_job_complete" in event_names
    assert "downloader_batch_complete" in event_names


def test_runner_skips_duplicate_completed_pano_without_downloading(
    tmp_path: Path,
) -> None:
    service = make_service()
    service.upsert_discovered_panorama(PanoramaId("pano-a"))
    service.mark_panorama_downloaded(
        PanoramaId("pano-a"),
        image_path=str(tmp_path / "pano-a.jpg"),
        image_hash="existing-hash",
        metadata_json={"pano_id": "pano-a"},
        latitude=None,
        longitude=None,
    )
    received = FakeReceivedJob(PanoDownloadJob(PanoramaId("pano-a")))
    processing_queue = FakeProcessingQueue()
    client = FakeStreetViewClient()

    result = asyncio.run(
        run_downloader_batch(
            panorama_service=service,
            job_source=FakeJobSource([received]),
            processing_queue=processing_queue,
            streetview_client=client,
            storage_dir=tmp_path,
            limit=5,
            concurrency=5,
            max_processing_queue_depth=50,
            session_factory=NullSession,
        )
    )

    assert result.skipped == 1
    assert received.acked
    assert client.resolve_calls == []
    assert processing_queue.messages == []


def test_runner_records_failure_and_acks_input(tmp_path: Path) -> None:
    class FailingStreetViewClient(FakeStreetViewClient):
        async def download(
            self,
            panorama: object,
            output_path: Path,
            session: object,
            *,
            zoom: int,
        ) -> None:
            raise RuntimeError("download failed")

    service = make_service()
    service.upsert_discovered_panorama(PanoramaId("pano-a"))
    received = FakeReceivedJob(PanoDownloadJob(PanoramaId("pano-a")))

    result = asyncio.run(
        run_downloader_batch(
            panorama_service=service,
            job_source=FakeJobSource([received]),
            processing_queue=FakeProcessingQueue(),
            streetview_client=FailingStreetViewClient(),
            storage_dir=tmp_path,
            limit=5,
            concurrency=5,
            max_processing_queue_depth=50,
            max_attempts=3,
            session_factory=NullSession,
        )
    )

    updated = service.find_panorama_by_orig_id("pano-a")
    assert result.failed == 1
    assert received.acked
    assert updated is not None
    assert updated.download_status == DownloadStatus.FAILED.value
    assert updated.attempt_count == 1
    assert updated.last_error == "download failed"


def test_runner_honors_concurrency_limit(tmp_path: Path) -> None:
    class ConcurrentStreetViewClient(FakeStreetViewClient):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.max_active = 0

        async def download(
            self,
            panorama: object,
            output_path: Path,
            session: object,
            *,
            zoom: int,
            ) -> None:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                await asyncio.sleep(0.01)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(output_path.name.encode("utf-8"))
                self.active -= 1

    service = make_service()
    jobs = []
    for index in range(4):
        pano_id = PanoramaId(f"pano-{index}")
        service.upsert_discovered_panorama(pano_id)
        jobs.append(FakeReceivedJob(PanoDownloadJob(pano_id)))
    client = ConcurrentStreetViewClient()

    result = asyncio.run(
        run_downloader_batch(
            panorama_service=service,
            job_source=FakeJobSource(jobs),
            processing_queue=FakeProcessingQueue(),
            streetview_client=client,
            storage_dir=tmp_path,
            limit=4,
            concurrency=2,
            max_processing_queue_depth=50,
            session_factory=NullSession,
        )
    )

    assert result.downloaded == 4
    assert client.max_active == 2
