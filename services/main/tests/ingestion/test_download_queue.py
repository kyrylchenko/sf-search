import json
import asyncio

from main_service.ingestion.download_queue import (
    BackgroundAsyncRunner,
    InMemoryPanoDownloadQueue,
    NatsJetStreamPanoDownloadQueue,
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


class FakeStreamState:
    messages = 7


class FakeStreamInfo:
    state = FakeStreamState()


class FakeJetStream:
    def __init__(self) -> None:
        self.stream_names: list[str] = []
        self.published: list[dict[str, object]] = []
        self.loop_ids: list[int] = []

    async def stream_info(self, stream_name: str) -> FakeStreamInfo:
        self.loop_ids.append(id(asyncio.get_running_loop()))
        self.stream_names.append(stream_name)
        return FakeStreamInfo()

    async def publish(self, subject: str, payload: bytes) -> None:
        self.loop_ids.append(id(asyncio.get_running_loop()))
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


def test_nats_queue_can_run_operations_on_one_background_event_loop() -> None:
    jetstream = FakeJetStream()
    runner = BackgroundAsyncRunner()
    queue = NatsJetStreamPanoDownloadQueue(
        jetstream=jetstream,
        stream_name="PANO_DOWNLOADS",
        subject="pano.download.requested",
        async_runner=runner,
    )

    try:
        queue.pending_count()
        queue.enqueue(
            PanoDownloadMessage(
                pano_id=PanoramaId("pano-a"),
                source_tile=MapTileKey(x=1, y=2, z=17),
            )
        )
    finally:
        runner.close()

    assert len(set(jetstream.loop_ids)) == 1
