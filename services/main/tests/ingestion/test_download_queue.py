import json
import asyncio

from main_service.ingestion.download_queue import (
    BackgroundAsyncRunner,
    InMemoryPanoDownloadQueue,
    InMemoryPanoEmbeddingQueue,
    InMemoryPanoProcessingQueue,
    NatsJetStreamPanoDownloadQueue,
    NatsJetStreamPanoEmbeddingQueue,
    NatsJetStreamPanoProcessingQueue,
    PanoDownloadMessage,
    PanoEmbeddingMessage,
    PanoProcessingMessage,
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


def test_processing_message_serializes_to_public_safe_payload() -> None:
    message = PanoProcessingMessage(
        pano_id=PanoramaId("pano-a"),
        image_path=".local/panoramas/pano-a.jpg",
    )

    assert message.to_dict() == {
        "pano_id": "pano-a",
        "image_path": ".local/panoramas/pano-a.jpg",
        "source": "pano_downloader",
    }


def test_embedding_message_serializes_to_public_safe_payload() -> None:
    message = PanoEmbeddingMessage(
        pano_id=PanoramaId("pano-a"),
        view_id=123,
        image_path=".local/panorama-views/pano-a/candidate/center.jpg",
    )

    assert message.to_dict() == {
        "pano_id": "pano-a",
        "view_id": 123,
        "image_path": ".local/panorama-views/pano-a/candidate/center.jpg",
        "source": "pano_processor",
    }


def test_in_memory_processing_queue_tracks_pending_count() -> None:
    queue = InMemoryPanoProcessingQueue()
    message = PanoProcessingMessage(
        pano_id=PanoramaId("pano-a"),
        image_path=".local/panoramas/pano-a.jpg",
    )

    queue.enqueue(message)

    assert queue.pending_count() == 1
    assert queue.messages == [message]


def test_in_memory_embedding_queue_tracks_pending_count() -> None:
    queue = InMemoryPanoEmbeddingQueue()
    message = PanoEmbeddingMessage(
        pano_id=PanoramaId("pano-a"),
        view_id=123,
        image_path=".local/panorama-views/pano-a/candidate/center.jpg",
    )

    queue.enqueue(message)

    assert queue.pending_count() == 1
    assert queue.messages == [message]


class FakeStreamState:
    messages = 7


class FakeStreamInfo:
    state = FakeStreamState()


class FakeConsumerInfo:
    num_pending = 4
    num_ack_pending = 3


class FakeJetStream:
    def __init__(self) -> None:
        self.stream_names: list[str] = []
        self.consumer_names: list[tuple[str, str]] = []
        self.published: list[dict[str, object]] = []
        self.loop_ids: list[int] = []

    async def stream_info(self, stream_name: str) -> FakeStreamInfo:
        self.loop_ids.append(id(asyncio.get_running_loop()))
        self.stream_names.append(stream_name)
        return FakeStreamInfo()

    async def consumer_info(self, stream_name: str, consumer_name: str) -> FakeConsumerInfo:
        self.loop_ids.append(id(asyncio.get_running_loop()))
        self.consumer_names.append((stream_name, consumer_name))
        return FakeConsumerInfo()

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


def test_nats_queue_reads_pending_count_from_durable_consumer_when_configured() -> None:
    jetstream = FakeJetStream()
    queue = NatsJetStreamPanoDownloadQueue(
        jetstream=jetstream,
        stream_name="PANO_DOWNLOADS",
        subject="pano.download.requested",
        consumer_name="pano-downloader",
    )

    assert queue.pending_count() == 7
    assert jetstream.consumer_names == [("PANO_DOWNLOADS", "pano-downloader")]
    assert jetstream.stream_names == []


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


def test_nats_queue_publishes_json_processing_message() -> None:
    jetstream = FakeJetStream()
    queue = NatsJetStreamPanoProcessingQueue(
        jetstream=jetstream,
        stream_name="PANO_PROCESSING",
        subject="pano.processing.requested",
    )
    message = PanoProcessingMessage(
        pano_id=PanoramaId("pano-a"),
        image_path=".local/panoramas/pano-a.jpg",
    )

    queue.enqueue(message)

    published = jetstream.published[0]
    assert published["subject"] == "pano.processing.requested"
    assert json.loads(published["payload"]) == message.to_dict()


def test_nats_queue_publishes_json_embedding_message() -> None:
    jetstream = FakeJetStream()
    queue = NatsJetStreamPanoEmbeddingQueue(
        jetstream=jetstream,
        stream_name="PANO_EMBEDDING",
        subject="pano.embedding.requested",
    )
    message = PanoEmbeddingMessage(
        pano_id=PanoramaId("pano-a"),
        view_id=123,
        image_path=".local/panorama-views/pano-a/candidate/center.jpg",
    )

    queue.enqueue(message)

    published = jetstream.published[0]
    assert published["subject"] == "pano.embedding.requested"
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
