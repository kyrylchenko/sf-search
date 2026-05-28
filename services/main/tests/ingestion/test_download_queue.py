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
