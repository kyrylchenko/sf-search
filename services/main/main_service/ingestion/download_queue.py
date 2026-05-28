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
