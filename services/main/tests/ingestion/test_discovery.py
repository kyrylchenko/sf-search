from sqlalchemy import create_engine

from main_service.db.models import embedding, map_tile, map_tile_panorama, panorama, tile
from main_service.db.models.base import Base
from main_service.db.services.panorama_service import PanoramaService
from main_service.ingestion.discovery import discover_panos_for_tiles
from main_service.ingestion.download_queue import InMemoryPanoDownloadQueue
from main_service.ingestion.types import MapTileKey, PanoramaId


class FakeCoverageClient:
    def __init__(self) -> None:
        self.calls: list[MapTileKey] = []

    def get_pano_ids_for_tile(self, tile: MapTileKey) -> list[PanoramaId]:
        self.calls.append(tile)
        if tile.x == 1:
            return [PanoramaId("pano-a"), PanoramaId("pano-b")]
        return [PanoramaId("pano-b"), PanoramaId("pano-c")]


def make_service() -> PanoramaService:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return PanoramaService(engine)


def test_discover_panos_for_tiles_persists_tiles_panos_and_links() -> None:
    service = make_service()
    coverage_client = FakeCoverageClient()
    download_queue = InMemoryPanoDownloadQueue()
    tiles = [MapTileKey(x=1, y=10, z=17), MapTileKey(x=2, y=10, z=17)]

    result = discover_panos_for_tiles(
        service,
        coverage_client,
        download_queue,
        tiles,
        max_downloader_queue_depth=1000,
    )

    assert result.tiles_processed == 2
    assert result.unique_panos_discovered == 3
    assert result.tile_pano_links_created == 4
    assert result.enqueued_downloads == 4
    assert result.paused is False
    assert coverage_client.calls == tiles
    assert download_queue.pending_count() == 4


def test_discovery_pauses_before_next_tile_when_downloader_queue_is_full() -> None:
    service = make_service()
    coverage_client = FakeCoverageClient()
    download_queue = InMemoryPanoDownloadQueue()
    tiles = [MapTileKey(x=1, y=10, z=17), MapTileKey(x=2, y=10, z=17)]

    result = discover_panos_for_tiles(
        service,
        coverage_client,
        download_queue,
        tiles,
        max_downloader_queue_depth=1,
    )

    assert result.tiles_processed == 1
    assert result.paused is True
    assert result.pause_reason == "downloader_queue_full"
    assert coverage_client.calls == [tiles[0]]
    assert download_queue.pending_count() == 2
