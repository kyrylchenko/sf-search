from sqlalchemy import create_engine

from main_service.db.models.base import Base
from main_service.db.services.panorama_service import PanoramaService
from main_service.downloader.requeue import requeue_download_jobs_from_db
from main_service.ingestion.download_queue import InMemoryPanoDownloadQueue
from main_service.ingestion.types import DownloadStatus, MapTileKey, PanoramaId


def test_requeue_download_jobs_from_db_publishes_retryable_panos() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    service = PanoramaService(engine)
    tile = service.upsert_map_tile(MapTileKey(x=1, y=2, z=17))
    pano = service.upsert_discovered_panorama(PanoramaId("pano-a"))
    service.link_map_tile_to_panorama(tile.id, pano.id)
    service.mark_panorama_download_status(pano.id, DownloadStatus.FAILED)
    queue = InMemoryPanoDownloadQueue()

    count = requeue_download_jobs_from_db(
        panorama_service=service,
        download_queue=queue,
        limit=5,
    )

    assert count == 1
    assert [message.pano_id.value for message in queue.messages] == ["pano-a"]
    assert queue.messages[0].source_tile == MapTileKey(x=1, y=2, z=17)
    row = service.find_panorama_by_orig_id("pano-a")
    assert row is not None
    assert row.download_status == DownloadStatus.QUEUED.value
