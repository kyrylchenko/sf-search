from sqlalchemy import create_engine

from main_service.db.models import embedding, map_tile, map_tile_panorama, panorama, tile
from main_service.db.models.base import Base
from main_service.db.services.panorama_service import PanoramaService
from main_service.ingestion.types import (
    DownloadStatus,
    MapTileKey,
    PanoramaId,
    ProcessingStatus,
)


def make_service() -> PanoramaService:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return PanoramaService(engine)


def test_upsert_map_tile_is_idempotent() -> None:
    service = make_service()
    key = MapTileKey(x=1, y=2, z=17)

    first = service.upsert_map_tile(key)
    second = service.upsert_map_tile(key)

    assert first.id == second.id
    assert first.discovery_status == ProcessingStatus.PENDING.value


def test_upsert_discovered_panorama_is_idempotent() -> None:
    service = make_service()
    pano_id = PanoramaId(value="example-pano-id")

    first = service.upsert_discovered_panorama(pano_id)
    second = service.upsert_discovered_panorama(pano_id)

    assert first.id == second.id
    assert first.orig_id == "example-pano-id"
    assert first.download_status == DownloadStatus.PENDING.value


def test_link_map_tile_to_panorama_is_idempotent() -> None:
    service = make_service()
    tile_row = service.upsert_map_tile(MapTileKey(x=1, y=2, z=17))
    pano_row = service.upsert_discovered_panorama(PanoramaId(value="example-pano-id"))

    service.link_map_tile_to_panorama(tile_row.id, pano_row.id)
    service.link_map_tile_to_panorama(tile_row.id, pano_row.id)

    assert service.count_map_tile_panorama_links() == 1


def test_mark_map_tile_discovery_complete_updates_status() -> None:
    service = make_service()
    tile_row = service.upsert_map_tile(MapTileKey(x=1, y=2, z=17))

    updated = service.mark_map_tile_discovery_complete(tile_row.id)

    assert updated.discovery_status == ProcessingStatus.COMPLETE.value


def test_mark_panorama_download_queued_updates_status() -> None:
    service = make_service()
    pano_row = service.upsert_discovered_panorama(PanoramaId(value="example-pano-id"))

    updated = service.mark_panorama_download_queued(pano_row.id)

    assert updated.download_status == DownloadStatus.QUEUED.value


def test_claim_panorama_for_download_marks_retryable_pano_downloading() -> None:
    service = make_service()
    pano_row = service.upsert_discovered_panorama(PanoramaId(value="example-pano-id"))
    service.mark_panorama_download_queued(pano_row.id)

    claimed = service.claim_panorama_for_download(PanoramaId(value="example-pano-id"))

    assert claimed is not None
    assert claimed.orig_id == "example-pano-id"
    assert claimed.download_status == DownloadStatus.DOWNLOADING.value


def test_claim_panorama_for_download_skips_complete_duplicate() -> None:
    service = make_service()
    pano_row = service.upsert_discovered_panorama(PanoramaId(value="example-pano-id"))
    service.mark_panorama_downloaded(
        PanoramaId(value="example-pano-id"),
        image_path=".local/panoramas/example-pano-id.jpg",
        image_hash="abc123",
        metadata_json={"pano_id": "example-pano-id"},
        latitude=37.1,
        longitude=-122.1,
    )

    claimed = service.claim_panorama_for_download(PanoramaId(value="example-pano-id"))

    assert claimed is None
    assert service.is_panorama_download_complete(PanoramaId(value="example-pano-id"))


def test_mark_panorama_downloaded_stores_outputs_and_metadata() -> None:
    service = make_service()
    service.upsert_discovered_panorama(PanoramaId(value="example-pano-id"))

    updated = service.mark_panorama_downloaded(
        PanoramaId(value="example-pano-id"),
        image_path=".local/panoramas/example-pano-id.jpg",
        image_hash="abc123",
        metadata_json={"date": "2024-01", "heading": 12.5},
        latitude=37.1,
        longitude=-122.1,
    )

    assert updated.download_status == DownloadStatus.DOWNLOADED.value
    assert updated.image_path == ".local/panoramas/example-pano-id.jpg"
    assert updated.image_hash == "abc123"
    assert updated.metadata_json == {"date": "2024-01", "heading": 12.5}
    assert updated.latitude == 37.1
    assert updated.longitude == -122.1
    assert updated.downloaded_at is not None
    assert updated.last_error is None


def test_mark_panorama_download_failed_increments_attempts_and_stores_error() -> None:
    service = make_service()
    service.upsert_discovered_panorama(PanoramaId(value="example-pano-id"))

    updated = service.mark_panorama_download_failed(
        PanoramaId(value="example-pano-id"),
        error="tile download failed",
        max_attempts=3,
    )

    assert updated.download_status == DownloadStatus.FAILED.value
    assert updated.attempt_count == 1
    assert updated.last_error == "tile download failed"


def test_mark_panorama_download_failed_sets_skipped_at_max_attempts() -> None:
    service = make_service()
    service.upsert_discovered_panorama(PanoramaId(value="example-pano-id"))

    service.mark_panorama_download_failed(
        PanoramaId(value="example-pano-id"),
        error="first failure",
        max_attempts=2,
    )
    updated = service.mark_panorama_download_failed(
        PanoramaId(value="example-pano-id"),
        error="second failure",
        max_attempts=2,
    )

    assert updated.download_status == DownloadStatus.SKIPPED.value
    assert updated.attempt_count == 2
    assert updated.last_error == "second failure"


def test_list_downloadable_pano_ids_for_map_tile_returns_linked_nonterminal_panos() -> None:
    service = make_service()
    tile_row = service.upsert_map_tile(MapTileKey(x=1, y=2, z=17))
    pending = service.upsert_discovered_panorama(PanoramaId(value="pano-pending"))
    queued = service.upsert_discovered_panorama(PanoramaId(value="pano-queued"))

    service.link_map_tile_to_panorama(tile_row.id, pending.id)
    service.link_map_tile_to_panorama(tile_row.id, queued.id)
    service.mark_panorama_download_queued(queued.id)

    pano_ids = service.list_downloadable_pano_ids_for_map_tile(tile_row.id)

    assert [pano_id.value for pano_id in pano_ids] == ["pano-pending", "pano-queued"]


def test_list_downloadable_pano_ids_for_map_tile_excludes_terminal_download_statuses() -> None:
    service = make_service()
    tile_row = service.upsert_map_tile(MapTileKey(x=1, y=2, z=17))
    downloaded = service.upsert_discovered_panorama(PanoramaId(value="pano-downloaded"))
    skipped = service.upsert_discovered_panorama(PanoramaId(value="pano-skipped"))
    pending = service.upsert_discovered_panorama(PanoramaId(value="pano-pending"))

    service.link_map_tile_to_panorama(tile_row.id, downloaded.id)
    service.link_map_tile_to_panorama(tile_row.id, skipped.id)
    service.link_map_tile_to_panorama(tile_row.id, pending.id)
    service.mark_panorama_download_status(downloaded.id, DownloadStatus.DOWNLOADED)
    service.mark_panorama_download_status(skipped.id, DownloadStatus.SKIPPED)

    pano_ids = service.list_downloadable_pano_ids_for_map_tile(tile_row.id)

    assert [pano_id.value for pano_id in pano_ids] == ["pano-pending"]
