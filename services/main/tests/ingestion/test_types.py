from main_service.ingestion.types import (
    DownloadStatus,
    MapTileKey,
    PanoramaId,
    ProcessingStatus,
)


def test_map_tile_key_tuple_round_trip() -> None:
    key = MapTileKey(x=1, y=2, z=17)

    assert key.as_tuple() == (1, 2, 17)


def test_processing_status_values_are_stable() -> None:
    assert ProcessingStatus.PENDING.value == "pending"
    assert ProcessingStatus.PROCESSING.value == "processing"
    assert ProcessingStatus.COMPLETE.value == "complete"
    assert ProcessingStatus.FAILED.value == "failed"
    assert ProcessingStatus.SKIPPED.value == "skipped"


def test_download_status_values_are_stable() -> None:
    assert DownloadStatus.PENDING.value == "pending"
    assert DownloadStatus.QUEUED.value == "queued"
    assert DownloadStatus.DOWNLOADING.value == "downloading"
    assert DownloadStatus.DOWNLOADED.value == "downloaded"
    assert DownloadStatus.FAILED.value == "failed"
    assert DownloadStatus.SKIPPED.value == "skipped"


def test_panorama_id_wraps_public_safe_example_value() -> None:
    pano_id = PanoramaId(value="example-pano-id")

    assert pano_id.value == "example-pano-id"
    assert pano_id.latitude is None
    assert pano_id.longitude is None


def test_panorama_id_can_carry_coverage_coordinates() -> None:
    pano_id = PanoramaId(value="example-pano-id", latitude=37.1, longitude=-122.1)

    assert pano_id.value == "example-pano-id"
    assert pano_id.latitude == 37.1
    assert pano_id.longitude == -122.1
