from types import SimpleNamespace

from main_service.ingestion.coverage_client import pano_ids_from_coverage_objects


def test_pano_ids_from_coverage_objects_deduplicates_and_sorts() -> None:
    coverage_objects = [
        SimpleNamespace(id="pano-b", lat=37.2, lon=-122.2),
        SimpleNamespace(id="pano-a", lat=37.1, lon=-122.1),
        SimpleNamespace(id="pano-b", lat=37.2, lon=-122.2),
        SimpleNamespace(id=None),
    ]

    pano_ids = pano_ids_from_coverage_objects(coverage_objects)

    assert [pano_id.value for pano_id in pano_ids] == ["pano-a", "pano-b"]
    assert pano_ids[0].latitude == 37.1
    assert pano_ids[0].longitude == -122.1
