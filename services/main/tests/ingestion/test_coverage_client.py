from types import SimpleNamespace

from main_service.ingestion.coverage_client import pano_ids_from_coverage_objects


def test_pano_ids_from_coverage_objects_deduplicates_and_sorts() -> None:
    coverage_objects = [
        SimpleNamespace(id="pano-b"),
        SimpleNamespace(id="pano-a"),
        SimpleNamespace(id="pano-b"),
        SimpleNamespace(id=None),
    ]

    pano_ids = pano_ids_from_coverage_objects(coverage_objects)

    assert [pano_id.value for pano_id in pano_ids] == ["pano-a", "pano-b"]
