from typing import Protocol

from streetlevel import streetview

from main_service.ingestion.types import MapTileKey, PanoramaId


class CoverageClient(Protocol):
    def get_pano_ids_for_tile(self, tile: MapTileKey) -> list[PanoramaId]:
        ...


def pano_ids_from_coverage_objects(coverage_objects: list[object]) -> list[PanoramaId]:
    ids: dict[str, PanoramaId] = {}
    for item in coverage_objects:
        raw_id = getattr(item, "id", None)
        if raw_id is None:
            continue
        pano_id = str(raw_id)
        ids[pano_id] = PanoramaId(
            value=pano_id,
            latitude=getattr(item, "lat", None),
            longitude=getattr(item, "lon", None),
        )
    return [ids[pano_id] for pano_id in sorted(ids)]


class StreetLevelCoverageClient:
    def get_pano_ids_for_tile(self, tile: MapTileKey) -> list[PanoramaId]:
        coverage_objects = streetview.get_coverage_tile(tile.x, tile.y)
        return pano_ids_from_coverage_objects(coverage_objects)
