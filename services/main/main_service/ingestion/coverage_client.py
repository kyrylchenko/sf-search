from typing import Protocol

from streetlevel import streetview

from main_service.ingestion.types import MapTileKey, PanoramaId


class CoverageClient(Protocol):
    def get_pano_ids_for_tile(self, tile: MapTileKey) -> list[PanoramaId]:
        ...


def pano_ids_from_coverage_objects(coverage_objects: list[object]) -> list[PanoramaId]:
    ids = {
        str(raw_id)
        for item in coverage_objects
        if (raw_id := getattr(item, "id", None)) is not None
    }
    return [PanoramaId(value=raw_id) for raw_id in sorted(ids)]


class StreetLevelCoverageClient:
    def get_pano_ids_for_tile(self, tile: MapTileKey) -> list[PanoramaId]:
        coverage_objects = streetview.get_coverage_tile(tile.x, tile.y)
        return pano_ids_from_coverage_objects(coverage_objects)
