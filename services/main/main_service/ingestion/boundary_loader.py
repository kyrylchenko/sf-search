import json
from pathlib import Path

from main_service.geo import generate_tiles_given_geojson
from main_service.ingestion.types import MapTileKey


def load_map_tiles_from_geojson(path: Path, zoom: int) -> list[MapTileKey]:
    geojson_data = json.loads(path.read_text())
    tiles = generate_tiles_given_geojson(geojson_data, zoom)
    return [
        MapTileKey(x=tile.x, y=tile.y, z=tile.z)
        for tile in sorted(tiles, key=lambda tile: (tile.z, tile.x, tile.y))
    ]
