import json
from pathlib import Path
import random
from collections.abc import Sequence
from typing import Literal

from main_service.geo import generate_tiles_given_geojson
from main_service.ingestion.types import MapTileKey

MapTileOrder = Literal["sequential", "random"]


def load_map_tiles_from_geojson(
    path: Path,
    zoom: int,
    *,
    order: MapTileOrder = "sequential",
    random_seed: int | None = None,
) -> list[MapTileKey]:
    geojson_data = json.loads(path.read_text())
    generated_tiles = generate_tiles_given_geojson(geojson_data, zoom)
    tiles = [
        MapTileKey(x=tile.x, y=tile.y, z=tile.z)
        for tile in generated_tiles
    ]
    return order_map_tiles(tiles, order=order, random_seed=random_seed)


def order_map_tiles(
    tiles: Sequence[MapTileKey],
    *,
    order: MapTileOrder,
    random_seed: int | None = None,
) -> list[MapTileKey]:
    ordered_tiles = sorted(tiles, key=lambda tile: (tile.z, tile.x, tile.y))
    if order == "sequential":
        return ordered_tiles
    if order == "random":
        rng = random.Random(random_seed)
        rng.shuffle(ordered_tiles)
        return ordered_tiles
    raise ValueError(f"Unsupported map tile order: {order}")
