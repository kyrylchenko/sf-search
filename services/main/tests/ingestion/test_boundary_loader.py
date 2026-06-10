from pathlib import Path

from main_service.config import Settings
from main_service.ingestion.boundary_loader import load_map_tiles_from_geojson


def test_load_map_tiles_from_geojson_uses_existing_geo_logic() -> None:
    tiles = load_map_tiles_from_geojson(Path("target.geojson"), zoom=17)

    assert len(tiles) == 2046
    assert tiles[0].z == 17
    assert min(tile.x for tile in tiles) == 20930
    assert max(tile.x for tile in tiles) == 20980
    assert min(tile.y for tile in tiles) == 50619
    assert max(tile.y for tile in tiles) == 50682
    assert tiles == sorted(tiles, key=lambda tile: (tile.z, tile.x, tile.y))


def test_load_map_tiles_from_geojson_can_randomize_valid_tile_order() -> None:
    sequential_tiles = load_map_tiles_from_geojson(
        Path("target.geojson"),
        zoom=17,
        order="sequential",
    )
    randomized_tiles = load_map_tiles_from_geojson(
        Path("target.geojson"),
        zoom=17,
        order="random",
        random_seed=123,
    )
    randomized_tiles_again = load_map_tiles_from_geojson(
        Path("target.geojson"),
        zoom=17,
        order="random",
        random_seed=123,
    )

    assert len(randomized_tiles) == len(sequential_tiles)
    assert set(randomized_tiles) == set(sequential_tiles)
    assert randomized_tiles == randomized_tiles_again
    assert randomized_tiles != sequential_tiles


def test_settings_exposes_discovery_defaults() -> None:
    settings = Settings(
        _env_file=None,
        db_user="user",
        db_password="password",
        db_host="localhost",
        db_port=3306,
        db_name="sf_search",
    )

    assert settings.map_tiles_zoom == 17
    assert settings.discovery_tile_order == "random"
    assert settings.discovery_tile_random_seed is None
    assert settings.discovery_concurrency == 20
    assert settings.max_attempts == 5
