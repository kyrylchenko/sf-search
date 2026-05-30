from pathlib import Path

from main_service.config import Settings
from main_service.ingestion.boundary_loader import load_map_tiles_from_geojson


def test_load_map_tiles_from_geojson_uses_existing_geo_logic() -> None:
    tiles = load_map_tiles_from_geojson(Path("target.geojson"), zoom=17)

    assert len(tiles) == 40
    assert tiles[0].z == 17
    assert min(tile.x for tile in tiles) == 20950
    assert max(tile.x for tile in tiles) == 20957
    assert min(tile.y for tile in tiles) == 50662
    assert max(tile.y for tile in tiles) == 50666
    assert tiles == sorted(tiles, key=lambda tile: (tile.z, tile.x, tile.y))


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
    assert settings.discovery_concurrency == 20
    assert settings.max_attempts == 5
