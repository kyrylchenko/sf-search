import json
from pathlib import Path

import pytest

from main_service.geo import generate_tiles_given_geojson


def test_generate_tiles_for_target_geojson_returns_expected_count() -> None:
    data = json.loads(Path("target.geojson").read_text())

    tiles = generate_tiles_given_geojson(data, 17)

    assert len(tiles) == 2046
    assert all(tile.z == 17 for tile in tiles)
    assert min(tile.x for tile in tiles) == 20930
    assert max(tile.x for tile in tiles) == 20980
    assert min(tile.y for tile in tiles) == 50619
    assert max(tile.y for tile in tiles) == 50682


def test_generate_tiles_rejects_empty_feature_collection() -> None:
    with pytest.raises(ValueError, match="features"):
        generate_tiles_given_geojson({"type": "FeatureCollection", "features": []}, 17)


def test_generate_tiles_rejects_multiple_features() -> None:
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-122.0, 37.0],
                    [-122.0, 37.1],
                    [-121.9, 37.1],
                    [-121.9, 37.0],
                    [-122.0, 37.0],
                ]
            ],
        },
    }
    data = {"type": "FeatureCollection", "features": [feature, feature]}

    with pytest.raises(ValueError, match="must be 1"):
        generate_tiles_given_geojson(data, 17)
