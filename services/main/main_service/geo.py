import shapely
import mercantile


def generate_tiles_given_geojson(geojson_data: dict, tiles_zoom: int):
    """Given geojson object and a zoom level of targets which needed to be extract
    generates tiles for that area
    """

    if geojson_data.get("features") is None or len(geojson_data["features"]) == 0:
        raise ValueError("geojson_data['features'] is None or the size is 0")
    if len(geojson_data["features"]) > 1:
        raise ValueError("len(geojson_data['features']) must be 1")

    source_geometry = geojson_data["features"][0]["geometry"]
    parsed_geometry = shapely.geometry.shape(source_geometry)
    all_area_tiles = mercantile.tiles(*parsed_geometry.bounds, tiles_zoom)

    filtered_tiles = set()
    for tile in all_area_tiles:
        tile_geometry = shapely.geometry.shape(mercantile.feature(tile)["geometry"])
        if tile_geometry.intersects(parsed_geometry):
            filtered_tiles.add(tile)

    return filtered_tiles
