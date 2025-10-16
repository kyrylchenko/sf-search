import math

def lonlat_to_tilexy(lon, lat, zoom):
    # Clamp latitude to Web Mercator bounds
    lat = max(-85.05112878, min(85.05112878, lat))
    
    n = 2 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int(
        (1.0 - math.log(
            math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))
            ) / math.pi) / 2.0 * n
    )
    return x_tile, y_tile

def tilexy_to_lonlat(x_tile, y_tile, zoom):
    """Convert tile coordinates back to longitude/latitude"""
    n = 2 ** zoom
    lon = x_tile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_tile / n)))
    lat = math.degrees(lat_rad)
    return lon, lat

# Example: longitude=-122.4280538, latitude=37.7837628 (San Francisco)
tile_x, tile_y = lonlat_to_tilexy(-122.4060301, 37.7879222, 17)
print(f"Tile coordinates: x={tile_x}, y={tile_y}")

# Convert back to lat/lng
lon, lat = tilexy_to_lonlat(tile_x, tile_y, 17)
print(f"Converted back: {lat}, {lon}")
print(f"Google Maps URL: https://www.google.com/maps/@{lat},{lon},17z")

