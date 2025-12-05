from main_service.config import CONFIG
from main_service.geo import generate_tiles_given_geojson
import json

print("config maain", CONFIG)


def main():
    with open(CONFIG.area_to_process_geojson_filename) as file:
        geojson_data: dict = json.loads(file.read())
    tiles = generate_tiles_given_geojson(geojson_data, 17)
    print(tiles)
    print(f"Found {len(tiles)}")


if __name__ == "__main__":
    main()
