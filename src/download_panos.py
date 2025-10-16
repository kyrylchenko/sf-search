from streetlevel import streetview
from aiohttp import ClientSession
import asyncio
from pathlib import Path
from typing import List, Tuple, Set, Dict, Optional
import os
import math

from tile import lonlat_to_tilexy, tilexy_to_lonlat


OUT_DIR = Path("/Users/illia/Desktop/ai-test/panos")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ZOOM = 17  # tile zoom used for coverage queries

# Limit downloads to the top-K closest panos per tile.
# Set to None to download all panos found in a tile.
TOP_K_PER_TILE: Optional[int] = 5

# Optional: allow overriding via environment variable TOP_K_PER_TILE
env_k = os.getenv("TOP_K_PER_TILE")
if env_k:
    try:
        TOP_K_PER_TILE = int(env_k)
    except ValueError:
        pass


COORDS: List[Tuple[float, float]] = [
    (37.7661053, -122.5129298),
    (37.7542543, -122.5086315),
    (37.7508628, -122.4500667),
    (37.7507338, -122.4177571),
    (37.7555293, -122.3851970),
    (37.7944367, -122.4014331),
]


async def download_single_pano(pano, semaphore: asyncio.Semaphore, session: ClientSession) -> bool:
    async with semaphore:
        try:
            # Resolve to a fully populated panorama (if needed) and download
            pano_real = await streetview.find_panorama_by_id_async(pano.id, session)
            out_path = OUT_DIR / f"{pano.id}.jpg"
            if out_path.exists():
                return True  # skip existing
            await streetview.download_panorama_async(pano_real, str(out_path), session)
            print(f"Downloaded {pano.id}")
            return True
        except Exception as e:
            print(f"Failed to download {getattr(pano, 'id', 'unknown')}: {e}")
            return False


def tiles_from_coords(coords: List[Tuple[float, float]], zoom: int) -> Set[Tuple[int, int]]:
    tiles: Set[Tuple[int, int]] = set()
    for lat, lon in coords:
        x, y = lonlat_to_tilexy(lon, lat, zoom)
        tiles.add((x, y))
    return tiles


def tile_to_seed_coords(coords: List[Tuple[float, float]], zoom: int) -> Dict[Tuple[int, int], List[Tuple[float, float]]]:
    """Group the input coordinates by their tile at the given zoom."""
    by_tile: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    for lat, lon in coords:
        x, y = lonlat_to_tilexy(lon, lat, zoom)
        by_tile.setdefault((x, y), []).append((lat, lon))
    return by_tile


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0  # meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def pano_latlon(p) -> Tuple[Optional[float], Optional[float]]:
    """Try to extract (lat, lon) from a panorama object."""
    lat = getattr(p, 'lat', None)
    lon = getattr(p, 'lon', None)
    if lat is not None and lon is not None:
        return float(lat), float(lon)
    loc = getattr(p, 'location', None)
    if loc is not None:
        lat = getattr(loc, 'lat', None)
        lon = getattr(loc, 'lon', None)
        if lat is not None and lon is not None:
            return float(lat), float(lon)
    return None, None


async def main():
    tiles = tiles_from_coords(COORDS, ZOOM)
    print(f"Unique tiles from coords: {len(tiles)} → {sorted(list(tiles))[:5]}…")

    seeds_by_tile = tile_to_seed_coords(COORDS, ZOOM)

    # Collect panoramas from all tiles and deduplicate by id
    panos_by_id: Dict[str, object] = {}
    total_tile_panos = 0
    total_selected_panos = 0
    for x, y in tiles:
        try:
            tile_panos = streetview.get_coverage_tile(x, y)
            total_tile_panos += len(tile_panos)
            # If limiting to top-K closest, sort by min distance to any seed coord in this tile
            selected = tile_panos
            if TOP_K_PER_TILE is not None and TOP_K_PER_TILE >= 0:
                seeds = seeds_by_tile.get((x, y), [])
                # If no seed coords map to this tile (edge case), fall back to tile center
                if not seeds:
                    lon_c, lat_c = tilexy_to_lonlat(x + 0.5, y + 0.5, ZOOM)
                    seeds = [(lat_c, lon_c)]

                def pano_min_dist_m(p) -> float:
                    plat, plon = pano_latlon(p)
                    if plat is None or plon is None:
                        # Fallback: distance to tile center if pano lacks coordinates
                        lon_c2, lat_c2 = tilexy_to_lonlat(x + 0.5, y + 0.5, ZOOM)
                        return min(haversine_m(lat_c2, lon_c2, slat, slon) for slat, slon in seeds)
                    return min(haversine_m(plat, plon, slat, slon) for slat, slon in seeds)

                selected = sorted(tile_panos, key=pano_min_dist_m)[:TOP_K_PER_TILE or None]

            total_selected_panos += len(selected)
            for p in selected:
                panos_by_id[getattr(p, 'id', None)] = p
        except Exception as e:
            print(f"Failed to get coverage for tile ({x},{y}): {e}")

    panos = [p for k, p in panos_by_id.items() if k]
    print(
        f"Tiles queried: {len(tiles)} | Panos found in tiles: {total_tile_panos}"
        f" | Selected per K: {total_selected_panos} | Unique panos: {len(panos)}"
        + (f" | Top-K per tile: {TOP_K_PER_TILE}" if TOP_K_PER_TILE is not None else " | Top-K per tile: ALL")
    )

    if not panos:
        print("No panoramas found for given coordinates.")
        return

    # Limit concurrency to avoid overwhelming the server
    semaphore = asyncio.Semaphore(5)
    async with ClientSession() as session:
        tasks = [download_single_pano(p, semaphore, session) for p in panos]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for r in results if r is True)
    print(f"Successfully downloaded: {successful}/{len(panos)}")


if __name__ == "__main__":
    asyncio.run(main())