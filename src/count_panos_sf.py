from __future__ import annotations
import asyncio
from typing import Tuple, Set, Dict, List

from streetlevel import streetview
from tqdm import tqdm

# Define the SF bounding box (upper-left and bottom-right)
UL: Tuple[float, float] = (37.806821748749, -122.52669542224751)  # (lat, lon)
BR: Tuple[float, float] = (37.70912377365498, -122.35466128724939)  # (lat, lon)
ZOOM = 17


def lonlat_to_tilexy(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    """Convert WGS84 lon/lat to XYZ tile coords at given zoom (Web Mercator)."""
    import math
    lat = max(-85.05112878, min(85.05112878, lat))
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int(
        (1.0 - math.log(
            math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))
        ) / math.pi) / 2.0 * n
    )
    return x, y


def tile_bounds_from_bbox(ul: Tuple[float, float], br: Tuple[float, float], zoom: int) -> Tuple[range, range]:
    """Return x-range and y-range (inclusive) of tiles covering the bbox."""
    ul_lat, ul_lon = ul
    br_lat, br_lon = br
    x1, y1 = lonlat_to_tilexy(ul_lon, ul_lat, zoom)
    x2, y2 = lonlat_to_tilexy(br_lon, br_lat, zoom)
    x_min, x_max = sorted((x1, x2))
    y_min, y_max = sorted((y1, y2))
    return range(x_min, x_max + 1), range(y_min, y_max + 1)


async def fetch_tile_coverage(x: int, y: int) -> List[object]:
    """Run get_coverage_tile in a worker thread to integrate with asyncio."""
    return await asyncio.to_thread(streetview.get_coverage_tile, x, y)


async def count_panos_sf() -> None:
    x_range, y_range = tile_bounds_from_bbox(UL, BR, ZOOM)
    tiles: List[Tuple[int, int]] = [(x, y) for y in y_range for x in x_range]
    total_tiles = len(tiles)

    # Shared state
    unique_ids: Set[str] = set()
    nonunique_sum = 0
    tiles_done = 0
    lock = asyncio.Lock()

    sem = asyncio.Semaphore(20)  # control concurrency

    async def worker(x: int, y: int, pbar: tqdm):
        nonlocal nonunique_sum, tiles_done
        try:
            async with sem:
                panos = await fetch_tile_coverage(x, y)
        except Exception as e:
            panos = []
        tile_count = len(panos)
        new_ids = {getattr(p, 'id', None) for p in panos if getattr(p, 'id', None)}
        async with lock:
            tiles_done += 1
            nonunique_sum += tile_count
            before = len(unique_ids)
            unique_ids.update(new_ids)
            avg = (nonunique_sum / tiles_done) if tiles_done else 0.0
            pbar.update(1)
            pbar.set_postfix(avg=f"{avg:.2f}", unique=len(unique_ids))

    print(f"Scanning tiles at z={ZOOM} for SF bbox UL={UL} BR={BR}")
    with tqdm(total=total_tiles, desc="Tiles", unit="tile") as pbar:
        tasks = [asyncio.create_task(worker(x, y, pbar)) for (x, y) in tiles]
        await asyncio.gather(*tasks)

    print("\nSummary:")
    print(f"Tiles processed: {tiles_done}/{total_tiles}")
    print(f"Total panos (unique by id): {len(unique_ids)}")
    print(f"Average panos per tile (non-unique): {nonunique_sum / max(1, tiles_done):.2f}")


if __name__ == "__main__":
    asyncio.run(count_panos_sf())
