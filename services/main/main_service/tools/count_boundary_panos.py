import argparse
import asyncio
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from main_service.ingestion.boundary_loader import load_map_tiles_from_geojson
from main_service.ingestion.coverage_client import StreetLevelCoverageClient
from main_service.ingestion.types import MapTileKey, PanoramaId


class CoverageClient(Protocol):
    def get_pano_ids_for_tile(self, tile: MapTileKey) -> list[PanoramaId]:
        ...


@dataclass(frozen=True)
class TileCoverageCount:
    tile: MapTileKey
    pano_ids: list[PanoramaId]
    error: str | None = None


@dataclass(frozen=True)
class BoundaryPanoCountResult:
    geojson_path: str
    zoom: int
    tiles_total: int
    tiles_scanned: int
    failed_tiles: int
    nonunique_pano_observations: int
    unique_pano_count: int
    sample_pano_ids: list[str]


async def count_boundary_panos(
    *,
    geojson_path: Path,
    zoom: int,
    concurrency: int,
    limit_tiles: int | None = None,
    sample_size: int = 20,
    coverage_client: CoverageClient | None = None,
) -> BoundaryPanoCountResult:
    tiles = load_map_tiles_from_geojson(geojson_path, zoom)
    if limit_tiles is not None:
        tiles = tiles[:limit_tiles]
    client = coverage_client or StreetLevelCoverageClient()
    semaphore = asyncio.Semaphore(max(1, concurrency))

    counts = await asyncio.gather(
        *[_count_tile(tile, client, semaphore) for tile in tiles]
    )
    unique_ids: set[str] = set()
    nonunique_observations = 0
    failed_tiles = 0
    for count in counts:
        if count.error is not None:
            failed_tiles += 1
            continue
        nonunique_observations += len(count.pano_ids)
        unique_ids.update(_pano_id_values(count.pano_ids))

    return BoundaryPanoCountResult(
        geojson_path=str(geojson_path),
        zoom=zoom,
        tiles_total=len(tiles),
        tiles_scanned=len(tiles) - failed_tiles,
        failed_tiles=failed_tiles,
        nonunique_pano_observations=nonunique_observations,
        unique_pano_count=len(unique_ids),
        sample_pano_ids=sorted(unique_ids)[: max(0, sample_size)],
    )


async def _count_tile(
    tile: MapTileKey,
    coverage_client: CoverageClient,
    semaphore: asyncio.Semaphore,
) -> TileCoverageCount:
    async with semaphore:
        try:
            pano_ids = await asyncio.to_thread(coverage_client.get_pano_ids_for_tile, tile)
            return TileCoverageCount(tile=tile, pano_ids=pano_ids)
        except Exception as exc:
            return TileCoverageCount(tile=tile, pano_ids=[], error=str(exc))


def _pano_id_values(pano_ids: Iterable[PanoramaId]) -> list[str]:
    return [pano_id.value for pano_id in pano_ids]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count unique Street View pano IDs inside a GeoJSON target boundary.",
    )
    parser.add_argument("--geojson", type=Path, default=Path("target.geojson"))
    parser.add_argument("--zoom", type=int, default=17)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument(
        "--limit-tiles",
        type=int,
        default=None,
        help="Only scan the first N tiles, useful for quick probes.",
    )
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for writing the JSON summary.",
    )
    return parser


async def run(args: argparse.Namespace) -> None:
    result = await count_boundary_panos(
        geojson_path=args.geojson,
        zoom=args.zoom,
        concurrency=args.concurrency,
        limit_tiles=args.limit_tiles,
        sample_size=args.sample_size,
    )
    output = json.dumps(asdict(result), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(output)


def main() -> None:
    asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
