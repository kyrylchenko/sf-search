import argparse
import asyncio
import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, AsyncContextManager, Protocol

from streetlevel import streetview

from main_service.downloader.storage import (
    finalize_temp_file,
    pano_image_path,
    sha256_file,
    temp_pano_image_path,
)
from main_service.downloader.streetview_client import (
    create_streetview_session,
    metadata_from_panorama,
)
from main_service.ingestion.types import PanoramaId


DEFAULT_SF_BBOX = "-122.515,37.708,-122.356,37.833"


@dataclass(frozen=True)
class BoundingBox:
    west: float
    south: float
    east: float
    north: float


@dataclass(frozen=True)
class RandomPanoDownload:
    pano_id: str
    image_path: str
    metadata_path: str
    requested_latitude: float
    requested_longitude: float
    resolved_latitude: float | None
    resolved_longitude: float | None
    image_hash: str


@dataclass(frozen=True)
class RandomPanoDownloadResult:
    requested_count: int
    downloaded_count: int
    attempts: int
    duplicates: int
    misses: int
    downloads: list[RandomPanoDownload]


class RandomPanoClient(Protocol):
    async def find_nearest(
        self,
        latitude: float,
        longitude: float,
        session: object,
        *,
        radius: int,
    ) -> object | None:
        ...

    async def download(
        self,
        pano: object,
        output_path: Path,
        session: object,
        *,
        zoom: int,
    ) -> None:
        ...


class StreetLevelRandomPanoClient:
    async def find_nearest(
        self,
        latitude: float,
        longitude: float,
        session: object,
        *,
        radius: int,
    ) -> object | None:
        return await streetview.find_panorama_async(
            latitude,
            longitude,
            session,
            radius=radius,
        )

    async def download(
        self,
        pano: object,
        output_path: Path,
        session: object,
        *,
        zoom: int,
    ) -> None:
        await streetview.download_panorama_async(pano, str(output_path), session, zoom=zoom)


SessionFactory = Callable[[], AsyncContextManager[object]]


async def download_random_panos(
    *,
    count: int,
    bbox: BoundingBox,
    output_dir: Path,
    max_attempts: int,
    radius: int,
    zoom: int,
    rng: random.Random,
    client: RandomPanoClient | None = None,
    session_factory: SessionFactory = create_streetview_session,
) -> RandomPanoDownloadResult:
    resolved_client = client or StreetLevelRandomPanoClient()
    output_dir.mkdir(parents=True, exist_ok=True)
    downloads: list[RandomPanoDownload] = []
    seen_pano_ids: set[str] = existing_pano_ids(output_dir)
    duplicates = 0
    misses = 0
    attempts = 0

    async with session_factory() as session:
        while len(downloads) < count and attempts < max_attempts:
            attempts += 1
            latitude, longitude = sample_coordinate(bbox, rng)
            pano = await resolved_client.find_nearest(
                latitude,
                longitude,
                session,
                radius=radius,
            )
            if pano is None:
                misses += 1
                continue

            pano_id = str(getattr(pano, "id", ""))
            if pano_id == "":
                misses += 1
                continue
            if pano_id in seen_pano_ids:
                duplicates += 1
                continue
            seen_pano_ids.add(pano_id)

            final_path = pano_image_path(output_dir, PanoramaId(pano_id))
            temp_path = temp_pano_image_path(final_path)
            await resolved_client.download(pano, temp_path, session, zoom=zoom)
            finalize_temp_file(temp_path, final_path)
            image_hash = sha256_file(final_path)

            metadata = metadata_from_panorama(pano)
            metadata.update(
                {
                    "requested_latitude": latitude,
                    "requested_longitude": longitude,
                    "resolved_pano_id": pano_id,
                    "image_path": str(final_path),
                    "image_hash": image_hash,
                }
            )
            metadata_path = final_path.with_suffix(".json")
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

            downloads.append(
                RandomPanoDownload(
                    pano_id=pano_id,
                    image_path=str(final_path),
                    metadata_path=str(metadata_path),
                    requested_latitude=latitude,
                    requested_longitude=longitude,
                    resolved_latitude=_optional_float(getattr(pano, "lat", None)),
                    resolved_longitude=_optional_float(getattr(pano, "lon", None)),
                    image_hash=image_hash,
                )
            )

    return RandomPanoDownloadResult(
        requested_count=count,
        downloaded_count=len(downloads),
        attempts=attempts,
        duplicates=duplicates,
        misses=misses,
        downloads=downloads,
    )


def parse_bbox(value: str) -> BoundingBox:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be west,south,east,north")
    bbox = BoundingBox(west=parts[0], south=parts[1], east=parts[2], north=parts[3])
    if bbox.west >= bbox.east:
        raise ValueError("bbox west must be less than east")
    if bbox.south >= bbox.north:
        raise ValueError("bbox south must be less than north")
    return bbox


def sample_coordinate(bbox: BoundingBox, rng: random.Random) -> tuple[float, float]:
    return (
        rng.uniform(bbox.south, bbox.north),
        rng.uniform(bbox.west, bbox.east),
    )


def existing_pano_ids(output_dir: Path) -> set[str]:
    return {
        path.stem
        for path in output_dir.glob("*.jpg")
        if path.is_file() and not path.name.endswith(".tmp.jpg")
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download panos nearest to random coordinates for visual inspection.",
    )
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--bbox", default=DEFAULT_SF_BBOX, help="west,south,east,north")
    parser.add_argument("--output-dir", type=Path, default=Path(".local/random-panoramas"))
    parser.add_argument("--max-attempts", type=int, default=120)
    parser.add_argument("--radius", type=int, default=75)
    parser.add_argument("--zoom", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    return parser


async def run(args: argparse.Namespace) -> None:
    result = await download_random_panos(
        count=args.count,
        bbox=parse_bbox(args.bbox),
        output_dir=args.output_dir,
        max_attempts=args.max_attempts,
        radius=args.radius,
        zoom=args.zoom,
        rng=random.Random(args.seed),
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


def main() -> None:
    asyncio.run(run(build_parser().parse_args()))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


if __name__ == "__main__":
    main()
