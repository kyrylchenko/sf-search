import asyncio
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from main_service.tools.random_pano_downloader import (
    BoundingBox,
    RandomPanoDownloadResult,
    download_random_panos,
    parse_bbox,
    sample_coordinate,
)


class NullSession:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *args: object) -> None:
        return None


class FakeRandomPanoClient:
    def __init__(self) -> None:
        self.find_calls: list[tuple[float, float, int]] = []
        self.download_calls: list[Path] = []
        self._panos = [
            SimpleNamespace(id="pano-a", lat=37.1, lon=-122.1),
            SimpleNamespace(id="pano-a", lat=37.1, lon=-122.1),
            None,
            SimpleNamespace(id="pano-b", lat=37.2, lon=-122.2),
        ]

    async def find_nearest(
        self,
        latitude: float,
        longitude: float,
        session: object,
        *,
        radius: int,
    ) -> object | None:
        self.find_calls.append((latitude, longitude, radius))
        return self._panos.pop(0)

    async def download(
        self,
        pano: object,
        output_path: Path,
        session: object,
        *,
        zoom: int,
    ) -> None:
        self.download_calls.append(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"{pano.id}-{zoom}".encode("utf-8"))


def test_parse_bbox_validates_order() -> None:
    assert parse_bbox("-122.5,37.7,-122.3,37.8") == BoundingBox(
        west=-122.5,
        south=37.7,
        east=-122.3,
        north=37.8,
    )

    with pytest.raises(ValueError, match="west"):
        parse_bbox("-122.3,37.7,-122.5,37.8")


def test_sample_coordinate_stays_inside_bbox() -> None:
    bbox = BoundingBox(west=-122.5, south=37.7, east=-122.3, north=37.8)

    latitude, longitude = sample_coordinate(bbox, random.Random(123))

    assert bbox.south <= latitude <= bbox.north
    assert bbox.west <= longitude <= bbox.east


def test_download_random_panos_dedupes_and_writes_metadata(tmp_path: Path) -> None:
    client = FakeRandomPanoClient()

    result = asyncio.run(
        download_random_panos(
            count=2,
            bbox=BoundingBox(west=-122.5, south=37.7, east=-122.3, north=37.8),
            output_dir=tmp_path,
            max_attempts=10,
            radius=50,
            zoom=5,
            rng=random.Random(0),
            client=client,
            session_factory=NullSession,
        )
    )

    assert isinstance(result, RandomPanoDownloadResult)
    assert result.downloaded_count == 2
    assert result.duplicates == 1
    assert result.misses == 1
    assert result.attempts == 4
    assert [download.pano_id for download in result.downloads] == ["pano-a", "pano-b"]
    assert [path.name for path in client.download_calls] == ["pano-a.tmp.jpg", "pano-b.tmp.jpg"]

    metadata = json.loads((tmp_path / "pano-a.json").read_text())
    assert metadata["pano_id"] == "pano-a"
    assert metadata["resolved_pano_id"] == "pano-a"
    assert metadata["image_path"] == str(tmp_path / "pano-a.jpg")
    assert "requested_latitude" in metadata
    assert (tmp_path / "pano-a.jpg").read_bytes() == b"pano-a-5"
