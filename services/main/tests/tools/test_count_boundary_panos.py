import asyncio
import json
from pathlib import Path

from main_service.ingestion.types import MapTileKey, PanoramaId
from main_service.tools.count_boundary_panos import (
    BoundaryPanoCountResult,
    build_parser,
    count_boundary_panos,
)


class FakeCoverageClient:
    def __init__(self) -> None:
        self.calls: list[MapTileKey] = []

    def get_pano_ids_for_tile(self, tile: MapTileKey) -> list[PanoramaId]:
        self.calls.append(tile)
        if len(self.calls) == 2:
            return [
                PanoramaId("pano-b", latitude=37.1, longitude=-122.1),
                PanoramaId("pano-a", latitude=37.2, longitude=-122.2),
            ]
        return [PanoramaId("pano-a", latitude=37.2, longitude=-122.2)]


class FailingCoverageClient:
    def __init__(self) -> None:
        self.calls = 0

    def get_pano_ids_for_tile(self, tile: MapTileKey) -> list[PanoramaId]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("coverage failed")
        return [PanoramaId("pano-ok")]


def write_boundary(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [-122.458, 37.7745],
                                    [-122.458, 37.7643],
                                    [-122.438, 37.7643],
                                    [-122.438, 37.7745],
                                    [-122.458, 37.7745],
                                ]
                            ],
                        },
                    }
                ],
            }
        )
    )


def test_count_boundary_panos_deduplicates_coverage_ids(tmp_path: Path) -> None:
    boundary = tmp_path / "target.geojson"
    write_boundary(boundary)
    client = FakeCoverageClient()

    result = asyncio.run(
        count_boundary_panos(
            geojson_path=boundary,
            zoom=17,
            concurrency=3,
            limit_tiles=2,
            sample_size=10,
            coverage_client=client,
        )
    )

    assert isinstance(result, BoundaryPanoCountResult)
    assert result.tiles_total == 2
    assert result.tiles_scanned == 2
    assert result.failed_tiles == 0
    assert result.nonunique_pano_observations == 3
    assert result.unique_pano_count == 2
    assert result.sample_pano_ids == ["pano-a", "pano-b"]


def test_count_boundary_panos_tracks_failed_tiles(tmp_path: Path) -> None:
    boundary = tmp_path / "target.geojson"
    write_boundary(boundary)

    result = asyncio.run(
        count_boundary_panos(
            geojson_path=boundary,
            zoom=17,
            concurrency=1,
            limit_tiles=2,
            coverage_client=FailingCoverageClient(),
        )
    )

    assert result.tiles_total == 2
    assert result.tiles_scanned == 1
    assert result.failed_tiles == 1
    assert result.nonunique_pano_observations == 1
    assert result.unique_pano_count == 1


def test_parser_accepts_safe_count_options() -> None:
    args = build_parser().parse_args(
        [
            "--geojson",
            "target.geojson",
            "--zoom",
            "17",
            "--concurrency",
            "5",
            "--limit-tiles",
            "10",
            "--sample-size",
            "3",
            "--output",
            ".local/count.json",
        ]
    )

    assert args.geojson == Path("target.geojson")
    assert args.zoom == 17
    assert args.concurrency == 5
    assert args.limit_tiles == 10
    assert args.sample_size == 3
    assert args.output == Path(".local/count.json")
