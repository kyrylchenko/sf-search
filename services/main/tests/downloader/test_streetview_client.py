import asyncio
from pathlib import Path
from types import SimpleNamespace

from main_service.downloader.streetview_client import (
    STREETVIEW_SESSION_HEADERS,
    RealStreetViewClient,
    metadata_from_panorama,
)
from main_service.ingestion.types import PanoramaId


def test_streetview_session_headers_include_google_maps_referer() -> None:
    assert STREETVIEW_SESSION_HEADERS["Referer"] == "https://www.google.com/maps/"
    assert "Mozilla/5.0" in STREETVIEW_SESSION_HEADERS["User-Agent"]


def test_metadata_from_panorama_extracts_scalar_and_size_fields() -> None:
    pano = SimpleNamespace(
        id="resolved-pano-id",
        lat=37.1,
        lon=-122.1,
        heading=12.5,
        pitch=None,
        roll=0,
        elevation=44.1,
        date="2024-01",
        upload_date=None,
        is_third_party=False,
        country_code="US",
        source="google",
        tile_size=SimpleNamespace(x=512, y=512),
        image_sizes=[
            SimpleNamespace(x=512, y=256),
            SimpleNamespace(x=1024, y=512),
        ],
    )

    assert metadata_from_panorama(pano) == {
        "pano_id": "resolved-pano-id",
        "lat": 37.1,
        "lon": -122.1,
        "heading": 12.5,
        "pitch": None,
        "roll": 0,
        "elevation": 44.1,
        "date": "2024-01",
        "upload_date": None,
        "is_third_party": False,
        "country_code": "US",
        "source": "google",
        "tile_size": {"x": 512, "y": 512},
        "image_sizes_by_zoom": {
            "0": {"x": 512, "y": 256},
            "1": {"x": 1024, "y": 512},
        },
    }


def test_metadata_from_panorama_converts_non_json_scalar_values_to_strings() -> None:
    class CaptureDateLike:
        def __str__(self) -> str:
            return "2024-01"

    pano = SimpleNamespace(
        id="resolved-pano-id",
        lat=37.1,
        lon=-122.1,
        heading=None,
        pitch=None,
        roll=None,
        elevation=None,
        date=CaptureDateLike(),
        upload_date=None,
        is_third_party=False,
        country_code=None,
        source=None,
    )

    assert metadata_from_panorama(pano)["date"] == "2024-01"


def test_resolve_falls_back_to_location_when_pano_id_is_stale() -> None:
    calls: list[tuple[str, object]] = []
    resolved_pano = SimpleNamespace(id="fresh-pano-id", lat=37.1, lon=-122.1)

    async def find_by_id(pano_id: str, session: object) -> object | None:
        calls.append(("by_id", pano_id))
        return None

    async def find_by_location(
        lat: float,
        lon: float,
        session: object,
        radius: int,
    ) -> object | None:
        calls.append(("by_location", (lat, lon, radius)))
        return resolved_pano

    client = RealStreetViewClient(
        find_by_id=find_by_id,
        find_by_location=find_by_location,
    )

    result = asyncio.run(
        client.resolve(
            PanoramaId("stale-pano-id"),
            session=object(),
            latitude=37.1,
            longitude=-122.1,
        )
    )

    assert result is not None
    assert result.requested_pano_id == PanoramaId("stale-pano-id")
    assert result.resolved_pano_id == "fresh-pano-id"
    assert result.latitude == 37.1
    assert result.longitude == -122.1
    assert calls == [
        ("by_id", "stale-pano-id"),
        ("by_location", (37.1, -122.1, 25)),
    ]


def test_download_delegates_to_streetlevel_with_configured_zoom() -> None:
    calls: list[tuple[object, str, object, int]] = []
    pano = SimpleNamespace(id="pano-a")

    async def download_panorama(
        pano_arg: object,
        path: str,
        session: object,
        zoom: int,
    ) -> None:
        calls.append((pano_arg, path, session, zoom))

    client = RealStreetViewClient(download_panorama=download_panorama)
    session = object()

    asyncio.run(
        client.download(pano, Path(".local/panoramas/pano-a.jpg"), session, zoom=5)
    )

    assert calls == [(pano, ".local/panoramas/pano-a.jpg", session, 5)]
