from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from aiohttp import ClientSession
from streetlevel import streetview

from main_service.ingestion.types import PanoramaId

STREETVIEW_SESSION_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/137.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.google.com/maps/",
}

FindById = Callable[[str, Any], Awaitable[Any | None]]
FindByLocation = Callable[[float, float, Any, int], Awaitable[Any | None]]
DownloadPanorama = Callable[[Any, str, Any, int], Awaitable[None]]


@dataclass(frozen=True)
class ResolvedPanorama:
    requested_pano_id: PanoramaId
    resolved_pano_id: str
    panorama: Any
    latitude: float | None
    longitude: float | None
    metadata_json: dict[str, object]


class StreetViewClient(Protocol):
    async def resolve(
        self,
        pano_id: PanoramaId,
        session: ClientSession,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> ResolvedPanorama | None:
        ...

    async def download(
        self,
        panorama: Any,
        output_path: Path,
        session: ClientSession,
        *,
        zoom: int,
    ) -> None:
        ...


class RealStreetViewClient:
    def __init__(
        self,
        *,
        find_by_id: FindById | None = None,
        find_by_location: FindByLocation | None = None,
        download_panorama: DownloadPanorama | None = None,
    ) -> None:
        self._find_by_id = find_by_id or streetview.find_panorama_by_id_async
        self._find_by_location = find_by_location or _find_panorama_by_location
        self._download_panorama = download_panorama or _download_panorama

    async def resolve(
        self,
        pano_id: PanoramaId,
        session: ClientSession,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> ResolvedPanorama | None:
        pano = await self._find_by_id(pano_id.value, session)
        if pano is None and latitude is not None and longitude is not None:
            pano = await self._find_by_location(latitude, longitude, session, 25)
        if pano is None:
            return None

        metadata = metadata_from_panorama(pano)
        return ResolvedPanorama(
            requested_pano_id=pano_id,
            resolved_pano_id=str(getattr(pano, "id", pano_id.value)),
            panorama=pano,
            latitude=_optional_float(getattr(pano, "lat", latitude)),
            longitude=_optional_float(getattr(pano, "lon", longitude)),
            metadata_json=metadata,
        )

    async def download(
        self,
        panorama: Any,
        output_path: Path,
        session: ClientSession,
        *,
        zoom: int,
    ) -> None:
        await self._download_panorama(panorama, str(output_path), session, zoom)


async def _find_panorama_by_location(
    lat: float,
    lon: float,
    session: ClientSession,
    radius: int,
) -> Any | None:
    return await streetview.find_panorama_async(lat, lon, session, radius=radius)


async def _download_panorama(
    pano: Any,
    path: str,
    session: ClientSession,
    zoom: int,
) -> None:
    await streetview.download_panorama_async(pano, path, session, zoom=zoom)


def create_streetview_session() -> ClientSession:
    return ClientSession(headers=STREETVIEW_SESSION_HEADERS)


def metadata_from_panorama(pano: Any) -> dict[str, object]:
    metadata: dict[str, object] = {
        "pano_id": getattr(pano, "id", None),
        "lat": getattr(pano, "lat", None),
        "lon": getattr(pano, "lon", None),
        "heading": getattr(pano, "heading", None),
        "pitch": getattr(pano, "pitch", None),
        "roll": getattr(pano, "roll", None),
        "elevation": getattr(pano, "elevation", None),
        "date": getattr(pano, "date", None),
        "upload_date": getattr(pano, "upload_date", None),
        "is_third_party": getattr(pano, "is_third_party", None),
        "country_code": getattr(pano, "country_code", None),
        "source": getattr(pano, "source", None),
    }

    tile_size = getattr(pano, "tile_size", None)
    if tile_size is not None:
        metadata["tile_size"] = _size_to_dict(tile_size)

    image_sizes = getattr(pano, "image_sizes", None)
    if image_sizes is not None:
        metadata["image_sizes_by_zoom"] = {
            str(index): _size_to_dict(size) for index, size in enumerate(image_sizes)
        }

    return metadata


def _size_to_dict(size: Any) -> dict[str, int | None]:
    return {
        "x": getattr(size, "x", None),
        "y": getattr(size, "y", None),
    }


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
