from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from main_service.db.models.map_tile import MapTile
from main_service.db.models.map_tile_panorama import MapTilePanorama
from main_service.db.models.embedding import Embedding
from main_service.db.models.panorama import Panorama
from main_service.ingestion.types import (
    DownloadStatus,
    MapTileKey,
    PanoramaId,
    ProcessingStatus,
)

from ..models.tile import Tile


@dataclass(frozen=True)
class DownloadQueueCandidate:
    panorama_id: int
    pano_id: PanoramaId
    source_tile: MapTileKey


class PanoramaService:
    def __init__(self, engine: Engine):
        self.engine = engine

    def create_tile(self, tile: Tile):
        with Session(self.engine) as session:
            session.add(tile)
            session.commit()

    def create_embedding(self, embedding: Embedding):
        with Session(self.engine) as session:
            session.add(embedding)
            session.commit()

    def find_panorama_by_orig_id(self, orig_id: str) -> Optional[Panorama]:
        with Session(self.engine) as session:
            return session.execute(
                select(Panorama).filter_by(orig_id=orig_id)
            ).scalar_one_or_none()

    def upsert_map_tile(self, key: MapTileKey) -> MapTile:
        with Session(self.engine) as session:
            tile = session.execute(
                select(MapTile).filter_by(x=key.x, y=key.y, z=key.z)
            ).scalar_one_or_none()
            if tile is None:
                tile = MapTile(
                    x=key.x,
                    y=key.y,
                    z=key.z,
                    discovery_status=ProcessingStatus.PENDING.value,
                )
                session.add(tile)
                session.flush()
                session.refresh(tile)
                session.expunge(tile)
                session.commit()
                return tile

            session.expunge(tile)
            return tile

    def upsert_discovered_panorama(self, pano_id: PanoramaId) -> Panorama:
        with Session(self.engine) as session:
            panorama = session.execute(
                select(Panorama).filter_by(orig_id=pano_id.value)
            ).scalar_one_or_none()
            if panorama is None:
                panorama = Panorama(
                    orig_id=pano_id.value,
                    image_hash=None,
                    latitude=pano_id.latitude,
                    longitude=pano_id.longitude,
                    metadata_status=ProcessingStatus.PENDING.value,
                    download_status=DownloadStatus.PENDING.value,
                )
                session.add(panorama)
                session.flush()
                session.refresh(panorama)
                session.expunge(panorama)
                session.commit()
                return panorama

            if panorama.latitude is None and pano_id.latitude is not None:
                panorama.latitude = pano_id.latitude
            if panorama.longitude is None and pano_id.longitude is not None:
                panorama.longitude = pano_id.longitude
            session.flush()
            session.refresh(panorama)
            session.expunge(panorama)
            session.commit()
            return panorama

    def link_map_tile_to_panorama(self, map_tile_id: int, panorama_id: int) -> bool:
        with Session(self.engine) as session:
            link = session.execute(
                select(MapTilePanorama).filter_by(
                    map_tile_id=map_tile_id,
                    panorama_id=panorama_id,
                )
            ).scalar_one_or_none()
            if link is not None:
                return False

            link = MapTilePanorama(
                map_tile_id=map_tile_id,
                panorama_id=panorama_id,
            )
            session.add(link)
            panorama = session.get(Panorama, panorama_id)
            if panorama is not None:
                panorama.discovered_at_tile_count += 1
            session.commit()
            return True

    def count_map_tile_panorama_links(self) -> int:
        with Session(self.engine) as session:
            return session.execute(
                select(func.count()).select_from(MapTilePanorama)
            ).scalar_one()

    def mark_map_tile_discovery_complete(self, map_tile_id: int) -> MapTile:
        with Session(self.engine) as session:
            tile = session.get(MapTile, map_tile_id)
            if tile is None:
                raise ValueError(f"Map tile does not exist: {map_tile_id}")

            tile.discovery_status = ProcessingStatus.COMPLETE.value
            tile.last_error = None
            session.flush()
            session.refresh(tile)
            session.expunge(tile)
            session.commit()
            return tile

    def mark_map_tile_discovery_failed(self, map_tile_id: int, error: str) -> MapTile:
        with Session(self.engine) as session:
            tile = session.get(MapTile, map_tile_id)
            if tile is None:
                raise ValueError(f"Map tile does not exist: {map_tile_id}")

            tile.discovery_status = ProcessingStatus.FAILED.value
            tile.attempt_count += 1
            tile.last_error = error
            session.flush()
            session.refresh(tile)
            session.expunge(tile)
            session.commit()
            return tile

    def mark_panorama_download_queued(self, panorama_id: int) -> Panorama:
        with Session(self.engine) as session:
            panorama = session.get(Panorama, panorama_id)
            if panorama is None:
                raise ValueError(f"Panorama does not exist: {panorama_id}")
            if panorama.download_status in {
                DownloadStatus.DOWNLOADED.value,
                DownloadStatus.SKIPPED.value,
            }:
                session.expunge(panorama)
                return panorama

            panorama.download_status = DownloadStatus.QUEUED.value
            session.flush()
            session.refresh(panorama)
            session.expunge(panorama)
            session.commit()
            return panorama

    def mark_panorama_download_status(
        self, panorama_id: int, status: DownloadStatus
    ) -> Panorama:
        with Session(self.engine) as session:
            panorama = session.get(Panorama, panorama_id)
            if panorama is None:
                raise ValueError(f"Panorama does not exist: {panorama_id}")

            panorama.download_status = status.value
            session.flush()
            session.refresh(panorama)
            session.expunge(panorama)
            session.commit()
            return panorama

    def claim_panorama_for_download(self, pano_id: PanoramaId) -> Panorama | None:
        with Session(self.engine) as session:
            panorama = session.execute(
                select(Panorama).filter_by(orig_id=pano_id.value)
            ).scalar_one_or_none()
            if panorama is None:
                panorama = Panorama(
                    orig_id=pano_id.value,
                    image_hash=None,
                    latitude=None,
                    longitude=None,
                    metadata_status=ProcessingStatus.PENDING.value,
                    download_status=DownloadStatus.PENDING.value,
                )
                session.add(panorama)
                session.flush()

            if panorama.download_status == DownloadStatus.SKIPPED.value:
                return None

            if (
                panorama.download_status == DownloadStatus.DOWNLOADED.value
                and panorama.image_path
                and panorama.image_hash
            ):
                return None

            panorama.download_status = DownloadStatus.DOWNLOADING.value
            panorama.last_error = None
            session.flush()
            session.refresh(panorama)
            session.expunge(panorama)
            session.commit()
            return panorama

    def is_panorama_download_complete(self, pano_id: PanoramaId) -> bool:
        with Session(self.engine) as session:
            panorama = session.execute(
                select(Panorama).filter_by(orig_id=pano_id.value)
            ).scalar_one_or_none()
            return bool(
                panorama is not None
                and panorama.download_status == DownloadStatus.DOWNLOADED.value
                and panorama.image_path
                and panorama.image_hash
            )

    def mark_panorama_downloaded(
        self,
        pano_id: PanoramaId,
        *,
        image_path: str,
        image_hash: str,
        metadata_json: dict[str, object] | None,
        latitude: float | None,
        longitude: float | None,
    ) -> Panorama:
        with Session(self.engine) as session:
            panorama = session.execute(
                select(Panorama).filter_by(orig_id=pano_id.value)
            ).scalar_one_or_none()
            if panorama is None:
                raise ValueError(f"Panorama does not exist: {pano_id.value}")

            panorama.download_status = DownloadStatus.DOWNLOADED.value
            panorama.image_path = image_path
            panorama.image_hash = image_hash
            panorama.metadata_json = metadata_json
            panorama.latitude = latitude
            panorama.longitude = longitude
            panorama.downloaded_at = datetime.now(timezone.utc)
            panorama.last_error = None
            session.flush()
            session.refresh(panorama)
            session.expunge(panorama)
            session.commit()
            return panorama

    def mark_panorama_download_failed(
        self,
        pano_id: PanoramaId,
        *,
        error: str,
        max_attempts: int,
    ) -> Panorama:
        with Session(self.engine) as session:
            panorama = session.execute(
                select(Panorama).filter_by(orig_id=pano_id.value)
            ).scalar_one_or_none()
            if panorama is None:
                raise ValueError(f"Panorama does not exist: {pano_id.value}")

            panorama.attempt_count += 1
            panorama.last_error = error[:2000]
            if panorama.attempt_count >= max_attempts:
                panorama.download_status = DownloadStatus.SKIPPED.value
            else:
                panorama.download_status = DownloadStatus.FAILED.value
            session.flush()
            session.refresh(panorama)
            session.expunge(panorama)
            session.commit()
            return panorama

    def list_downloadable_pano_ids_for_map_tile(
        self, map_tile_id: int
    ) -> list[PanoramaId]:
        with Session(self.engine) as session:
            rows = session.execute(
                select(Panorama.orig_id)
                .join(
                    MapTilePanorama,
                    MapTilePanorama.panorama_id == Panorama.id,
                )
                .where(MapTilePanorama.map_tile_id == map_tile_id)
                .where(
                    Panorama.download_status.not_in(
                        [
                            DownloadStatus.DOWNLOADED.value,
                            DownloadStatus.SKIPPED.value,
                        ]
                    )
                )
                .order_by(Panorama.orig_id)
            ).all()

            return [PanoramaId(value=orig_id) for (orig_id,) in rows]

    def list_download_queue_candidates(self, limit: int) -> list[DownloadQueueCandidate]:
        first_tile_link = (
            select(
                MapTilePanorama.panorama_id.label("panorama_id"),
                func.min(MapTilePanorama.map_tile_id).label("map_tile_id"),
            )
            .group_by(MapTilePanorama.panorama_id)
            .subquery()
        )
        retryable_statuses = [
            DownloadStatus.PENDING.value,
            DownloadStatus.QUEUED.value,
            DownloadStatus.FAILED.value,
        ]
        with Session(self.engine) as session:
            rows = (
                session.execute(
                    select(
                        Panorama.id,
                        Panorama.orig_id,
                        Panorama.latitude,
                        Panorama.longitude,
                        MapTile.x,
                        MapTile.y,
                        MapTile.z,
                    )
                    .join(first_tile_link, first_tile_link.c.panorama_id == Panorama.id)
                    .join(MapTile, MapTile.id == first_tile_link.c.map_tile_id)
                    .where(Panorama.download_status.in_(retryable_statuses))
                    .order_by(Panorama.id)
                    .limit(limit)
                )
                .all()
            )

        return [
            DownloadQueueCandidate(
                panorama_id=panorama_id,
                pano_id=PanoramaId(
                    value=orig_id,
                    latitude=latitude,
                    longitude=longitude,
                ),
                source_tile=MapTileKey(x=x, y=y, z=z),
            )
            for panorama_id, orig_id, latitude, longitude, x, y, z in rows
        ]
