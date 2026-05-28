from dataclasses import dataclass
from typing import Sequence

from main_service.db.services.panorama_service import PanoramaService
from main_service.ingestion.coverage_client import CoverageClient
from main_service.ingestion.download_queue import PanoDownloadMessage, PanoDownloadQueue
from main_service.ingestion.types import MapTileKey, ProcessingStatus


@dataclass(frozen=True)
class DiscoveryResult:
    tiles_processed: int
    unique_panos_discovered: int
    tile_pano_links_created: int
    enqueued_downloads: int
    paused: bool = False
    pause_reason: str | None = None


def discover_panos_for_tiles(
    pano_service: PanoramaService,
    coverage_client: CoverageClient,
    download_queue: PanoDownloadQueue,
    tiles: Sequence[MapTileKey],
    max_downloader_queue_depth: int,
) -> DiscoveryResult:
    unique_pano_ids: set[str] = set()
    link_count = 0
    enqueued_count = 0
    tiles_processed = 0

    for tile in tiles:
        if download_queue.pending_count() >= max_downloader_queue_depth:
            return DiscoveryResult(
                tiles_processed=tiles_processed,
                unique_panos_discovered=len(unique_pano_ids),
                tile_pano_links_created=link_count,
                enqueued_downloads=enqueued_count,
                paused=True,
                pause_reason="downloader_queue_full",
            )

        tile_row = pano_service.upsert_map_tile(tile)
        if tile_row.discovery_status == ProcessingStatus.COMPLETE.value:
            pano_ids = pano_service.list_downloadable_pano_ids_for_map_tile(tile_row.id)
            for pano_id in pano_ids:
                pano_row = pano_service.upsert_discovered_panorama(pano_id)
                unique_pano_ids.add(pano_id.value)
                download_queue.enqueue(
                    PanoDownloadMessage(pano_id=pano_id, source_tile=tile)
                )
                pano_service.mark_panorama_download_queued(pano_row.id)
                enqueued_count += 1
            tiles_processed += 1
            continue

        pano_ids = coverage_client.get_pano_ids_for_tile(tile)
        for pano_id in pano_ids:
            pano_row = pano_service.upsert_discovered_panorama(pano_id)
            created = pano_service.link_map_tile_to_panorama(tile_row.id, pano_row.id)
            unique_pano_ids.add(pano_id.value)
            if created:
                link_count += 1
            download_queue.enqueue(PanoDownloadMessage(pano_id=pano_id, source_tile=tile))
            pano_service.mark_panorama_download_queued(pano_row.id)
            enqueued_count += 1
        pano_service.mark_map_tile_discovery_complete(tile_row.id)
        tiles_processed += 1

    return DiscoveryResult(
        tiles_processed=tiles_processed,
        unique_panos_discovered=len(unique_pano_ids),
        tile_pano_links_created=link_count,
        enqueued_downloads=enqueued_count,
    )
