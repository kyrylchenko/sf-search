from dataclasses import dataclass
import json
import logging
from typing import Sequence

from main_service.db.services.panorama_service import PanoramaService
from main_service.ingestion.coverage_client import CoverageClient
from main_service.ingestion.download_queue import PanoDownloadMessage, PanoDownloadQueue
from main_service.ingestion.types import MapTileKey, ProcessingStatus

logger = logging.getLogger(__name__)


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
        pending = download_queue.pending_count()
        logger.debug(
            "discovery_backpressure_check %s",
            json.dumps(
                {
                    "tile": tile.to_dict() if hasattr(tile, "to_dict") else str(tile),
                    "downloader_queue_pending": pending,
                    "max_downloader_queue_depth": max_downloader_queue_depth,
                },
                sort_keys=True,
            ),
        )
        if pending >= max_downloader_queue_depth:
            logger.warning(
                "discovery_paused %s",
                json.dumps({"pause_reason": "downloader_queue_full"}, sort_keys=True),
            )
            return DiscoveryResult(
                tiles_processed=tiles_processed,
                unique_panos_discovered=len(unique_pano_ids),
                tile_pano_links_created=link_count,
                enqueued_downloads=enqueued_count,
                paused=True,
                pause_reason="downloader_queue_full",
            )

        tile_row = pano_service.upsert_map_tile(tile)
        logger.info(
            "discovery_tile_start %s",
            json.dumps({"x": tile.x, "y": tile.y, "z": tile.z}, sort_keys=True),
        )
        if tile_row.discovery_status == ProcessingStatus.COMPLETE.value:
            logger.info(
                "discovery_tile_replay_start %s",
                json.dumps({"map_tile_id": tile_row.id}, sort_keys=True),
            )
            pano_ids = pano_service.list_downloadable_pano_ids_for_map_tile(tile_row.id)
            for pano_id in pano_ids:
                pano_row = pano_service.upsert_discovered_panorama(pano_id)
                unique_pano_ids.add(pano_id.value)
                download_queue.enqueue(
                    PanoDownloadMessage(pano_id=pano_id, source_tile=tile)
                )
                pano_service.mark_panorama_download_queued(pano_row.id)
                enqueued_count += 1
                logger.info(
                    "discovery_download_enqueued %s",
                    json.dumps({"pano_id": pano_id.value, "map_tile_id": tile_row.id}, sort_keys=True),
                )
            tiles_processed += 1
            continue

        logger.info(
            "discovery_coverage_fetch_start %s",
            json.dumps({"x": tile.x, "y": tile.y, "z": tile.z}, sort_keys=True),
        )
        pano_ids = coverage_client.get_pano_ids_for_tile(tile)
        logger.info(
            "discovery_coverage_fetch_complete %s",
            json.dumps({"x": tile.x, "y": tile.y, "z": tile.z, "pano_ids": len(pano_ids)}, sort_keys=True),
        )
        for pano_id in pano_ids:
            pano_row = pano_service.upsert_discovered_panorama(pano_id)
            created = pano_service.link_map_tile_to_panorama(tile_row.id, pano_row.id)
            unique_pano_ids.add(pano_id.value)
            if created:
                link_count += 1
            download_queue.enqueue(PanoDownloadMessage(pano_id=pano_id, source_tile=tile))
            pano_service.mark_panorama_download_queued(pano_row.id)
            enqueued_count += 1
            logger.info(
                "discovery_download_enqueued %s",
                json.dumps({"pano_id": pano_id.value, "map_tile_id": tile_row.id}, sort_keys=True),
            )
        pano_service.mark_map_tile_discovery_complete(tile_row.id)
        tiles_processed += 1
        logger.info(
            "discovery_tile_complete %s",
            json.dumps({"map_tile_id": tile_row.id, "pano_ids": len(pano_ids)}, sort_keys=True),
        )

    result = DiscoveryResult(
        tiles_processed=tiles_processed,
        unique_panos_discovered=len(unique_pano_ids),
        tile_pano_links_created=link_count,
        enqueued_downloads=enqueued_count,
    )
    logger.info("discovery_complete %s", json.dumps(result.__dict__, sort_keys=True))
    return result
