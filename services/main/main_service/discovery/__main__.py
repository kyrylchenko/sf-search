import argparse
import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path

from main_service.config import CONFIG
from main_service.db.initialize_engine import initialize_engine
from main_service.db.models.base import Base
from main_service.db.services.panorama_service import PanoramaService
from main_service.ingestion.boundary_loader import load_map_tiles_from_geojson
from main_service.ingestion.coverage_client import StreetLevelCoverageClient
from main_service.ingestion.discovery import DiscoveryResult, discover_panos_for_tiles
from main_service.ingestion.download_queue import NatsJetStreamPanoDownloadQueue
from main_service.logging_config import configure_cli_logging
from main_service.service_loop import run_service_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run coverage discovery service.")
    parser.add_argument(
        "--geojson",
        default=None,
        help="Boundary GeoJSON path. Defaults to AREA_TO_PROCESS_GEOJSON_FILEPATH.",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=None,
        help="Map tile zoom for coverage discovery.",
    )
    parser.add_argument(
        "--max-downloader-queue-depth",
        type=int,
        default=None,
        help="Pause before discovering another tile when downloader backlog reaches this value.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Console log level: DEBUG, INFO, WARNING, ERROR.",
    )
    parser.add_argument(
        "--idle-sleep-seconds",
        type=float,
        default=None,
        help="Seconds to sleep after a complete or paused discovery pass.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one discovery pass and exit. Default is to keep polling forever.",
    )
    return parser


async def run(args: argparse.Namespace) -> None:
    settings = CONFIG
    configure_cli_logging(args.log_level or settings.log_level)
    logger = logging.getLogger(__name__)
    geojson_path = Path(args.geojson or settings.area_to_process_geojson_filepath)
    zoom = _value_or_default(args.zoom, settings.map_tiles_zoom)
    logger.info("🗺️ discovery_cli_start geojson=%s zoom=%s", geojson_path, zoom)

    engine = initialize_engine(settings)
    Base.metadata.create_all(engine)
    pano_service = PanoramaService(engine)
    coverage_client = StreetLevelCoverageClient()
    download_queue = NatsJetStreamPanoDownloadQueue.connect(
        servers=settings.nats_url,
        stream_name=settings.pano_download_stream,
        subject=settings.pano_download_subject,
        consumer_name=settings.pano_downloader_consumer,
    )
    tiles = load_map_tiles_from_geojson(geojson_path, zoom)
    logger.info("🧱 discovery_tiles_loaded count=%s", len(tiles))

    try:
        async def run_batch() -> DiscoveryResult:
            result = discover_panos_for_tiles(
                pano_service=pano_service,
                coverage_client=coverage_client,
                download_queue=download_queue,
                tiles=tiles,
                max_downloader_queue_depth=_value_or_default(
                    args.max_downloader_queue_depth,
                    settings.max_downloader_queue_depth,
                ),
            )
            logger.info(
                "📦 discovery_cli_batch_complete result=%s",
                json.dumps(asdict(result), sort_keys=True),
            )
            print(json.dumps(asdict(result), sort_keys=True), flush=True)
            return result

        await run_service_loop(
            service_name="discovery",
            run_batch=run_batch,
            should_idle=lambda result: result.paused or result.tiles_processed == len(tiles),
            idle_sleep_seconds=_value_or_default(
                args.idle_sleep_seconds,
                settings.service_idle_sleep_seconds,
            ),
            once=args.once,
        )
    finally:
        download_queue.close()
        logger.info("discovery_cli_closed")


def main() -> None:
    asyncio.run(run(build_parser().parse_args()))


def _value_or_default[T](value: T | None, default: T) -> T:
    return default if value is None else value


if __name__ == "__main__":
    main()
