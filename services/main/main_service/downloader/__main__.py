import argparse
import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path

from main_service.config import CONFIG
from main_service.db.initialize_engine import initialize_engine
from main_service.db.services.panorama_service import PanoramaService
from main_service.downloader.nats_source import NatsPanoDownloadJobSource
from main_service.downloader.requeue import requeue_download_jobs_from_db
from main_service.downloader.runner import run_downloader_batch
from main_service.ingestion.download_queue import (
    NatsJetStreamPanoDownloadQueue,
    NatsJetStreamPanoProcessingQueue,
)
from main_service.logging_config import configure_cli_logging
from main_service.observability import configure_observability
from main_service.service_loop import run_service_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded pano download batch.")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum downloader jobs to pull in this run.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of pano downloads to run concurrently.",
    )
    parser.add_argument(
        "--max-processing-queue-depth",
        type=int,
        default=None,
        help="Pause before fetching when processing queue depth is at this value.",
    )
    parser.add_argument(
        "--storage-dir",
        default=None,
        help="Directory where downloaded panorama images are stored.",
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
        help="Seconds to sleep after an empty or paused batch.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one bounded batch and exit. Default is to keep polling forever.",
    )
    return parser


async def run(args: argparse.Namespace) -> None:
    settings = CONFIG
    configure_cli_logging(args.log_level or settings.log_level)
    telemetry = configure_observability(settings, "sf-search-downloader")
    logger = logging.getLogger(__name__)
    logger.info(
        "downloader_cli_start limit=%s concurrency=%s storage_dir=%s",
        args.limit,
        args.concurrency or settings.pano_download_concurrency,
        args.storage_dir or settings.pano_download_storage_dir,
    )
    engine = initialize_engine(settings)
    panorama_service = PanoramaService(engine)
    source = await NatsPanoDownloadJobSource.connect(
        servers=settings.nats_url,
        stream_name=settings.pano_download_stream,
        subject=settings.pano_download_subject,
        durable_consumer=settings.pano_downloader_consumer,
    )
    processing_queue = NatsJetStreamPanoProcessingQueue.connect(
        servers=settings.nats_url,
        stream_name=settings.pano_processing_stream,
        subject=settings.pano_processing_subject,
        consumer_name=settings.pano_processing_consumer,
    )
    download_queue = NatsJetStreamPanoDownloadQueue.connect(
        servers=settings.nats_url,
        stream_name=settings.pano_download_stream,
        subject=settings.pano_download_subject,
        consumer_name=settings.pano_downloader_consumer,
    )

    try:
        async def run_batch() -> object:
            with telemetry.span("downloader.batch"):
                if download_queue.pending_count() == 0:
                    requeue_download_jobs_from_db(
                        panorama_service=panorama_service,
                        download_queue=download_queue,
                        limit=args.limit,
                    )
                result = await run_downloader_batch(
                    panorama_service=panorama_service,
                    job_source=source,
                    processing_queue=processing_queue,
                    storage_dir=Path(args.storage_dir or settings.pano_download_storage_dir),
                    limit=args.limit,
                    concurrency=args.concurrency or settings.pano_download_concurrency,
                    max_processing_queue_depth=_value_or_default(
                        args.max_processing_queue_depth,
                        settings.max_processing_queue_depth,
                    ),
                    max_attempts=settings.max_attempts,
                )
            telemetry.record_event("downloader_batch_complete", asdict(result))
            logger.info(
                "downloader_cli_batch_complete result=%s",
                json.dumps(asdict(result), sort_keys=True),
            )
            print(json.dumps(asdict(result), sort_keys=True), flush=True)
            return result

        await run_service_loop(
            service_name="downloader",
            run_batch=run_batch,
            should_idle=lambda result: _downloader_should_idle(result),
            idle_sleep_seconds=_value_or_default(
                args.idle_sleep_seconds,
                settings.service_idle_sleep_seconds,
            ),
            once=args.once,
        )
    finally:
        await source.close()
        download_queue.close()
        processing_queue.close()
        telemetry.shutdown()
        logger.info("downloader_cli_closed")


def main() -> None:
    asyncio.run(run(build_parser().parse_args()))


def _value_or_default[T](value: T | None, default: T) -> T:
    return default if value is None else value


def _downloader_should_idle(result: object) -> bool:
    return bool(getattr(result, "pause_reason", None)) or (
        getattr(result, "downloaded", 0) == 0
        and getattr(result, "skipped", 0) == 0
        and getattr(result, "failed", 0) == 0
    )


if __name__ == "__main__":
    main()
