import argparse
import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path

from main_service.config import CONFIG
from main_service.db.initialize_engine import initialize_engine
from main_service.db.services.panorama_view_service import PanoramaViewService
from main_service.ingestion.download_queue import NatsJetStreamPanoEmbeddingQueue
from main_service.logging_config import configure_cli_logging
from main_service.processing.nats_source import NatsPanoProcessingJobSource
from main_service.processing.runner import run_processing_batch
from main_service.service_loop import run_service_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded pano preprocessing batch.")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum processing jobs to pull in this run.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of pano processing jobs to run concurrently.",
    )
    parser.add_argument(
        "--render-scale",
        type=int,
        default=None,
        help="Multiplier applied to each viewset output dimension.",
    )
    parser.add_argument(
        "--viewsets-dir",
        default=None,
        help="Directory containing preprocessing viewset JSON files.",
    )
    parser.add_argument(
        "--storage-dir",
        default=None,
        help="Directory where generated panorama view images are stored.",
    )
    parser.add_argument(
        "--output-format",
        default=None,
        choices=["jpeg", "jpg", "png"],
        help="Generated view image format.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        help="JPEG quality for generated view images.",
    )
    parser.add_argument(
        "--max-embedding-queue-depth",
        type=int,
        default=None,
        help="Pause before fetching when embedding queue depth is at this value.",
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
    logger = logging.getLogger(__name__)
    logger.info(
        "processing_cli_start limit=%s concurrency=%s render_scale=%s",
        args.limit,
        args.concurrency or settings.pano_processing_concurrency,
        args.render_scale or settings.pano_view_render_scale,
    )
    engine = initialize_engine(settings)
    view_service = PanoramaViewService(engine)
    source = await NatsPanoProcessingJobSource.connect(
        servers=settings.nats_url,
        stream_name=settings.pano_processing_stream,
        subject=settings.pano_processing_subject,
        durable_consumer=settings.pano_processing_consumer,
    )
    embedding_queue = NatsJetStreamPanoEmbeddingQueue.connect(
        servers=settings.nats_url,
        stream_name=settings.pano_embedding_stream,
        subject=settings.pano_embedding_subject,
        consumer_name=settings.pano_embedding_consumer,
    )
    try:
        async def run_batch() -> object:
            result = await run_processing_batch(
                panorama_view_service=view_service,
                job_source=source,
                viewsets_dir=Path(args.viewsets_dir or settings.pano_viewsets_dir),
                storage_dir=Path(args.storage_dir or settings.pano_view_storage_dir),
                limit=args.limit,
                concurrency=args.concurrency or settings.pano_processing_concurrency,
                render_scale=args.render_scale or settings.pano_view_render_scale,
                output_format=args.output_format or settings.pano_view_output_format,
                image_quality=args.jpeg_quality or settings.pano_view_jpeg_quality,
                embedding_queue=embedding_queue,
                max_embedding_queue_depth=_value_or_default(
                    args.max_embedding_queue_depth,
                    settings.max_embedding_queue_depth,
                ),
            )
            logger.info(
                "processing_cli_batch_complete result=%s",
                json.dumps(asdict(result), sort_keys=True),
            )
            print(json.dumps(asdict(result), sort_keys=True), flush=True)
            return result

        await run_service_loop(
            service_name="processing",
            run_batch=run_batch,
            should_idle=lambda result: _processing_should_idle(result),
            idle_sleep_seconds=_value_or_default(
                args.idle_sleep_seconds,
                settings.service_idle_sleep_seconds,
            ),
            once=args.once,
        )
    finally:
        await source.close()
        embedding_queue.close()
        logger.info("processing_cli_closed")


def main() -> None:
    asyncio.run(run(build_parser().parse_args()))


def _value_or_default[T](value: T | None, default: T) -> T:
    return default if value is None else value


def _processing_should_idle(result: object) -> bool:
    return bool(getattr(result, "pause_reason", None)) or (
        getattr(result, "processed_jobs", 0) == 0
        and getattr(result, "failed_jobs", 0) == 0
    )


if __name__ == "__main__":
    main()
