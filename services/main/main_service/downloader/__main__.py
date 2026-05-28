import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from main_service.config import CONFIG
from main_service.db.initialize_engine import initialize_engine
from main_service.db.services.panorama_service import PanoramaService
from main_service.downloader.nats_source import NatsPanoDownloadJobSource
from main_service.downloader.runner import run_downloader_batch
from main_service.ingestion.download_queue import NatsJetStreamPanoProcessingQueue


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
    return parser


async def run(args: argparse.Namespace) -> None:
    settings = CONFIG
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
    )

    try:
        result = await run_downloader_batch(
            panorama_service=panorama_service,
            job_source=source,
            processing_queue=processing_queue,
            storage_dir=Path(args.storage_dir or settings.pano_download_storage_dir),
            limit=args.limit,
            concurrency=args.concurrency or settings.pano_download_concurrency,
            max_processing_queue_depth=(
                args.max_processing_queue_depth or settings.max_processing_queue_depth
            ),
            max_attempts=settings.max_attempts,
        )
        print(json.dumps(asdict(result), sort_keys=True))
    finally:
        await source.close()
        processing_queue.close()


def main() -> None:
    asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
