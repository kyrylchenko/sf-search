import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from main_service.config import CONFIG
from main_service.db.initialize_engine import initialize_engine
from main_service.db.services.panorama_view_service import PanoramaViewService
from main_service.ingestion.download_queue import NatsJetStreamPanoEmbeddingQueue
from main_service.processing.nats_source import NatsPanoProcessingJobSource
from main_service.processing.runner import run_processing_batch


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
    return parser


async def run(args: argparse.Namespace) -> None:
    settings = CONFIG
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
    )
    try:
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
            max_embedding_queue_depth=(
                args.max_embedding_queue_depth or settings.max_embedding_queue_depth
            ),
        )
        print(json.dumps(asdict(result), sort_keys=True))
    finally:
        await source.close()
        embedding_queue.close()


def main() -> None:
    asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
