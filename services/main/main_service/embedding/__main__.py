import argparse
import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path

from main_service.config import CONFIG, Settings
from main_service.db.initialize_engine import initialize_engine
from main_service.db.services.panorama_view_embedding_service import (
    EmbeddingModelSpec,
    PanoramaViewEmbeddingService,
)
from main_service.embedding.model import TransformersSiglipEmbedder
from main_service.embedding.nats_source import NatsPanoEmbeddingJobSource
from main_service.embedding.runner import run_embedding_batch
from main_service.embedding.vector_store_factory import create_vector_store
from main_service.logging_config import configure_cli_logging
from main_service.observability import TimedProgressReporter, configure_observability
from main_service.service_loop import run_service_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded pano embedding batch.")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum embedding jobs to pull in this run.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of embedding jobs to run concurrently.",
    )
    parser.add_argument(
        "--vector-store-dir",
        default=None,
        help="Directory where local HNSW vector indexes are stored.",
    )
    parser.add_argument(
        "--vector-store-kind",
        default=None,
        choices=["qdrant", "local_hnsw"],
        help="Vector store backend to use.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Qdrant HTTP URL.",
    )
    parser.add_argument(
        "--qdrant-collection",
        default=None,
        help="Qdrant collection name for panorama view embeddings.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Embedding model ID.",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Embedding model revision.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Embedding device: auto, cuda, mps, or cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of images to encode in one model forward pass.",
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
        help="Seconds to sleep after an empty batch.",
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
    telemetry = configure_observability(settings, "sf-search-embedding")
    progress = TimedProgressReporter(
        telemetry=telemetry,
        service_name="embedding",
    )
    logger = logging.getLogger(__name__)
    logger.info(
        (
            "embedding_cli_start limit=%s concurrency=%s batch_size=%s "
            "model_id=%s device=%s dtype=%s vector_store_kind=%s vector_store_path=%s"
        ),
        args.limit,
        args.concurrency or settings.pano_embedding_concurrency,
        args.batch_size or settings.embedding_batch_size,
        args.model_id or settings.embedding_model_id,
        args.device or settings.embedding_device,
        settings.embedding_dtype,
        args.vector_store_kind or settings.embedding_vector_store_kind,
        _vector_store_log_path(args, settings),
    )
    engine = initialize_engine(settings)
    embedding_service = PanoramaViewEmbeddingService(engine)
    model_spec = EmbeddingModelSpec(
        model_provider=settings.embedding_model_provider,
        model_id=args.model_id or settings.embedding_model_id,
        model_revision=args.model_revision or settings.embedding_model_revision,
        preprocess_version=settings.embedding_preprocess_version,
        embedding_dimension=settings.embedding_dimension,
        embedding_dtype=settings.embedding_dtype,
        embedding_normalized=True,
    )
    source = await NatsPanoEmbeddingJobSource.connect(
        servers=settings.nats_url,
        stream_name=settings.pano_embedding_stream,
        subject=settings.pano_embedding_subject,
        durable_consumer=settings.pano_embedding_consumer,
    )
    embedder = TransformersSiglipEmbedder(
        model_id=model_spec.model_id,
        revision=model_spec.model_revision,
        dtype=model_spec.embedding_dtype,
        device=_device_or_none(args.device or settings.embedding_device),
    )
    vector_store = create_vector_store(
        settings=settings,
        model_spec=model_spec,
        vector_store_kind=args.vector_store_kind,
        vector_store_dir=Path(args.vector_store_dir) if args.vector_store_dir else None,
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.qdrant_collection,
    )
    try:
        async def run_batch() -> object:
            with telemetry.span("embedding.batch"):
                result = await run_embedding_batch(
                    embedding_service=embedding_service,
                    job_source=source,
                    image_embedder=embedder,
                    vector_store=vector_store,
                    model_spec=model_spec,
                    limit=args.limit,
                    concurrency=args.concurrency or settings.pano_embedding_concurrency,
                    batch_size=args.batch_size or settings.embedding_batch_size,
                    progress=progress,
                )
            logger.info(
                "embedding_cli_batch_complete result=%s",
                json.dumps(asdict(result), sort_keys=True),
            )
            print(json.dumps(asdict(result), sort_keys=True), flush=True)
            return result

        await run_service_loop(
            service_name="embedding",
            run_batch=run_batch,
            should_idle=lambda result: _embedding_should_idle(result),
            idle_sleep_seconds=_value_or_default(
                args.idle_sleep_seconds,
                settings.service_idle_sleep_seconds,
            ),
            once=args.once,
        )
    finally:
        await source.close()
        telemetry.shutdown()
        logger.info("embedding_cli_closed")


def main() -> None:
    asyncio.run(run(build_parser().parse_args()))


def _value_or_default[T](value: T | None, default: T) -> T:
    return default if value is None else value


def _device_or_none(device: str | None) -> str | None:
    return None if device is None or device == "auto" else device


def _vector_store_log_path(args: argparse.Namespace, settings: Settings) -> str:
    kind = args.vector_store_kind or settings.embedding_vector_store_kind
    if kind == "qdrant":
        url = (args.qdrant_url or settings.qdrant_url).rstrip("/")
        collection = args.qdrant_collection or settings.qdrant_collection
        return f"{url}/{collection}"
    return args.vector_store_dir or settings.embedding_vector_store_dir


def _embedding_should_idle(result: object) -> bool:
    return (
        getattr(result, "embedded", 0) == 0
        and getattr(result, "skipped", 0) == 0
        and getattr(result, "failed", 0) == 0
    )


if __name__ == "__main__":
    main()
