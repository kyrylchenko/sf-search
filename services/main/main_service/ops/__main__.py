import argparse
import json
import logging

from main_service.config import CONFIG
from main_service.db.initialize_engine import initialize_engine
from main_service.db.services.panorama_view_embedding_service import EmbeddingModelSpec
from main_service.ingestion.download_queue import (
    NatsJetStreamPanoEmbeddingQueue,
    NatsJetStreamPanoProcessingQueue,
)
from main_service.logging_config import configure_cli_logging
from main_service.ops.requeue import (
    requeue_embedding_jobs_from_db,
    requeue_processing_jobs_from_db,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pipeline operational commands.")
    parser.add_argument("--log-level", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    processing = subparsers.add_parser(
        "requeue-processing",
        help="Enqueue downloaded panos that need preprocessing.",
    )
    processing.add_argument("--limit", type=int, default=100)
    processing.add_argument(
        "--include-already-processed",
        action="store_true",
        help="Also enqueue panos that already have completed view rows.",
    )

    embedding = subparsers.add_parser(
        "requeue-embedding",
        help="Enqueue completed views that need embeddings.",
    )
    embedding.add_argument("--limit", type=int, default=1000)
    embedding.add_argument(
        "--include-already-embedded",
        action="store_true",
        help="Also enqueue views that already have completed embeddings.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = CONFIG
    configure_cli_logging(args.log_level or settings.log_level)
    logging.getLogger(__name__).info("ops_cli_start command=%s", args.command)
    engine = initialize_engine(settings)
    if args.command == "requeue-processing":
        queue = NatsJetStreamPanoProcessingQueue.connect(
            servers=settings.nats_url,
            stream_name=settings.pano_processing_stream,
            subject=settings.pano_processing_subject,
            consumer_name=settings.pano_processing_consumer,
        )
        try:
            count = requeue_processing_jobs_from_db(
                engine=engine,
                processing_queue=queue,
                limit=args.limit,
                include_already_processed=args.include_already_processed,
            )
        finally:
            queue.close()
    elif args.command == "requeue-embedding":
        queue = NatsJetStreamPanoEmbeddingQueue.connect(
            servers=settings.nats_url,
            stream_name=settings.pano_embedding_stream,
            subject=settings.pano_embedding_subject,
            consumer_name=settings.pano_embedding_consumer,
        )
        model_spec = EmbeddingModelSpec(
            model_provider=settings.embedding_model_provider,
            model_id=settings.embedding_model_id,
            model_revision=settings.embedding_model_revision,
            preprocess_version=settings.embedding_preprocess_version,
            embedding_dimension=settings.embedding_dimension,
            embedding_dtype=settings.embedding_dtype,
            embedding_normalized=True,
        )
        try:
            count = requeue_embedding_jobs_from_db(
                engine=engine,
                embedding_queue=queue,
                model_spec=model_spec,
                limit=args.limit,
                include_already_embedded=args.include_already_embedded,
            )
        finally:
            queue.close()
    else:
        raise ValueError(f"Unsupported ops command: {args.command}")
    print(json.dumps({"command": args.command, "requeued": count}, sort_keys=True))


if __name__ == "__main__":
    main()
