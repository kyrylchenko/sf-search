import argparse
import asyncio
import json
import logging

from main_service.config import CONFIG
from main_service.db.initialize_engine import initialize_engine
from main_service.ingestion.download_queue import (
    NatsJetStreamPanoDownloadQueue,
    NatsJetStreamPanoEmbeddingQueue,
    NatsJetStreamPanoProcessingQueue,
)
from main_service.logging_config import configure_cli_logging, format_log_event
from main_service.monitoring.snapshot import QueueSnapshotSource, build_pipeline_snapshot
from main_service.observability import TimedProgressReporter, configure_observability
from main_service.observability.telemetry import PipelineTelemetry
from main_service.service_loop import run_service_loop

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pipeline monitoring snapshots.")
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=None,
        help="Seconds between monitoring snapshots.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Console log level: DEBUG, INFO, WARNING, ERROR.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one snapshot and exit.",
    )
    return parser


async def run(args: argparse.Namespace) -> None:
    settings = CONFIG
    configure_cli_logging(args.log_level or settings.log_level)
    telemetry = configure_observability(settings, "sf-search-monitoring")
    progress = TimedProgressReporter(
        telemetry=telemetry,
        service_name="monitoring",
    )
    engine = initialize_engine(settings)
    queues = QueueSnapshotSource(
        download=NatsJetStreamPanoDownloadQueue.connect(
            servers=settings.nats_url,
            stream_name=settings.pano_download_stream,
            subject=settings.pano_download_subject,
            consumer_name=settings.pano_downloader_consumer,
        ),
        processing=NatsJetStreamPanoProcessingQueue.connect(
            servers=settings.nats_url,
            stream_name=settings.pano_processing_stream,
            subject=settings.pano_processing_subject,
            consumer_name=settings.pano_processing_consumer,
        ),
        embedding=NatsJetStreamPanoEmbeddingQueue.connect(
            servers=settings.nats_url,
            stream_name=settings.pano_embedding_stream,
            subject=settings.pano_embedding_subject,
            consumer_name=settings.pano_embedding_consumer,
        ),
    )
    try:
        async def run_batch() -> object:
            with telemetry.span("monitoring.snapshot"):
                progress("monitoring_snapshot_start", {})
                snapshot = build_pipeline_snapshot(engine=engine, queues=queues)
                _emit_snapshot_metrics(telemetry, snapshot.to_dict())
                progress("monitoring_snapshot_complete", {})
                logger.info(
                    "%s",
                    format_log_event("monitoring_snapshot_complete", snapshot.to_dict()),
                )
                print(json.dumps(snapshot.to_dict(), sort_keys=True), flush=True)
                return snapshot

        await run_service_loop(
            service_name="monitoring",
            run_batch=run_batch,
            should_idle=lambda _result: True,
            idle_sleep_seconds=_value_or_default(
                args.interval_seconds,
                settings.monitoring_interval_seconds,
            ),
            once=args.once,
        )
    finally:
        queues.download.close()
        queues.processing.close()
        queues.embedding.close()
        telemetry.shutdown()


def main() -> None:
    asyncio.run(run(build_parser().parse_args()))


def _emit_snapshot_metrics(
    telemetry: PipelineTelemetry,
    snapshot: dict[str, object],
) -> None:
    status_counts = snapshot.get("status_counts", {})
    if isinstance(status_counts, dict):
        for table, counts in status_counts.items():
            if not isinstance(counts, dict):
                continue
            for status, count in counts.items():
                if isinstance(count, int | float):
                    telemetry.set_gauge(
                        "sf_search_status_rows",
                        count,
                        {"table": str(table), "status": str(status)},
                    )

    queue_depths = snapshot.get("queue_depths", {})
    if isinstance(queue_depths, dict):
        for queue, depth in queue_depths.items():
            if isinstance(depth, int | float):
                telemetry.set_gauge(
                    "sf_search_nats_queue_pending",
                    depth,
                    {"queue": str(queue)},
                )

    coverage = snapshot.get("coverage", {})
    if isinstance(coverage, dict):
        for key, value in coverage.items():
            if isinstance(value, int | float):
                telemetry.set_gauge(
                    "sf_search_coverage",
                    value,
                    {"field": str(key)},
                )


def _value_or_default[T](value: T | None, default: T) -> T:
    return default if value is None else value


if __name__ == "__main__":
    main()
