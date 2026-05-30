import json
import logging
import sys


EVENT_MARKERS = {
    "discovery_complete": "📦",
    "discovery_coverage_fetch_complete": "🔎",
    "discovery_coverage_fetch_start": "🔎",
    "discovery_download_enqueued": "📤",
    "discovery_paused": "⏸️",
    "discovery_tile_complete": "✅",
    "discovery_tile_replay_start": "🔁",
    "discovery_tile_start": "🧱",
    "downloader_batch_complete": "📦",
    "downloader_fetch_complete": "📥",
    "downloader_fetch_start": "📥",
    "downloader_job_start": "⬇️",
    "downloader_job_complete": "✅",
    "downloader_job_failed": "❌",
    "downloader_job_skipped": "↩️",
    "downloader_paused": "⏸️",
    "embedding_batch_complete": "📦",
    "embedding_fetch_complete": "📥",
    "embedding_fetch_start": "📥",
    "embedding_image_start": "🧠",
    "embedding_job_start": "🧠",
    "embedding_job_complete": "✅",
    "embedding_job_failed": "❌",
    "embedding_job_skipped": "↩️",
    "embedding_vector_store_start": "📍",
    "processing_batch_complete": "📦",
    "processing_job_complete": "✅",
    "processing_job_failed": "❌",
    "processing_job_start": "🧭",
    "processing_paused": "⏸️",
    "processing_view_complete": "✅",
    "processing_view_concurrency_capped": "🧯",
    "processing_view_failed": "❌",
    "processing_view_render_pool_complete": "🏁",
    "processing_view_render_pool_start": "🧵",
    "processing_view_skipped": "↩️",
    "processing_view_start": "🧩",
    "vector_store_add_complete": "📍",
    "service_idle_sleep": "💤",
}


def format_log_event(event: str, payload: dict[str, object]) -> str:
    marker = EVENT_MARKERS.get(event, "•")
    return f"{marker} {event} {json.dumps(payload, sort_keys=True)}"


def configure_cli_logging(level_name: str = "INFO") -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
        force=True,
    )
