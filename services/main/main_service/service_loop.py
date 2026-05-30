import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from main_service.logging_config import format_log_event

T = TypeVar("T")
logger = logging.getLogger(__name__)


async def run_service_loop(
    *,
    service_name: str,
    run_batch: Callable[[], Awaitable[T]],
    should_idle: Callable[[T], bool],
    idle_sleep_seconds: float,
    once: bool = False,
    max_batches: int | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    batches = 0
    while True:
        result = await run_batch()
        batches += 1

        if once:
            logger.info("%s_service_once_complete batches=%s", service_name, batches)
            return

        if max_batches is not None and batches >= max_batches:
            logger.info("%s_service_max_batches_reached batches=%s", service_name, batches)
            return

        if should_idle(result):
            logger.info(
                "%s",
                format_log_event(
                    "service_idle_sleep",
                    {
                        "service": service_name,
                        "seconds": idle_sleep_seconds,
                        "batches": batches,
                    },
                ),
            )
            await sleep(max(0.0, idle_sleep_seconds))
