import threading
from collections.abc import Callable
from time import monotonic

from main_service.observability.telemetry import PipelineTelemetry


class TimedProgressReporter:
    def __init__(
        self,
        *,
        telemetry: PipelineTelemetry,
        service_name: str,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.telemetry = telemetry
        self.service_name = service_name
        self.clock = clock
        self._starts: dict[tuple[str, tuple[tuple[str, object], ...]], float] = {}
        self._lock = threading.Lock()

    def __call__(self, event: str, payload: dict[str, object]) -> None:
        self.telemetry.record_event(event, payload)
        base_name = _base_event_name(event)
        if base_name is None:
            return

        key = (base_name, _identity_items(payload))
        now = self.clock()
        if event.endswith("_start"):
            with self._lock:
                self._starts[key] = now
            return

        with self._lock:
            started_at = self._starts.pop(key, None)
        if started_at is None:
            return
        self.telemetry.record_duration(
            base_name,
            max(0.0, now - started_at),
            {
                "service": self.service_name,
                "event": base_name,
            },
        )


def _base_event_name(event: str) -> str | None:
    for suffix in ("_start", "_complete", "_failed", "_skipped"):
        if event.endswith(suffix):
            return event[: -len(suffix)]
    return None


def _identity_items(payload: dict[str, object]) -> tuple[tuple[str, object], ...]:
    items = []
    for key in (
        "pano_id",
        "view_id",
        "viewset_name",
        "model_id",
        "batch_size",
    ):
        value = payload.get(key)
        if isinstance(value, str | int | float | bool):
            items.append((key, value))
    return tuple(items)
