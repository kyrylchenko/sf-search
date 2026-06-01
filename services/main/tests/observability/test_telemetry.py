from contextlib import nullcontext

from main_service.observability.telemetry import observed_span, recorded_duration


class FakeTelemetry:
    def __init__(self) -> None:
        self.durations: list[tuple[str, float, dict[str, object]]] = []
        self.spans: list[tuple[str, dict[str, object] | None]] = []

    def record_duration(
        self,
        name: str,
        seconds: float,
        attributes: dict[str, object],
    ) -> None:
        self.durations.append((name, seconds, attributes))

    def span(self, name: str, attributes: dict[str, object] | None = None):
        self.spans.append((name, attributes))
        return nullcontext()


def test_recorded_duration_records_elapsed_seconds() -> None:
    current_time = [10.0]
    telemetry = FakeTelemetry()

    with recorded_duration(
        telemetry,  # type: ignore[arg-type]
        "downloader_batch",
        {"service": "downloader", "event": "downloader_batch"},
        clock=lambda: current_time[0],
    ):
        current_time[0] = 12.75

    assert telemetry.durations == [
        (
            "downloader_batch",
            2.75,
            {"service": "downloader", "event": "downloader_batch"},
        )
    ]


def test_observed_span_records_trace_span_and_duration() -> None:
    current_time = [1.0]
    telemetry = FakeTelemetry()

    with observed_span(
        telemetry,  # type: ignore[arg-type]
        "embedding.qdrant_upsert",
        "embedding_qdrant_upsert",
        {"service": "embedding", "stage": "qdrant_upsert", "batch_size": 5},
        clock=lambda: current_time[0],
    ):
        current_time[0] = 1.8

    assert telemetry.spans == [
        (
            "embedding.qdrant_upsert",
            {"service": "embedding", "stage": "qdrant_upsert", "batch_size": 5},
        )
    ]
    assert telemetry.durations == [
        (
            "embedding_qdrant_upsert",
            0.8,
            {"service": "embedding", "stage": "qdrant_upsert", "batch_size": 5},
        )
    ]
