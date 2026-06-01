from main_service.observability.telemetry import recorded_duration


class FakeTelemetry:
    def __init__(self) -> None:
        self.durations: list[tuple[str, float, dict[str, object]]] = []

    def record_duration(
        self,
        name: str,
        seconds: float,
        attributes: dict[str, object],
    ) -> None:
        self.durations.append((name, seconds, attributes))


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
