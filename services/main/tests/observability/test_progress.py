from main_service.observability.progress import TimedProgressReporter


class FakeTelemetry:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []
        self.durations: list[tuple[str, float, dict[str, object]]] = []

    def record_event(self, event: str, payload: dict[str, object]) -> None:
        self.events.append((event, payload))

    def record_duration(
        self,
        name: str,
        seconds: float,
        attributes: dict[str, object],
    ) -> None:
        self.durations.append((name, seconds, attributes))


def test_timed_progress_reporter_records_events_and_matching_duration() -> None:
    current_time = [10.0]
    telemetry = FakeTelemetry()
    reporter = TimedProgressReporter(
        telemetry=telemetry,
        service_name="processing",
        clock=lambda: current_time[0],
    )

    reporter("processing_job_start", {"pano_id": "pano-a"})
    current_time[0] = 12.5
    reporter("processing_job_complete", {"pano_id": "pano-a"})

    assert telemetry.events == [
        ("processing_job_start", {"pano_id": "pano-a"}),
        ("processing_job_complete", {"pano_id": "pano-a"}),
    ]
    assert telemetry.durations == [
        (
            "processing_job",
            2.5,
            {"service": "processing", "event": "processing_job"},
        )
    ]


def test_timed_progress_reporter_ignores_complete_without_matching_start() -> None:
    telemetry = FakeTelemetry()
    reporter = TimedProgressReporter(
        telemetry=telemetry,
        service_name="embedding",
        clock=lambda: 20.0,
    )

    reporter("embedding_job_complete", {"pano_id": "pano-a", "view_id": 1})

    assert telemetry.events == [
        ("embedding_job_complete", {"pano_id": "pano-a", "view_id": 1})
    ]
    assert telemetry.durations == []


def test_timed_progress_reporter_uses_low_cardinality_attributes() -> None:
    current_time = [1.0]
    telemetry = FakeTelemetry()
    reporter = TimedProgressReporter(
        telemetry=telemetry,
        service_name="embedding",
        clock=lambda: current_time[0],
    )

    reporter("embedding_image_batch_start", {"model_id": "model-a", "batch_size": 4})
    current_time[0] = 3.0
    reporter("embedding_image_batch_complete", {"model_id": "model-a", "batch_size": 4})

    assert telemetry.durations == [
        (
            "embedding_image_batch",
            2.0,
            {"service": "embedding", "event": "embedding_image_batch"},
        )
    ]


def test_timed_progress_reporter_ignores_storage_ids_for_matching_duration() -> None:
    current_time = [5.0]
    telemetry = FakeTelemetry()
    reporter = TimedProgressReporter(
        telemetry=telemetry,
        service_name="embedding",
        clock=lambda: current_time[0],
    )

    reporter("embedding_job_start", {"pano_id": "pano-a", "view_id": 12})
    current_time[0] = 6.25
    reporter(
        "embedding_job_complete",
        {
            "pano_id": "pano-a",
            "view_id": 12,
            "embedding_id": 99,
            "vector_id": "99",
        },
    )

    assert telemetry.durations == [
        (
            "embedding_job",
            1.25,
            {"service": "embedding", "event": "embedding_job"},
        )
    ]
