from main_service.monitoring.__main__ import _emit_snapshot_metrics, build_parser


def test_monitoring_parser_accepts_runtime_options() -> None:
    args = build_parser().parse_args(
        [
            "--interval-seconds",
            "2.5",
            "--log-level",
            "DEBUG",
            "--once",
        ]
    )

    assert args.interval_seconds == 2.5
    assert args.log_level == "DEBUG"
    assert args.once is True


class FakeTelemetry:
    def __init__(self) -> None:
        self.gauges: list[tuple[str, int | float, dict[str, str]]] = []

    def set_gauge(
        self,
        name: str,
        value: int | float,
        attributes: dict[str, str],
    ) -> None:
        self.gauges.append((name, value, attributes))


def test_emit_snapshot_metrics_includes_numeric_qdrant_fields() -> None:
    telemetry = FakeTelemetry()

    _emit_snapshot_metrics(
        telemetry,  # type: ignore[arg-type]
        {
            "qdrant": {
                "collection": "panorama_view_embeddings_siglip2",
                "status": "green",
                "points_count": 42,
                "vectors_count": 42,
                "indexed_vectors_count": 40,
            }
        },
    )

    assert (
        "sf_search_qdrant_collection",
        42,
        {
            "collection": "panorama_view_embeddings_siglip2",
            "field": "points_count",
        },
    ) in telemetry.gauges
    assert (
        "sf_search_qdrant_collection",
        40,
        {
            "collection": "panorama_view_embeddings_siglip2",
            "field": "indexed_vectors_count",
        },
    ) in telemetry.gauges
