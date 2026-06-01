from main_service.monitoring.__main__ import build_parser


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
