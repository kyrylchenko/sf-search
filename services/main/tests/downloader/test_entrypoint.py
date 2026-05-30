from main_service.downloader.__main__ import build_parser


def test_downloader_entrypoint_parser_accepts_safe_manual_options() -> None:
    args = build_parser().parse_args(
        [
            "--limit",
            "5",
            "--concurrency",
            "2",
            "--max-processing-queue-depth",
            "10",
            "--idle-sleep-seconds",
            "0.25",
            "--once",
        ]
    )

    assert args.limit == 5
    assert args.concurrency == 2
    assert args.max_processing_queue_depth == 10
    assert args.idle_sleep_seconds == 0.25
    assert args.once is True
