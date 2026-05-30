from main_service.processing.__main__ import build_parser


def test_processing_entrypoint_parser_accepts_safe_manual_options() -> None:
    args = build_parser().parse_args(
        [
            "--limit",
            "5",
            "--concurrency",
            "8",
            "--max-view-concurrency",
            "4",
            "--render-scale",
            "2",
            "--viewsets-dir",
            "../../docs/data/viewsets",
            "--storage-dir",
            ".local/panorama-views",
            "--idle-sleep-seconds",
            "0.25",
            "--once",
        ]
    )

    assert args.limit == 5
    assert args.concurrency == 8
    assert args.max_view_concurrency == 4
    assert args.render_scale == 2
    assert args.viewsets_dir == "../../docs/data/viewsets"
    assert args.storage_dir == ".local/panorama-views"
    assert args.idle_sleep_seconds == 0.25
    assert args.once is True
