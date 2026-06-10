from main_service.discovery.__main__ import build_parser


def test_discovery_entrypoint_parser_accepts_service_options() -> None:
    args = build_parser().parse_args(
        [
            "--geojson",
            "city.geojson",
            "--zoom",
            "17",
            "--max-downloader-queue-depth",
            "250",
            "--tile-order",
            "sequential",
            "--tile-random-seed",
            "123",
            "--idle-sleep-seconds",
            "0.25",
            "--once",
        ]
    )

    assert args.geojson == "city.geojson"
    assert args.zoom == 17
    assert args.max_downloader_queue_depth == 250
    assert args.tile_order == "sequential"
    assert args.tile_random_seed == 123
    assert args.idle_sleep_seconds == 0.25
    assert args.once is True
