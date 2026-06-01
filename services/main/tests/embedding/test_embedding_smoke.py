from main_service.embedding.smoke import build_parser


def test_embedding_smoke_parser_accepts_runtime_options() -> None:
    args = build_parser().parse_args(
        [
            "--tiles-dir",
            ".local/panorama-views",
            "--limit",
            "3",
            "--device",
            "cuda",
            "--dtype",
            "float16",
            "--batch-size",
            "8",
        ]
    )

    assert args.tiles_dir == ".local/panorama-views"
    assert args.limit == 3
    assert args.device == "cuda"
    assert args.dtype == "float16"
    assert args.batch_size == 8
