from main_service.embedding.__main__ import build_parser


def test_embedding_entrypoint_parser_accepts_safe_manual_options() -> None:
    args = build_parser().parse_args(
        [
            "--limit",
            "5",
            "--concurrency",
            "1",
            "--vector-store-dir",
            ".local/embedding-indexes",
            "--model-id",
            "google/siglip2-so400m-patch14-384",
        ]
    )

    assert args.limit == 5
    assert args.concurrency == 1
    assert args.vector_store_dir == ".local/embedding-indexes"
    assert args.model_id == "google/siglip2-so400m-patch14-384"
