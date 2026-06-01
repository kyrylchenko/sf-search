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
            "--vector-store-kind",
            "qdrant",
            "--qdrant-url",
            "http://localhost:6333",
            "--qdrant-collection",
            "panorama_view_embeddings_siglip2",
            "--model-id",
            "google/siglip2-so400m-patch14-384",
            "--device",
            "cuda",
            "--batch-size",
            "16",
            "--idle-sleep-seconds",
            "0.25",
            "--once",
        ]
    )

    assert args.limit == 5
    assert args.concurrency == 1
    assert args.vector_store_dir == ".local/embedding-indexes"
    assert args.vector_store_kind == "qdrant"
    assert args.qdrant_url == "http://localhost:6333"
    assert args.qdrant_collection == "panorama_view_embeddings_siglip2"
    assert args.model_id == "google/siglip2-so400m-patch14-384"
    assert args.device == "cuda"
    assert args.batch_size == 16
    assert args.idle_sleep_seconds == 0.25
    assert args.once is True
