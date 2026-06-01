from main_service.config import Settings


def test_embedding_defaults_are_configured_for_local_dev() -> None:
    settings = Settings(
        _env_file=None,
        db_user="user",
        db_password="password",
        db_host="localhost",
        db_port=5432,
        db_name="sf_search",
    )

    assert settings.pano_embedding_stream == "PANO_EMBEDDING"
    assert settings.pano_embedding_subject == "pano.embedding.requested"
    assert settings.pano_embedding_consumer == "pano-embedder"
    assert settings.max_embedding_queue_depth == 100
    assert settings.embedding_model_provider == "transformers"
    assert settings.embedding_model_id == "google/siglip2-so400m-patch14-384"
    assert settings.embedding_preprocess_version == "siglip2-384-rgb-v1"
    assert settings.embedding_vector_store_dir == ".local/embedding-indexes"
    assert settings.embedding_vector_store_kind == "qdrant"
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "panorama_view_embeddings_siglip2"
    assert settings.qdrant_vector_on_disk is True
    assert settings.qdrant_hnsw_on_disk is False
    assert settings.qdrant_on_disk_payload is True
    assert settings.qdrant_upsert_wait is True
    assert settings.qdrant_timeout_seconds == 30.0
    assert settings.embedding_device == "auto"
    assert settings.embedding_batch_size == 1
