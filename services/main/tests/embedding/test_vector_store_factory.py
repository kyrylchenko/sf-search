from pathlib import Path

from main_service.config import Settings
from main_service.db.services.panorama_view_embedding_service import EmbeddingModelSpec
from main_service.embedding import vector_store_factory


class FakeQdrantStore:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class FakeLocalStore:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def make_settings(**overrides: object) -> Settings:
    values = {
        "_env_file": None,
        "db_user": "user",
        "db_password": "password",
        "db_host": "localhost",
        "db_port": 5432,
        "db_name": "sf_search",
    }
    values.update(overrides)
    return Settings(**values)


def make_model_spec() -> EmbeddingModelSpec:
    return EmbeddingModelSpec(
        model_provider="transformers",
        model_id="google/siglip2-so400m-patch14-384",
        model_revision="main",
        preprocess_version="siglip2-384-rgb-v1",
        embedding_dimension=1152,
        embedding_dtype="float16",
        embedding_normalized=True,
    )


def test_factory_creates_qdrant_store_by_default(monkeypatch) -> None:
    monkeypatch.setattr(vector_store_factory, "QdrantVectorStore", FakeQdrantStore)

    store = vector_store_factory.create_vector_store(
        settings=make_settings(),
        model_spec=make_model_spec(),
    )

    assert isinstance(store, FakeQdrantStore)
    assert store.kwargs == {
        "url": "http://localhost:6333",
        "collection_name": "panorama_view_embeddings_siglip2",
        "dimension": 1152,
        "vector_on_disk": True,
        "hnsw_on_disk": False,
        "on_disk_payload": True,
        "upsert_wait": True,
        "timeout_seconds": 30.0,
        "telemetry": None,
    }


def test_factory_creates_local_hnsw_store_when_configured(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(vector_store_factory, "LocalHnswVectorStore", FakeLocalStore)

    store = vector_store_factory.create_vector_store(
        settings=make_settings(embedding_vector_store_kind="local_hnsw"),
        model_spec=make_model_spec(),
        vector_store_dir=tmp_path,
    )

    assert isinstance(store, FakeLocalStore)
    assert store.kwargs == {
        "root_dir": tmp_path,
        "model_id": "google/siglip2-so400m-patch14-384",
        "dimension": 1152,
    }


def test_factory_rejects_unknown_store_kind() -> None:
    try:
        vector_store_factory.create_vector_store(
            settings=make_settings(embedding_vector_store_kind="unknown"),
            model_spec=make_model_spec(),
        )
    except ValueError as exc:
        assert str(exc) == "Unsupported vector store kind: unknown"
    else:
        raise AssertionError("expected unsupported vector store kind")
