from types import SimpleNamespace

import numpy as np
import pytest

from main_service.embedding.qdrant_store import QdrantVectorStore
from main_service.embedding.vector_store import VectorStoreRecord


class FakeDistance:
    COSINE = "Cosine"


class FakeVectorParams:
    def __init__(
        self,
        *,
        size: int,
        distance: str,
        on_disk: bool,
    ) -> None:
        self.size = size
        self.distance = distance
        self.on_disk = on_disk


class FakeHnswConfigDiff:
    def __init__(self, *, on_disk: bool) -> None:
        self.on_disk = on_disk


class FakePointStruct:
    def __init__(
        self,
        *,
        id: int,
        vector: list[float],
        payload: dict[str, object],
    ) -> None:
        self.id = id
        self.vector = vector
        self.payload = payload


class FakeModels:
    Distance = FakeDistance
    VectorParams = FakeVectorParams
    HnswConfigDiff = FakeHnswConfigDiff
    PointStruct = FakePointStruct


class FakeQdrantClient:
    def __init__(self, *, collection_exists: bool = False) -> None:
        self.collection_exists_result = collection_exists
        self.collection_exists_calls: list[str] = []
        self.create_collection_calls: list[dict[str, object]] = []
        self.upsert_calls: list[dict[str, object]] = []
        self.query_points_calls: list[dict[str, object]] = []
        self.raise_on_upsert: Exception | None = None

    def collection_exists(self, collection_name: str) -> bool:
        self.collection_exists_calls.append(collection_name)
        return self.collection_exists_result

    def create_collection(self, **kwargs: object) -> bool:
        self.create_collection_calls.append(kwargs)
        self.collection_exists_result = True
        return True

    def upsert(self, **kwargs: object) -> object:
        if self.raise_on_upsert is not None:
            raise self.raise_on_upsert
        self.upsert_calls.append(kwargs)
        return SimpleNamespace(status="completed")

    def query_points(self, **kwargs: object) -> object:
        self.query_points_calls.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(id=10, score=0.75),
                SimpleNamespace(id=11, score=0.5),
            ]
        )


class FakeTelemetry:
    def __init__(self) -> None:
        self.durations: list[tuple[str, float, dict[str, object]]] = []
        self.spans: list[tuple[str, dict[str, object] | None]] = []

    def record_duration(
        self,
        name: str,
        seconds: float,
        attributes: dict[str, object],
    ) -> None:
        self.durations.append((name, seconds, attributes))

    def span(self, name: str, attributes: dict[str, object] | None = None):
        from contextlib import nullcontext

        self.spans.append((name, attributes))
        return nullcontext()


def make_store(client: FakeQdrantClient) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        models=FakeModels,
        url="http://qdrant:6333",
        collection_name="panorama_view_embeddings_siglip2",
        dimension=3,
        vector_on_disk=True,
        hnsw_on_disk=False,
        on_disk_payload=True,
        upsert_wait=True,
    )


def test_add_many_records_qdrant_upsert_span_and_duration() -> None:
    client = FakeQdrantClient(collection_exists=True)
    telemetry = FakeTelemetry()
    store = QdrantVectorStore(
        client=client,
        models=FakeModels,
        url="http://qdrant:6333",
        collection_name="panorama_view_embeddings_siglip2",
        dimension=3,
        vector_on_disk=True,
        hnsw_on_disk=False,
        on_disk_payload=True,
        upsert_wait=True,
        telemetry=telemetry,  # type: ignore[arg-type]
    )

    store.add_many(
        [
            VectorStoreRecord(
                vector_id=42,
                vector=np.array([3.0, 4.0, 0.0]),
                metadata={"embedding_id": 42},
            )
        ]
    )

    assert telemetry.spans == [
        (
            "embedding.qdrant_upsert",
            {
                "service": "embedding",
                "collection": "panorama_view_embeddings_siglip2",
                "records": 1,
            },
        )
    ]
    assert telemetry.durations[0][0] == "embedding_qdrant_upsert"
    assert telemetry.durations[0][2] == {
        "service": "embedding",
        "stage": "qdrant_upsert",
        "collection": "panorama_view_embeddings_siglip2",
        "batch_size": 1,
    }


def test_add_many_creates_collection_and_upserts_normalized_points() -> None:
    client = FakeQdrantClient(collection_exists=False)
    store = make_store(client)

    vector_ids = store.add_many(
        [
            VectorStoreRecord(
                vector_id=42,
                vector=np.array([3.0, 4.0, 0.0]),
                metadata={
                    "embedding_id": 42,
                    "view_id": 7,
                    "model_id": "google/siglip2-so400m-patch14-384",
                },
            )
        ]
    )

    assert vector_ids == ["42"]
    assert store.kind == "qdrant"
    assert store.path == "http://qdrant:6333/panorama_view_embeddings_siglip2"
    assert client.collection_exists_calls == ["panorama_view_embeddings_siglip2"]
    create_call = client.create_collection_calls[0]
    assert create_call["collection_name"] == "panorama_view_embeddings_siglip2"
    assert create_call["on_disk_payload"] is True
    vectors_config = create_call["vectors_config"]
    assert vectors_config.size == 3
    assert vectors_config.distance == "Cosine"
    assert vectors_config.on_disk is True
    hnsw_config = create_call["hnsw_config"]
    assert hnsw_config.on_disk is False
    upsert_call = client.upsert_calls[0]
    assert upsert_call["collection_name"] == "panorama_view_embeddings_siglip2"
    assert upsert_call["wait"] is True
    point = upsert_call["points"][0]
    assert point.id == 42
    assert point.vector == pytest.approx([0.6, 0.8, 0.0])
    assert point.payload == {
        "embedding_id": 42,
        "view_id": 7,
        "model_id": "google/siglip2-so400m-patch14-384",
    }


def test_add_many_reuses_existing_collection() -> None:
    client = FakeQdrantClient(collection_exists=True)
    store = make_store(client)

    store.add_many(
        [
            VectorStoreRecord(
                vector_id=43,
                vector=np.array([1.0, 0.0, 0.0]),
                metadata={"embedding_id": 43},
            )
        ]
    )

    assert client.create_collection_calls == []
    assert len(client.upsert_calls) == 1


def test_add_many_propagates_qdrant_upsert_failure() -> None:
    client = FakeQdrantClient(collection_exists=True)
    client.raise_on_upsert = RuntimeError("qdrant unavailable")
    store = make_store(client)

    with pytest.raises(RuntimeError, match="qdrant unavailable"):
        store.add_many(
            [
                VectorStoreRecord(
                    vector_id=44,
                    vector=np.array([1.0, 0.0, 0.0]),
                    metadata={"embedding_id": 44},
                )
            ]
        )


def test_search_returns_qdrant_scores_in_rank_order() -> None:
    client = FakeQdrantClient(collection_exists=True)
    store = make_store(client)

    results = store.search(np.array([1.0, 0.0, 0.0]), limit=2)

    assert results == [("10", 0.75), ("11", 0.5)]
    assert client.query_points_calls == [
        {
            "collection_name": "panorama_view_embeddings_siglip2",
            "query": [1.0, 0.0, 0.0],
            "limit": 2,
            "with_payload": False,
            "with_vectors": False,
        }
    ]
