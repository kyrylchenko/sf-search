import json
from pathlib import Path

import numpy as np

from main_service.embedding import vector_store
from main_service.embedding.vector_store import LocalHnswVectorStore, VectorStoreRecord


class FakeIndex:
    load_calls = 0
    init_calls = 0
    add_calls = 0
    save_calls = 0
    resize_calls = 0
    save_paths: list[str] = []
    item_vectors: dict[int, np.ndarray] = {
        1: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }

    def __init__(self, *, space: str, dim: int) -> None:
        self.space = space
        self.dim = dim
        self.ef_values: list[int] = []

    def load_index(self, path: str, *, max_elements: int) -> None:
        FakeIndex.load_calls += 1
        self.path = path
        self.max_elements = max_elements

    def init_index(self, *, max_elements: int, ef_construction: int, M: int) -> None:
        FakeIndex.init_calls += 1
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.m = M

    def set_ef(self, ef: int) -> None:
        self.ef_values.append(ef)

    def knn_query(self, _vector: np.ndarray, *, k: int) -> tuple[np.ndarray, np.ndarray]:
        labels = np.array([[label for label in range(1, k + 1)]], dtype=np.int64)
        distances = np.array([[0.25 for _ in range(k)]], dtype=np.float32)
        return labels, distances

    def add_items(self, _vectors: np.ndarray, _labels: np.ndarray) -> None:
        FakeIndex.add_calls += 1

    def resize_index(self, _max_elements: int) -> None:
        FakeIndex.resize_calls += 1

    def save_index(self, path: str) -> None:
        FakeIndex.save_calls += 1
        FakeIndex.save_paths.append(path)
        Path(path).touch()

    def get_items(self, labels: np.ndarray) -> np.ndarray:
        return np.stack([self.item_vectors[int(label)] for label in labels])


class FakeHnswLib:
    Index = FakeIndex


def reset_fake_index() -> None:
    FakeIndex.load_calls = 0
    FakeIndex.init_calls = 0
    FakeIndex.add_calls = 0
    FakeIndex.save_calls = 0
    FakeIndex.resize_calls = 0
    FakeIndex.save_paths = []


def write_existing_index(root_dir: Path) -> None:
    root_dir.mkdir(parents=True)
    (root_dir / "index.bin").write_bytes(b"fake-index")
    (root_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dimension": 3,
                "max_elements": 100,
                "items": {
                    "1": {"embedding_id": 1, "view_id": 10},
                    "2": {"embedding_id": 2, "view_id": 20},
                },
            }
        )
    )


def make_store(tmp_path: Path, now: list[float]) -> LocalHnswVectorStore:
    return LocalHnswVectorStore(
        root_dir=tmp_path,
        model_id="test/model",
        dimension=3,
        search_cache_ttl_seconds=300.0,
        clock=lambda: now[0],
    )


def test_search_reuses_loaded_index_within_cache_ttl(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_fake_index()
    monkeypatch.setattr(vector_store, "_import_hnswlib", lambda: FakeHnswLib)
    write_existing_index(tmp_path / "test_model")
    now = [1000.0]
    store = make_store(tmp_path, now)

    first = store.search(np.array([1.0, 0.0, 0.0]), limit=2)
    second = store.search(np.array([0.0, 1.0, 0.0]), limit=2)

    assert first == [("1", 0.75), ("2", 0.75)]
    assert second == [("1", 0.75), ("2", 0.75)]
    assert FakeIndex.load_calls == 1


def test_search_reloads_index_after_cache_ttl(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_fake_index()
    monkeypatch.setattr(vector_store, "_import_hnswlib", lambda: FakeHnswLib)
    write_existing_index(tmp_path / "test_model")
    now = [1000.0]
    store = make_store(tmp_path, now)

    store.search(np.array([1.0, 0.0, 0.0]), limit=2)
    now[0] = 1300.0
    store.search(np.array([1.0, 0.0, 0.0]), limit=2)

    assert FakeIndex.load_calls == 2


def test_add_invalidates_cached_search_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_fake_index()
    monkeypatch.setattr(vector_store, "_import_hnswlib", lambda: FakeHnswLib)
    write_existing_index(tmp_path / "test_model")
    now = [1000.0]
    store = make_store(tmp_path, now)

    store.search(np.array([1.0, 0.0, 0.0]), limit=2)
    store.add(
        vector_id=3,
        vector=np.array([0.0, 0.0, 1.0]),
        metadata={"embedding_id": 3, "view_id": 30},
    )
    store.search(np.array([1.0, 0.0, 0.0]), limit=2)

    assert FakeIndex.load_calls == 3
    assert FakeIndex.add_calls == 1
    assert FakeIndex.save_calls == 1
    assert Path(FakeIndex.save_paths[0]).name.startswith(".index.bin.")
    assert Path(FakeIndex.save_paths[0]).name.endswith(".tmp")
    assert (tmp_path / "test_model" / "index.bin").exists()
    assert not [
        path
        for path in (tmp_path / "test_model").iterdir()
        if path.name.endswith(".tmp")
    ]


def test_add_many_writes_multiple_records_in_one_index_update(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_fake_index()
    monkeypatch.setattr(vector_store, "_import_hnswlib", lambda: FakeHnswLib)
    now = [1000.0]
    store = make_store(tmp_path, now)

    vector_ids = store.add_many(
        [
            VectorStoreRecord(
                vector_id=10,
                vector=np.array([1.0, 0.0, 0.0]),
                metadata={"embedding_id": 10, "view_id": 100},
            ),
            VectorStoreRecord(
                vector_id=11,
                vector=np.array([0.0, 1.0, 0.0]),
                metadata={"embedding_id": 11, "view_id": 110},
            ),
        ]
    )

    metadata = json.loads((tmp_path / "test_model" / "metadata.json").read_text())
    assert vector_ids == ["10", "11"]
    assert FakeIndex.init_calls == 1
    assert FakeIndex.add_calls == 1
    assert FakeIndex.save_calls == 1
    assert metadata["items"] == {
        "10": {"embedding_id": 10, "view_id": 100},
        "11": {"embedding_id": 11, "view_id": 110},
    }


def test_iter_records_exports_existing_vectors_in_stable_batches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_fake_index()
    monkeypatch.setattr(vector_store, "_import_hnswlib", lambda: FakeHnswLib)
    write_existing_index(tmp_path / "test_model")
    now = [1000.0]
    store = make_store(tmp_path, now)

    batches = list(store.iter_records(batch_size=1))

    assert len(batches) == 2
    assert batches[0][0].vector_id == 1
    assert batches[0][0].metadata == {"embedding_id": 1, "view_id": 10}
    assert batches[0][0].vector.tolist() == [1.0, 0.0, 0.0]
    assert batches[1][0].vector_id == 2
    assert batches[1][0].metadata == {"embedding_id": 2, "view_id": 20}
    assert batches[1][0].vector.tolist() == [0.0, 1.0, 0.0]
