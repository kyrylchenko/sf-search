import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from main_service.db.models.base import Base
from main_service.db.models.panorama import Panorama
from main_service.db.models.panorama_view import PanoramaView
from main_service.db.models.panorama_view_embedding import PanoramaViewEmbedding
from main_service.embedding.vector_store import VectorStoreRecord
from main_service.ingestion.types import DownloadStatus, ProcessingStatus
from main_service.ops.__main__ import build_parser
from main_service.ops.vector_backfill import backfill_local_hnsw_to_qdrant


class FakeLocalStore:
    def __init__(self, batches: list[list[VectorStoreRecord]]) -> None:
        self.batches = batches
        self.calls: list[dict[str, int | None]] = []

    def iter_records(
        self,
        *,
        batch_size: int,
        limit: int | None = None,
    ) -> list[list[VectorStoreRecord]]:
        self.calls.append({"batch_size": batch_size, "limit": limit})
        return self.batches


class FakeQdrantStore:
    kind = "qdrant"
    path = "http://qdrant:6333/panorama_view_embeddings_siglip2"

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.batches: list[list[int]] = []

    def add_many(self, records: list[VectorStoreRecord]) -> list[str]:
        if self.error is not None:
            raise self.error
        self.batches.append([record.vector_id for record in records])
        return [str(record.vector_id) for record in records]


def make_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def seed_embedding(engine, embedding_id: int) -> None:
    with Session(engine) as session:
        panorama = Panorama(
            orig_id=f"pano-{embedding_id}",
            image_hash=f"pano-hash-{embedding_id}",
            download_status=DownloadStatus.DOWNLOADED.value,
        )
        session.add(panorama)
        session.flush()
        view = PanoramaView(
            panorama_id=panorama.id,
            viewset_name="small",
            viewset_description="small views",
            view_id=f"small-{embedding_id}",
            view_kind="object",
            view_spec_json={"id": f"small-{embedding_id}"},
            view_spec_hash=f"spec-hash-{embedding_id}",
            relative_heading=0,
            pitch=0,
            fov=40,
            output_width=512,
            output_height=512,
            render_scale=2,
            rendered_width=1024,
            rendered_height=1024,
            output_format="jpeg",
            image_quality=95,
            interpolation_mode="bicubic",
            renderer_version="py360convert.e2p",
            source_image_path=f".local/panos/pano-{embedding_id}.jpg",
            source_image_hash=f"source-hash-{embedding_id}",
            image_path=f".local/views/{embedding_id}.jpg",
            image_hash=f"view-hash-{embedding_id}",
            image_bytes=123,
            processing_status=ProcessingStatus.COMPLETE.value,
        )
        session.add(view)
        session.flush()
        session.add(
            PanoramaViewEmbedding(
                id=embedding_id,
                panorama_view_id=view.id,
                model_provider="transformers",
                model_id="google/siglip2-so400m-patch14-384",
                model_revision="main",
                preprocess_version="siglip2-384-rgb-v1",
                source_image_path=view.image_path or "",
                source_image_hash=view.image_hash or "",
                source_image_bytes=view.image_bytes,
                embedding_status=ProcessingStatus.COMPLETE.value,
                embedding_dimension=3,
                embedding_dtype="float32",
                embedding_normalized=True,
                vector_store_kind="local_hnsw",
                vector_store_path=".local/embedding-indexes/index.bin",
                vector_id=str(embedding_id),
            )
        )
        session.commit()


def get_embedding(engine, embedding_id: int) -> PanoramaViewEmbedding:
    with Session(engine) as session:
        row = session.execute(
            select(PanoramaViewEmbedding).where(PanoramaViewEmbedding.id == embedding_id)
        ).scalar_one()
        session.expunge(row)
        return row


def make_record(vector_id: int) -> VectorStoreRecord:
    return VectorStoreRecord(
        vector_id=vector_id,
        vector=np.array([1.0, 0.0, 0.0]),
        metadata={"embedding_id": vector_id, "view_id": vector_id + 100},
    )


def test_backfill_copies_records_to_qdrant_and_marks_db_rows() -> None:
    engine = make_engine()
    seed_embedding(engine, 1)
    seed_embedding(engine, 2)
    local_store = FakeLocalStore([[make_record(1), make_record(2)]])
    qdrant_store = FakeQdrantStore()

    result = backfill_local_hnsw_to_qdrant(
        engine=engine,
        local_store=local_store,
        qdrant_store=qdrant_store,
        batch_size=2,
    )

    assert result.total_records == 2
    assert result.transferred_records == 2
    assert result.updated_db_rows == 2
    assert result.batches == 1
    assert local_store.calls == [{"batch_size": 2, "limit": None}]
    assert qdrant_store.batches == [[1, 2]]
    assert get_embedding(engine, 1).vector_store_kind == "qdrant"
    assert get_embedding(engine, 1).vector_store_path == qdrant_store.path
    assert get_embedding(engine, 2).vector_store_kind == "qdrant"


def test_backfill_does_not_update_db_when_qdrant_upsert_fails() -> None:
    engine = make_engine()
    seed_embedding(engine, 1)
    local_store = FakeLocalStore([[make_record(1)]])
    qdrant_store = FakeQdrantStore(error=RuntimeError("qdrant unavailable"))

    with pytest.raises(RuntimeError, match="qdrant unavailable"):
        backfill_local_hnsw_to_qdrant(
            engine=engine,
            local_store=local_store,
            qdrant_store=qdrant_store,
            batch_size=1,
        )

    assert get_embedding(engine, 1).vector_store_kind == "local_hnsw"


def test_ops_parser_accepts_qdrant_backfill_command() -> None:
    args = build_parser().parse_args(
        [
            "backfill-qdrant-from-local-hnsw",
            "--batch-size",
            "128",
            "--limit",
            "1000",
        ]
    )

    assert args.command == "backfill-qdrant-from-local-hnsw"
    assert args.batch_size == 128
    assert args.limit == 1000
