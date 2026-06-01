import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import Engine, update
from sqlalchemy.orm import Session

from main_service.db.models.panorama_view_embedding import PanoramaViewEmbedding
from main_service.embedding.vector_store import VectorStoreRecord

logger = logging.getLogger(__name__)


class HnswRecordSource(Protocol):
    def iter_records(
        self,
        *,
        batch_size: int,
        limit: int | None = None,
    ) -> Iterable[list[VectorStoreRecord]]:
        ...


class BatchVectorStore(Protocol):
    kind: str
    path: str

    def add_many(self, records: list[VectorStoreRecord]) -> list[str]:
        ...


@dataclass(frozen=True)
class QdrantBackfillResult:
    total_records: int
    transferred_records: int
    updated_db_rows: int
    batches: int


def backfill_local_hnsw_to_qdrant(
    *,
    engine: Engine,
    local_store: HnswRecordSource,
    qdrant_store: BatchVectorStore,
    batch_size: int,
    limit: int | None = None,
) -> QdrantBackfillResult:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    transferred = 0
    updated_db_rows = 0
    batches = 0
    for records in local_store.iter_records(batch_size=batch_size, limit=limit):
        if not records:
            continue
        batches += 1
        vector_ids = qdrant_store.add_many(records)
        updated_db_rows += _mark_embeddings_migrated(
            engine=engine,
            embedding_ids=[int(vector_id) for vector_id in vector_ids],
            vector_store_kind=qdrant_store.kind,
            vector_store_path=qdrant_store.path,
        )
        transferred += len(records)
        logger.info(
            "qdrant_backfill_batch_complete batch=%s records=%s transferred=%s",
            batches,
            len(records),
            transferred,
        )

    result = QdrantBackfillResult(
        total_records=transferred,
        transferred_records=transferred,
        updated_db_rows=updated_db_rows,
        batches=batches,
    )
    logger.info("qdrant_backfill_complete result=%s", result)
    return result


def _mark_embeddings_migrated(
    *,
    engine: Engine,
    embedding_ids: list[int],
    vector_store_kind: str,
    vector_store_path: str,
) -> int:
    if not embedding_ids:
        return 0
    with Session(engine) as session:
        result = session.execute(
            update(PanoramaViewEmbedding)
            .where(PanoramaViewEmbedding.id.in_(embedding_ids))
            .values(
                vector_store_kind=vector_store_kind,
                vector_store_path=vector_store_path,
            )
        )
        session.commit()
        return int(result.rowcount or 0)
