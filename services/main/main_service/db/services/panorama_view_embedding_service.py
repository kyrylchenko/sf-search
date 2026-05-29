from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from main_service.db.models.panorama_view import PanoramaView
from main_service.db.models.panorama_view_embedding import PanoramaViewEmbedding
from main_service.ingestion.types import ProcessingStatus


@dataclass(frozen=True)
class EmbeddingModelSpec:
    model_provider: str
    model_id: str
    model_revision: str
    preprocess_version: str
    embedding_dimension: int
    embedding_dtype: str
    embedding_normalized: bool


class PanoramaViewEmbeddingService:
    def __init__(self, engine: Engine):
        self.engine = engine

    def claim_embedding_for_view(
        self,
        view_id: int,
        model_spec: EmbeddingModelSpec,
    ) -> PanoramaViewEmbedding | None:
        with Session(self.engine) as session:
            view = session.get(PanoramaView, view_id)
            if (
                view is None
                or view.processing_status != ProcessingStatus.COMPLETE.value
                or not view.image_path
                or not view.image_hash
            ):
                return None

            embedding = session.execute(
                select(PanoramaViewEmbedding).filter_by(
                    panorama_view_id=view.id,
                    model_provider=model_spec.model_provider,
                    model_id=model_spec.model_id,
                    model_revision=model_spec.model_revision,
                    preprocess_version=model_spec.preprocess_version,
                    source_image_hash=view.image_hash,
                )
            ).scalar_one_or_none()
            if (
                embedding is not None
                and embedding.embedding_status == ProcessingStatus.COMPLETE.value
                and embedding.vector_id
            ):
                return None

            if embedding is None:
                embedding = PanoramaViewEmbedding(
                    panorama_view_id=view.id,
                    model_provider=model_spec.model_provider,
                    model_id=model_spec.model_id,
                    model_revision=model_spec.model_revision,
                    preprocess_version=model_spec.preprocess_version,
                    source_image_hash=view.image_hash,
                )
                session.add(embedding)

            embedding.source_image_path = view.image_path
            embedding.source_image_hash = view.image_hash
            embedding.source_image_bytes = view.image_bytes
            embedding.embedding_dimension = model_spec.embedding_dimension
            embedding.embedding_dtype = model_spec.embedding_dtype
            embedding.embedding_normalized = model_spec.embedding_normalized
            embedding.embedding_status = ProcessingStatus.PROCESSING.value
            embedding.embedding_attempt_count = (
                embedding.embedding_attempt_count or 0
            ) + 1
            embedding.last_error = None
            embedding.vector_store_kind = None
            embedding.vector_store_path = None
            embedding.vector_id = None
            embedding.embedded_at = None

            view.embedding_status = ProcessingStatus.PROCESSING.value
            view.embedding_model = model_spec.model_id
            view.embedding_attempt_count = (view.embedding_attempt_count or 0) + 1

            session.flush()
            session.refresh(embedding)
            session.expunge(embedding)
            session.commit()
            return embedding

    def mark_embedding_complete(
        self,
        embedding_id: int,
        *,
        vector_store_kind: str,
        vector_store_path: str,
        vector_id: str,
    ) -> PanoramaViewEmbedding:
        with Session(self.engine) as session:
            embedding = self._get_embedding(session, embedding_id)
            embedding.embedding_status = ProcessingStatus.COMPLETE.value
            embedding.vector_store_kind = vector_store_kind
            embedding.vector_store_path = vector_store_path
            embedding.vector_id = vector_id
            embedding.last_error = None
            embedding.embedded_at = datetime.now(timezone.utc)
            view = embedding.panorama_view
            view.embedding_status = ProcessingStatus.COMPLETE.value
            view.embedding_model = embedding.model_id
            view.embedding_vector_id = vector_id
            session.flush()
            session.refresh(embedding)
            session.expunge(embedding)
            session.commit()
            return embedding

    def mark_embedding_failed(
        self,
        embedding_id: int,
        error: str,
    ) -> PanoramaViewEmbedding:
        with Session(self.engine) as session:
            embedding = self._get_embedding(session, embedding_id)
            embedding.embedding_status = ProcessingStatus.FAILED.value
            embedding.last_error = error[:2000]
            embedding.embedded_at = datetime.now(timezone.utc)
            view = embedding.panorama_view
            view.embedding_status = ProcessingStatus.FAILED.value
            view.embedding_model = embedding.model_id
            session.flush()
            session.refresh(embedding)
            session.expunge(embedding)
            session.commit()
            return embedding

    def list_embeddings_for_view(self, view_id: int) -> list[PanoramaViewEmbedding]:
        with Session(self.engine) as session:
            rows = (
                session.execute(
                    select(PanoramaViewEmbedding)
                    .where(PanoramaViewEmbedding.panorama_view_id == view_id)
                    .order_by(PanoramaViewEmbedding.id)
                )
                .scalars()
                .all()
            )
            for row in rows:
                session.expunge(row)
            return rows

    def get_completed_embedding(self, embedding_id: int) -> PanoramaViewEmbedding | None:
        with Session(self.engine) as session:
            embedding = session.get(PanoramaViewEmbedding, embedding_id)
            if (
                embedding is None
                or embedding.embedding_status != ProcessingStatus.COMPLETE.value
                or not embedding.vector_id
            ):
                return None
            session.expunge(embedding)
            return embedding

    def _get_embedding(
        self,
        session: Session,
        embedding_id: int,
    ) -> PanoramaViewEmbedding:
        embedding = session.get(PanoramaViewEmbedding, embedding_id)
        if embedding is None:
            raise ValueError(f"Panorama view embedding does not exist: {embedding_id}")
        return embedding
