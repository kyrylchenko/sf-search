import logging

from sqlalchemy import Engine, exists, select
from sqlalchemy.orm import Session

from main_service.db.models.panorama import Panorama
from main_service.db.models.panorama_view import PanoramaView
from main_service.db.models.panorama_view_embedding import PanoramaViewEmbedding
from main_service.db.services.panorama_view_embedding_service import EmbeddingModelSpec
from main_service.ingestion.download_queue import (
    PanoEmbeddingMessage,
    PanoEmbeddingQueue,
    PanoProcessingMessage,
    PanoProcessingQueue,
)
from main_service.ingestion.types import DownloadStatus, PanoramaId, ProcessingStatus

logger = logging.getLogger(__name__)


def requeue_processing_jobs_from_db(
    *,
    engine: Engine,
    processing_queue: PanoProcessingQueue,
    limit: int,
    include_already_processed: bool = False,
) -> int:
    with Session(engine) as session:
        statement = (
            select(Panorama)
            .where(Panorama.download_status == DownloadStatus.DOWNLOADED.value)
            .where(Panorama.image_path.is_not(None))
            .order_by(Panorama.id)
            .limit(limit)
        )
        if not include_already_processed:
            complete_view_exists = (
                exists()
                .where(PanoramaView.panorama_id == Panorama.id)
                .where(PanoramaView.processing_status == ProcessingStatus.COMPLETE.value)
                .where(PanoramaView.image_path.is_not(None))
            )
            statement = statement.where(~complete_view_exists)
        panoramas = session.execute(statement).scalars().all()

    for panorama in panoramas:
        processing_queue.enqueue(
            PanoProcessingMessage(
                pano_id=PanoramaId(panorama.orig_id),
                image_path=panorama.image_path or "",
            )
        )
    logger.info("📤 processing_requeued_from_db count=%s", len(panoramas))
    return len(panoramas)


def requeue_embedding_jobs_from_db(
    *,
    engine: Engine,
    embedding_queue: PanoEmbeddingQueue,
    model_spec: EmbeddingModelSpec,
    limit: int,
    include_already_embedded: bool = False,
) -> int:
    with Session(engine) as session:
        statement = (
            select(PanoramaView, Panorama)
            .join(Panorama, Panorama.id == PanoramaView.panorama_id)
            .where(PanoramaView.processing_status == ProcessingStatus.COMPLETE.value)
            .where(PanoramaView.image_path.is_not(None))
            .where(PanoramaView.image_hash.is_not(None))
            .order_by(PanoramaView.id)
            .limit(limit)
        )
        if not include_already_embedded:
            complete_embedding_exists = (
                exists()
                .where(PanoramaViewEmbedding.panorama_view_id == PanoramaView.id)
                .where(PanoramaViewEmbedding.model_provider == model_spec.model_provider)
                .where(PanoramaViewEmbedding.model_id == model_spec.model_id)
                .where(PanoramaViewEmbedding.model_revision == model_spec.model_revision)
                .where(
                    PanoramaViewEmbedding.preprocess_version
                    == model_spec.preprocess_version
                )
                .where(PanoramaViewEmbedding.source_image_hash == PanoramaView.image_hash)
                .where(
                    PanoramaViewEmbedding.embedding_status
                    == ProcessingStatus.COMPLETE.value
                )
                .where(PanoramaViewEmbedding.vector_id.is_not(None))
            )
            statement = statement.where(~complete_embedding_exists)
        rows = session.execute(statement).all()

    for view, panorama in rows:
        embedding_queue.enqueue(
            PanoEmbeddingMessage(
                pano_id=PanoramaId(panorama.orig_id),
                view_id=view.id,
                image_path=view.image_path or "",
            )
        )
    logger.info("📤 embedding_requeued_from_db count=%s", len(rows))
    return len(rows)
