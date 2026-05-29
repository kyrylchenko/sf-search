from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from main_service.ingestion.types import ProcessingStatus

from .base import Base

if TYPE_CHECKING:
    from .panorama_view import PanoramaView


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PanoramaViewEmbedding(Base):
    __tablename__ = "panorama_view_embedding_table"
    __table_args__ = (
        UniqueConstraint(
            "panorama_view_id",
            "model_provider",
            "model_id",
            "model_revision",
            "preprocess_version",
            "source_image_hash",
            name="uq_panorama_view_embedding_model_source",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    panorama_view_id: Mapped[int] = mapped_column(
        ForeignKey("panorama_view_table.id")
    )
    panorama_view: Mapped["PanoramaView"] = relationship(back_populates="embeddings")

    model_provider: Mapped[str]
    model_id: Mapped[str]
    model_revision: Mapped[str]
    preprocess_version: Mapped[str]

    source_image_path: Mapped[str]
    source_image_hash: Mapped[str]
    source_image_bytes: Mapped[int | None] = mapped_column(nullable=True)

    embedding_status: Mapped[str] = mapped_column(
        default=ProcessingStatus.PENDING.value
    )
    embedding_attempt_count: Mapped[int] = mapped_column(default=0)
    last_error: Mapped[str | None] = mapped_column(nullable=True)
    embedding_dimension: Mapped[int]
    embedding_dtype: Mapped[str]
    embedding_normalized: Mapped[bool] = mapped_column(default=True)

    vector_store_kind: Mapped[str | None] = mapped_column(nullable=True)
    vector_store_path: Mapped[str | None] = mapped_column(nullable=True)
    vector_id: Mapped[str | None] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)
    embedded_at: Mapped[datetime | None] = mapped_column(nullable=True)

    def __repr__(self) -> str:
        return (
            "PanoramaViewEmbedding("
            f"id={self.id!r}, "
            f"panorama_view_id={self.panorama_view_id!r}, "
            f"model_id={self.model_id!r}, "
            f"embedding_status={self.embedding_status!r}"
            ")"
        )
