from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import JSON, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from main_service.ingestion.types import ProcessingStatus

from .base import Base

if TYPE_CHECKING:
    from .panorama import Panorama


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PanoramaView(Base):
    __tablename__ = "panorama_view_table"
    __table_args__ = (
        UniqueConstraint(
            "panorama_id",
            "viewset_name",
            "view_id",
            "view_spec_hash",
            "render_scale",
            "output_format",
            "source_image_hash",
            name="uq_panorama_view_render",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    panorama_id: Mapped[int] = mapped_column(ForeignKey("panorama_table.id"))
    panorama: Mapped["Panorama"] = relationship(back_populates="views")

    viewset_name: Mapped[str]
    viewset_description: Mapped[str] = mapped_column(default="")
    view_id: Mapped[str]
    view_kind: Mapped[str]
    view_spec_json: Mapped[dict[str, object]] = mapped_column(JSON)
    view_spec_hash: Mapped[str]

    relative_heading: Mapped[float]
    pitch: Mapped[float]
    fov: Mapped[float]
    output_width: Mapped[int]
    output_height: Mapped[int]
    render_scale: Mapped[int]
    rendered_width: Mapped[int]
    rendered_height: Mapped[int]
    output_format: Mapped[str]
    image_quality: Mapped[int | None] = mapped_column(nullable=True)
    interpolation_mode: Mapped[str]
    renderer_version: Mapped[str]

    source_image_path: Mapped[str]
    source_image_hash: Mapped[str]
    image_path: Mapped[str | None] = mapped_column(nullable=True)
    image_hash: Mapped[str | None] = mapped_column(nullable=True)
    image_bytes: Mapped[int | None] = mapped_column(nullable=True)

    processing_status: Mapped[str] = mapped_column(
        default=ProcessingStatus.PENDING.value
    )
    attempt_count: Mapped[int] = mapped_column(default=0)
    last_error: Mapped[str | None] = mapped_column(nullable=True)
    embedding_status: Mapped[str] = mapped_column(default=ProcessingStatus.PENDING.value)
    embedding_attempt_count: Mapped[int] = mapped_column(default=0)
    embedding_model: Mapped[str | None] = mapped_column(nullable=True)
    embedding_vector_id: Mapped[str | None] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    def __repr__(self) -> str:
        return (
            "PanoramaView("
            f"id={self.id!r}, "
            f"panorama_id={self.panorama_id!r}, "
            f"viewset_name={self.viewset_name!r}, "
            f"view_id={self.view_id!r}, "
            f"processing_status={self.processing_status!r}"
            ")"
        )
