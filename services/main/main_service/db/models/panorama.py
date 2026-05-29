from datetime import datetime
from typing import TYPE_CHECKING, List

from sqlalchemy import JSON, VARCHAR
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column, relationship

from main_service.ingestion.types import DownloadStatus, ProcessingStatus

from .base import Base

if TYPE_CHECKING:
    from .tile import Tile
    from .panorama_view import PanoramaView


class Panorama(Base):
    __tablename__ = "panorama_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    orig_id: Mapped[str] = mapped_column(VARCHAR(64), unique=True)
    image_hash: Mapped[str | None] = mapped_column(VARCHAR(128), unique=True)
    latitude: Mapped[float | None] = mapped_column(nullable=True)
    longitude: Mapped[float | None] = mapped_column(nullable=True)
    metadata_status: Mapped[str] = mapped_column(default=ProcessingStatus.PENDING.value)
    download_status: Mapped[str] = mapped_column(default=DownloadStatus.PENDING.value)
    discovered_at_tile_count: Mapped[int] = mapped_column(default=0)
    attempt_count: Mapped[int] = mapped_column(default=0)
    last_error: Mapped[str | None] = mapped_column(nullable=True)
    image_path: Mapped[str | None] = mapped_column(nullable=True)
    metadata_json: Mapped[dict[str, object] | None] = mapped_column(
        JSON,
        nullable=True,
    )
    downloaded_at: Mapped[datetime | None] = mapped_column(nullable=True)
    tiles: Mapped[List["Tile"]] = relationship(back_populates="panorama")
    views: Mapped[List["PanoramaView"]] = relationship(back_populates="panorama")

    def __repr__(self) -> str:
        return (
            "Panorama("
            f"id={self.id!r}, "
            f"orig_id={self.orig_id!r}, "
            f"image_hash={self.image_hash!r}, "
            f"latitude={self.latitude!r}, "
            f"longitude={self.longitude!r}"
            ")"
        )
