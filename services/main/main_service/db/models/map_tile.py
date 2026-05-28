from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from main_service.ingestion.types import ProcessingStatus

from .base import Base


class MapTile(Base):
    __tablename__ = "map_tile_table"
    __table_args__ = (UniqueConstraint("x", "y", "z", name="uq_map_tile_xyz"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    x: Mapped[int]
    y: Mapped[int]
    z: Mapped[int]
    discovery_status: Mapped[str] = mapped_column(
        default=ProcessingStatus.PENDING.value
    )
    attempt_count: Mapped[int] = mapped_column(default=0)
    last_error: Mapped[str | None] = mapped_column(nullable=True)
