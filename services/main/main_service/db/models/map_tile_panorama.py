from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class MapTilePanorama(Base):
    __tablename__ = "map_tile_panorama_table"
    __table_args__ = (
        UniqueConstraint(
            "map_tile_id",
            "panorama_id",
            name="uq_map_tile_panorama",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    map_tile_id: Mapped[int] = mapped_column(ForeignKey("map_tile_table.id"))
    panorama_id: Mapped[int] = mapped_column(ForeignKey("panorama_table.id"))
