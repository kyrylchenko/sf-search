from typing import TYPE_CHECKING
from sqlalchemy import ForeignKey
from .base import Base
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column, relationship

if TYPE_CHECKING:
    from .panorama import Panorama
    from .embedding import Embedding


class Tile(Base):
    __tablename__ = "tile_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    panorama_id: Mapped[int] = mapped_column(ForeignKey("panorama_table.id"))
    panorama: Mapped["Panorama"] = relationship(back_populates="tiles")
    image_id: Mapped[int] = mapped_column()
    embedding_id: Mapped[int] = mapped_column(ForeignKey("embedding_table.id"))
    embedding: Mapped["Embedding"] = relationship(back_populates="tile")
    pitch: Mapped[float]
    yaw: Mapped[float]
    roll: Mapped[float]
    fov: Mapped[float]
    google_pitch: Mapped[float]
    google_yaw: Mapped[float]
    google_roll: Mapped[float]
    google_fov: Mapped[float]

    def __repr__(self) -> str:
        return (
            "Tile("
            f"id={self.id!r}, "
            f"panorama_id={self.panorama_id!r}, "
            f"image_id={self.image_id!r}, "
            f"pitch={self.pitch!r}, "
            f"yaw={self.yaw!r}, "
            f"roll={self.roll!r}, "
            f"fov={self.fov!r}, "
            f"google_pitch={self.google_pitch!r}, "
            f"google_yaw={self.google_yaw!r}, "
            f"google_roll={self.google_roll!r}, "
            f"google_fov={self.google_fov!r}"
            ")"
        )
