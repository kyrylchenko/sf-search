from typing import TYPE_CHECKING, List
from sqlalchemy import VARCHAR
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column, relationship
from .base import Base

if TYPE_CHECKING:
    from .tile import Tile


class Panorama(Base):
    __tablename__ = "panorama_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    orig_id: Mapped[str] = mapped_column(VARCHAR(20), unique=True)
    image_hash: Mapped[str] = mapped_column(VARCHAR(20), unique=True)
    latitude: Mapped[int]
    longitude: Mapped[int]
    tiles: Mapped[List["Tile"]] = relationship(back_populates="panorama")

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
