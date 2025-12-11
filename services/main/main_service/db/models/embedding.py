from typing import TYPE_CHECKING
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column, relationship
from .base import Base

if TYPE_CHECKING:
    from .tile import Tile


class Embedding(Base):
    __tablename__ = "embedding_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    tile: Mapped["Tile"] = relationship(back_populates="embedding")

    def __repr__(self) -> str:
        return f"Embedding(id={self.id!r}"
