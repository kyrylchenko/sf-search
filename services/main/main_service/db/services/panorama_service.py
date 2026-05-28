from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import Engine, select
from main_service.db.models.embedding import Embedding
from main_service.db.models.panorama import Panorama
from ..models.tile import Tile


class PanoramaService:
    def __init__(self, engine: Engine):
        self.engine = engine

    def create_tile(self, tile: Tile):
        with Session(self.engine) as session:
            session.add(tile)
            session.commit()

    def create_embedding(self, embedding: Embedding):
        with Session(self.engine) as session:
            session.add(embedding)
            session.commit()

    def find_panorama_by_orig_id(self, orig_id: str) -> Optional[Panorama]:
        with Session(self.engine) as session:
            return session.execute(
                select(Panorama).filter_by(orig_id=orig_id)
            ).scalar_one_or_none()
