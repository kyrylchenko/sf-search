from sqlalchemy.orm import Session
from sqlalchemy import Engine

from main_service.db.models.embedding import Embedding
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
