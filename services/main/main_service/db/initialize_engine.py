from sqlalchemy import URL, create_engine

from main_service.config import CONFIG, Settings
from main_service.db.models.base import Base
from main_service.db.models import (
    embedding,
    map_tile,
    map_tile_panorama,
    panorama,
    panorama_view,
    tile,
)


def build_database_url(settings: Settings) -> URL:
    return URL.create(
        settings.db_driver,
        username=settings.db_user,
        password=settings.db_password,
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
    )


def initialize_engine(settings: Settings = CONFIG):
    engine = create_engine(build_database_url(settings))
    Base.metadata.create_all(engine)
    return engine
