from sqlalchemy import URL, create_engine

from main_service.config import CONFIG
from main_service.db.models.base import Base
from main_service.db.models import embedding, panorama, tile


def initialize_engine():
    url = URL.create(
        "mysql+pymysql",
        username=CONFIG.db_user,
        password=CONFIG.db_password,
        host=CONFIG.db_host,
        port=CONFIG.db_port,
        database=CONFIG.db_name,
    )
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return engine
