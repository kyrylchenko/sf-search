from main_service.db.initialize_engine import initialize_engine
from main_service.db.models.embedding import Embedding
from main_service.db.models.tile import Tile
from main_service.db.services.panorama_service import PanoramaService
import shared.logger
import logging
import sys
from types import TracebackType


def main():
    shared.logger.configure_root_logger()
    logger = logging.getLogger(__package__)

    def exception_hook(
        type: type[BaseException],
        exception: BaseException,
        exception_traceback: TracebackType,
    ):
        logger.critical(
            "Unhandled exception",
            exc_info=(type, exception, exception_traceback),
        )

    sys.excepthook = exception_hook

    engine = initialize_engine()
    pano_service = PanoramaService(engine)
    # pano_service.create_tile(Tile())
    pano_service.create_embedding(Embedding())
    #     geojson_data: dict = json.loads(file.read())
    # tiles = generate_tiles_given_geojson(geojson_data, 17)
    # print(tiles)
    # print(f"Found {len(tiles)}")


if __name__ == "__main__":
    main()
