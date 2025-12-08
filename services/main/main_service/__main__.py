from main_service.config import CONFIG
from main_service.geo import generate_tiles_given_geojson
import json
import shared.logger
import logging
import sys
from types import TracebackType


def main():
    shared.logger.configure_root_logger()
    print(__name__)

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

    #     geojson_data: dict = json.loads(file.read())
    # tiles = generate_tiles_given_geojson(geojson_data, 17)
    # print(tiles)
    # print(f"Found {len(tiles)}")

    sys.excepthook = exception_hook


if __name__ == "__main__":
    main()
