import logging
import sys
from types import TracebackType

import shared.logger


def configure_exception_logging(logger: logging.Logger) -> None:
    def exception_hook(
        exception_type: type[BaseException],
        exception: BaseException,
        exception_traceback: TracebackType,
    ) -> None:
        logger.critical(
            "Unhandled exception",
            exc_info=(exception_type, exception, exception_traceback),
        )

    sys.excepthook = exception_hook


def main() -> None:
    shared.logger.configure_root_logger()
    logger = logging.getLogger(__package__)
    configure_exception_logging(logger)
    logger.info("main service scaffold started")


if __name__ == "__main__":
    main()
