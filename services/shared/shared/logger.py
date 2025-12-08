import logging
import logging.handlers
import logging.config
from typing import cast, override
import datetime as dt
import atexit


class CustomFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return (
            dt.datetime.fromtimestamp(record.created)
            .astimezone()
            .isoformat(timespec="milliseconds")
        )


root_logger_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "shared.logger.CustomFormatter",
            "format": "%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(lineno)d:%(message)s",
        },
        "json": {"()": "pythonjsonlogger.json.JsonFormatter", "reserved_attrs": []},
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file_normal": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "DEBUG",
            "when": "D",
            "interval": 1,
            "formatter": "default",
            "filename": "logs/default/log.log",
            "backupCount": 31,
        },
        "file_json": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "DEBUG",
            "when": "D",
            "interval": 1,
            "formatter": "json",
            "filename": "logs/json/log.jsonl",
            "backupCount": 31,
        },
        # "queue_handler": {
        #     "class": "logging.handlers.QueueHandler",
        #     "handlers": ["stdout", "file_normal", "file_json"],
        #     "respect_handler_level": True,
        # },
    },
    "loggers": {
        "root": {"level": "DEBUG", "handlers": ["file_json", "file_normal", "stdout"]}
    },
}


def configure_root_logger():
    logging.config.dictConfig(root_logger_config)
    original_makeRecord = logging.Logger.makeRecord

    def make_record_with_extra(
        self,
        name,
        level,
        fn,
        lno,
        msg,
        args,
        exc_info,
        func=None,
        extra=None,
        sinfo=None,
    ):
        record = original_makeRecord(
            self, name, level, fn, lno, msg, args, exc_info, func, None, sinfo
        )
        record.extra = extra
        return record

    logging.Logger.makeRecord = make_record_with_extra

    queue_handler: logging.handlers.QueueHandler | None = cast(
        logging.handlers.QueueHandler, logging.getHandlerByName("queue_handler")
    )

    if queue_handler is not None and queue_handler.listener is not None:
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)
