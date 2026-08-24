import logging
import sys

from app.errors import request_id_var

LOG_FORMAT = "%(asctime)s %(levelname)s request_id=%(request_id)s %(name)s %(message)s"


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("-")
        return True


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
