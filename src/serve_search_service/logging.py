"""JSON logging for the search service."""

import json
import logging
import socket
import sys
from collections.abc import Mapping, MutableMapping
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler, SocketHandler
from pathlib import Path
from typing import Any

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_app_name: ContextVar[str | None] = ContextVar("app_name", default=None)
_handlers: list[logging.Handler] = []


def bind_log_context(trace_id: str, app_name: str) -> tuple[Token[str | None], Token[str | None]]:
    return _trace_id.set(trace_id), _app_name.set(app_name)


def reset_log_context(tokens: tuple[Token[str | None], Token[str | None]]) -> None:
    _trace_id.reset(tokens[0])
    _app_name.reset(tokens[1])


def current_trace_id() -> str | None:
    return _trace_id.get()


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "trace_id"):
            record.trace_id = _trace_id.get()
        if not hasattr(record, "app_name"):
            record.app_name = _app_name.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, UTC).strftime("%Y-%m-%d %H:%M:%S.%f")
        payload: dict[str, object] = {
            "timestamp": timestamp[:-3],
            "level": record.levelname,
            "thread": record.threadName,
            "traceId": getattr(record, "trace_id", None),
            "hostname": socket.gethostname(),
            "message": record.getMessage(),
            "logger": record.name,
            "appName": getattr(record, "app_name", None),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class TraceIdAdapter(logging.LoggerAdapter[logging.Logger]):
    def process(
        self, msg: object, kwargs: MutableMapping[str, Any]
    ) -> tuple[object, MutableMapping[str, Any]]:
        supplied_trace_id = kwargs.pop("trace_id", kwargs.pop("traceId", current_trace_id()))
        supplied_app_name = kwargs.pop("app_name", kwargs.pop("appName", _app_name.get()))
        supplied_extra = kwargs.get("extra", {})
        extra = dict(supplied_extra) if isinstance(supplied_extra, Mapping) else {}
        extra["trace_id"] = supplied_trace_id
        extra["app_name"] = supplied_app_name
        kwargs["extra"] = extra
        return msg, kwargs


def configure_logging(*, app_name: str, log_directory: Path) -> None:
    """Configure shared stdout JSON, rotating file, and local socket handlers once."""
    if _handlers:
        return
    log_directory.mkdir(parents=True, exist_ok=True)
    formatter = JsonFormatter()
    context_filter = ContextFilter()
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = RotatingFileHandler(
        log_directory / "app.log", maxBytes=10 * 1024 * 1024, backupCount=4, encoding="utf-8"
    )
    socket_handler = SocketHandler("127.0.0.1", 19996)
    for handler in (console_handler, file_handler):
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)
    socket_handler.setLevel(logging.DEBUG)
    _handlers.extend((console_handler, file_handler, socket_handler))
    _app_name.set(app_name)


def get_logger(name: str) -> TraceIdAdapter:
    """Return a named logger that emits the template's JSON record schema."""
    if not _handlers:
        configure_logging(app_name="serve-search-service", log_directory=Path("logs"))
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    for handler in _handlers:
        logger.addHandler(handler)
    logger.propagate = False
    return TraceIdAdapter(logger, {})
