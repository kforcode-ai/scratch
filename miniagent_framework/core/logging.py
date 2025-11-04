"""Structured logging utilities for MiniAgent."""
import logging
import os
import sys
from typing import Any, Dict

import structlog
from structlog.contextvars import merge_contextvars

SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "token",
    "auth",
    "authorization",
    "password",
    "secret",
    "access_key",
}


def _mask_sensitive(value: Any, redactions: set[str]) -> Any:
    if isinstance(value, dict):
        return {k: _mask_sensitive(v, redactions) if k not in SENSITIVE_KEYS else _mask_value(v, k, redactions) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask_sensitive(item, redactions) for item in value]
    return value


def _mask_value(value: Any, key: str, redactions: set[str]) -> str:
    redactions.add(key)
    return "***REDACTED***"


def _redact_sensitive(_: logging.Logger, __: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    redactions: set[str] = set()

    for key in list(event_dict.keys()):
        if key in SENSITIVE_KEYS:
            event_dict[key] = _mask_value(event_dict[key], key, redactions)
        else:
            event_dict[key] = _mask_sensitive(event_dict[key], redactions)

    if redactions:
        existing = event_dict.get("redactions")
        if isinstance(existing, list):
            event_dict["redactions"] = sorted(set(existing) | redactions)
        else:
            event_dict["redactions"] = sorted(redactions)
    return event_dict

os.environ.setdefault("MINIAGENT_LOG_LEVEL", "INFO")
_thread_loggers: Dict[str, structlog.BoundLogger] = {}


def _ensure_event_key(_: logging.Logger, __: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee an `event` field even when logger.info('msg', ...) is used."""
    if "event" in event_dict:
        return event_dict
    message = event_dict.pop("message", None) or event_dict.pop("msg", None) or ""
    event_dict["event"] = message
    return event_dict


def configure_logging() -> structlog.BoundLogger:
    """Configure structlog with JSON output."""
    log_level = os.getenv("MINIAGENT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)

    root_logger = logging.getLogger("miniagent")
    root_logger.setLevel(level)
    root_logger.handlers = []

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    stream_handler.setLevel(level)
    root_logger.addHandler(stream_handler)

    structlog.configure_once(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", key="timestamp", utc=True),
            merge_contextvars,
            structlog.processors.add_log_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            _redact_sensitive,
            _ensure_event_key,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger("miniagent")


def get_thread_logger(thread_id: str, metadata: Dict[str, Any] | None = None) -> structlog.BoundLogger:
    if thread_id in _thread_loggers:
        return _thread_loggers[thread_id]

    logger = structlog.get_logger("miniagent")
    metadata = metadata or {}
    bound = logger.bind(thread_id=thread_id, **metadata)
    _thread_loggers[thread_id] = bound
    return bound


logger = configure_logging()
