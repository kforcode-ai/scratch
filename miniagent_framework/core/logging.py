"""Structured logging utilities for MiniAgent."""
import logging
import os
import sys
from typing import Any, Dict

import structlog

os.environ.setdefault("MINIAGENT_LOG_LEVEL", "INFO")
_thread_loggers: Dict[str, structlog.BoundLogger] = {}


def configure_logging() -> structlog.BoundLogger:
    """Configure structlog with JSON output."""
    log_level = os.getenv("MINIAGENT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)

    root_logger = logging.getLogger("miniagent")
    root_logger.setLevel(level)
    root_logger.handlers = []

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    stream_handler.setLevel(level)
    root_logger.addHandler(stream_handler)

    structlog.configure_once(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
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
