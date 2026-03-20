"""
Structured logging configuration for MLOps observability.
Implements JSON-formatted logs with correlation ID support.
"""

import json
import logging
import sys
import uuid
from datetime import datetime


class CorrelationIDFilter(logging.Filter):
    """Injects a correlation ID into each log record for request tracing."""

    def __init__(self, correlation_id: str | None = None):
        super().__init__()
        self.correlation_id = correlation_id or str(uuid.uuid4())

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = self.correlation_id
        record.timestamp = datetime.utcnow().isoformat() + "Z"
        return True


class JSONFormatter(logging.Formatter):
    """Formats log records as structured JSON for observability pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": getattr(record, "timestamp", datetime.utcnow().isoformat() + "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        for key in ["model_version", "endpoint", "latency_ms", "user_id"]:
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return json.dumps(log_entry)


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: str | None = None,
    correlation_id: str | None = None,
) -> logging.Logger:
    """
    Configure and return a structured logger.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for file handler
        correlation_id: Optional fixed correlation ID for request-scoped logging

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if logger.handlers:
        return logger

    json_formatter = JSONFormatter()
    corr_filter = CorrelationIDFilter(correlation_id)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(json_formatter)
    console_handler.addFilter(corr_filter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(json_formatter)
        file_handler.addFilter(corr_filter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger
