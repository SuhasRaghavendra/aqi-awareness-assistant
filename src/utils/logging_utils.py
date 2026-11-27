
"""Project-wide structured logging and JSONL audit logging."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
)

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


class JsonFormatter(logging.Formatter):
    """Format each stdout record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=UTC
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in vars(record).items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class JsonlAuditHandler(logging.Handler):
    """Append structured audit events to a JSON Lines file."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=UTC
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in vars(record).items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        try:
            self._file.write(json.dumps(payload, default=str) + "\n")
            self._file.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            self._file.close()
        finally:
            super().close()


def configure_json_logging(level: int = logging.INFO) -> None:
    """Configure JSON-formatted stdout logging."""

    root = logging.getLogger()

    if getattr(root, "_aqi_json_configured", False):
        return

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    root._aqi_json_configured = True


def get_audit_logger() -> logging.Logger:
    """Return the dedicated JSONL audit logger."""

    logger = logging.getLogger("aqi.audit")

    if getattr(logger, "_aqi_audit_configured", False):
        return logger

    path = Path(
        os.environ.get(
            "PREDICTION_LOG_PATH",
            str(_LOG_DIR / "predictions_audit.jsonl"),
        )
    )

    handler = JsonlAuditHandler(path)
    handler.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = True
    logger._aqi_audit_configured = True

    return logger


def log_audit_event(event: str, **fields: Any) -> None:
    """Write one structured event to stdout and the audit stream."""

    logger = get_audit_logger()
    logger.info(event, extra={"event": event, **fields})


def setup_logging(
    stage: str = "app",
    *,
    log_dir: Path | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    extra: dict[str, Any] | None = None,
) -> logging.Logger:
    """Maintain compatibility with existing logging calls."""

    logger = logging.getLogger(stage)

    if getattr(logger, "_aqi_stage_configured", False):
        return logger

    logger.setLevel(min(console_level, file_level))
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(JsonFormatter())
    logger.addHandler(console)

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    path = (log_dir or _LOG_DIR) / f"{stage}_{timestamp}.jsonl"

    handler = JsonlAuditHandler(path)
    handler.setLevel(file_level)
    logger.addHandler(handler)

    logger._aqi_stage_configured = True

    if extra:
        logger.debug("Logging metadata configured", extra=extra)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a named project logger."""

    logger = logging.getLogger(name)

    if not logging.getLogger().handlers:
        configure_json_logging()

    return logger
