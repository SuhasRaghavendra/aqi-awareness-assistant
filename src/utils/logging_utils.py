"""Project-wide structured logging and JSONL audit logging."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


class JsonFormatter(logging.Formatter):
    """Format each stdout record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
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
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        try:
            self._file.write(json.dumps(payload, default=str) + "\n")
            self._file.flush()
        except Exception:  # noqa: BLE001
            self.handleError(record)

    def close(self) -> None:
        try:
            self._file.close()
        finally:
            super().close()


def configure_json_logging(level: int = logging.INFO) -> None:
    """Install one JSON stdout handler on the root logger (idempotent)."""
    root = logging.getLogger()
    if getattr(root, "_aqi_json_configured", False):
        return
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    root._aqi_json_configured = True  # type: ignore[attr-defined]


def _resolve_audit_log_path() -> Path | None:
    """Return the configured JSONL audit log path, or None if explicitly disabled."""
    raw = os.environ.get("PREDICTION_LOG_PATH")
    if raw is None:
        return _LOG_DIR / "predictions_audit.jsonl"
    raw = raw.strip()
    return Path(raw) if raw else None


def get_audit_logger() -> logging.Logger:
    """Return the dedicated structured audit logger (stdout + optional JSONL file)."""
    logger = logging.getLogger("aqi.audit")
    target_path = _resolve_audit_log_path()
    current_path = getattr(logger, "_aqi_audit_path", "__unset__")

    if current_path == target_path and getattr(logger, "_aqi_audit_configured", False):
        return logger

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if target_path is not None:
        try:
            handler = JsonlAuditHandler(target_path)
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
        except OSError:
            # Read-only or ephemeral filesystem: fall back cleanly to stdout JSON
            pass

    logger.setLevel(logging.INFO)
    logger.propagate = True
    logger._aqi_audit_path = target_path  # type: ignore[attr-defined]
    logger._aqi_audit_configured = True  # type: ignore[attr-defined]
    return logger


def log_audit_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Write one structured audit event to stdout JSON and the JSONL audit stream."""
    if not logging.getLogger().handlers:
        configure_json_logging()
    logger = get_audit_logger()
    logger.log(level, event, extra={"event": event, **fields})


def setup_logging(
    stage: str = "app",
    *,
    log_dir: Path | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    extra: dict[str, Any] | None = None,
) -> logging.Logger:
    """Configure console + JSONL file logging for pipeline stages."""
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

    logger._aqi_stage_configured = True  # type: ignore[attr-defined]
    if extra:
        logger.debug("Logging metadata configured", extra=extra)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a named project logger."""
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        configure_json_logging()
    return logger
