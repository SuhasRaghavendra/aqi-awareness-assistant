"""
src/utils/logging_utils.py
--------------------------
Configures project-wide logging:
  - Human-readable output to the console (coloured by level).
  - Machine-readable JSON Lines to logs/<stage>_<timestamp>.jsonl.

Every src/ script should call ``get_logger(__name__)`` (or
``setup_logging(stage)`` once at the entry-point) before doing any work.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── ANSI colour codes for console output ──────────────────────────────────────
_COLOURS = {
    "DEBUG": "\033[36m",    # cyan
    "INFO": "\033[32m",     # green
    "WARNING": "\033[33m",  # yellow
    "ERROR": "\033[31m",    # red
    "CRITICAL": "\033[35m", # magenta
    "RESET": "\033[0m",
}


class _ColouredFormatter(logging.Formatter):
    """Console formatter with ANSI colour by level."""

    FMT = "%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        colour = _COLOURS.get(record.levelname, "")
        reset = _COLOURS["RESET"]
        record.levelname = f"{colour}{record.levelname}{reset}"
        return super().format(record)


class _JsonlHandler(logging.Handler):
    """Appends one JSON object per log record to a .jsonl file."""

    def __init__(self, path: Path, stage: str, extra: dict[str, Any] | None = None) -> None:
        super().__init__()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")
        self._stage = stage
        self._extra = extra or {}

    def emit(self, record: logging.LogRecord) -> None:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "stage": self._stage,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        payload.update(self._extra)
        # Merge any extra keys set via logger.info("msg", extra={"key": val})
        for key in vars(record):
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                payload[key] = getattr(record, key)
        try:
            self._fh.write(json.dumps(payload, default=str) + "\n")
            self._fh.flush()
        except Exception:  # noqa: BLE001
            self.handleError(record)

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            super().close()


# ── Public API ────────────────────────────────────────────────────────────────

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_configured_stages: set[str] = set()


def setup_logging(
    stage: str,
    *,
    log_dir: Path | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    extra: dict[str, Any] | None = None,
) -> logging.Logger:
    """Configure logging for *stage* and return the root-level logger for it.

    Call this **once** at the entry point of each script/module.  Subsequent
    calls with the same stage name are no-ops (idempotent).

    Parameters
    ----------
    stage:
        Short identifier for the pipeline stage (e.g. ``"download_openaq"``).
        Used as the JSONL filename prefix and included in every log record.
    log_dir:
        Directory to write the JSONL file into.  Defaults to ``<repo>/logs/``.
    console_level:
        Minimum level for console output (default ``INFO``).
    file_level:
        Minimum level for JSONL file output (default ``DEBUG``).
    extra:
        Static key-value pairs to include in every JSON record (e.g.
        ``{"git_commit": "abc1234"}``).

    Returns
    -------
    logging.Logger
        A logger named ``stage``.
    """
    if stage in _configured_stages:
        return logging.getLogger(stage)

    logger = logging.getLogger(stage)
    logger.setLevel(min(console_level, file_level))
    logger.propagate = False

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    fmt = _ColouredFormatter(_ColouredFormatter.FMT, datefmt=_ColouredFormatter.DATE_FMT)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # JSONL file handler
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = (log_dir or _LOG_DIR) / f"{stage}_{ts}.jsonl"
    fh = _JsonlHandler(log_path, stage=stage, extra=extra)
    fh.setLevel(file_level)
    logger.addHandler(fh)

    _configured_stages.add(stage)
    logger.debug("Logging initialised for stage '%s' → %s", stage, log_path)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger for *name* (e.g. ``__name__``).

    If the root logger for the enclosing stage has not been configured yet,
    this falls back to a basic console-only logger so callers never crash.
    """
    logger = logging.getLogger(name)
    if not logger.handlers and not logging.getLogger().handlers:
        # Minimal fallback so imports don't silently lose messages
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s",
            stream=sys.stdout,
        )
    return logger
