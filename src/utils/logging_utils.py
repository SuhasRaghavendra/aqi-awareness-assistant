"""
src/utils/logging_utils.py
Configures structured logging for all pipeline stages.
Writes human-readable output to console and JSON Lines to logs/<stage>_<timestamp>.jsonl.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonLinesHandler(logging.Handler):
    """Appends each log record as a JSON line to a .jsonl file."""

    def __init__(self, log_path: Path) -> None:
        super().__init__()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = log_path

    def emit(self, record: logging.LogRecord) -> None:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "stage": getattr(record, "stage", "general"),
            "level": record.levelname,
            "message": self.format(record),
            "logger": record.name,
            "extra": {
                k: v
                for k, v in record.__dict__.items()
                if k not in logging.LogRecord.__dict__ and k not in {"msg", "args", "exc_info", "exc_text"}
            },
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def get_logger(
    stage: str,
    log_dir: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Return a logger for the given pipeline stage.

    Attaches:
    - StreamHandler (console, human-readable)
    - JsonLinesHandler (logs/<stage>_<timestamp>.jsonl)

    Args:
        stage: Pipeline stage name, e.g. 'discover_stations'.
        log_dir: Directory for log files. Defaults to project_root/logs/.
        level: Logging level. Defaults to INFO.

    Returns:
        Configured Logger instance.
    """
    if log_dir is None:
        # Resolve relative to the project root (three levels up from this file)
        log_dir = Path(__file__).resolve().parents[2] / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_file = log_dir / f"{stage}_{timestamp}.jsonl"

    logger = logging.getLogger(stage)
    logger.setLevel(level)

    if logger.handlers:
        # Already configured — avoid duplicate handlers on reimport
        return logger

    # Console handler — human-readable
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(fmt)

    # JSON Lines file handler
    json_handler = JsonLinesHandler(log_file)
    json_handler.setLevel(level)
    json_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(console_handler)
    logger.addHandler(json_handler)
    logger.propagate = False

    logger.info("Logger initialised", extra={"stage": stage, "log_file": str(log_file)})
    return logger
