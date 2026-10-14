"""
src/utils/stage_stats.py
------------------------
Context manager ``track_stage`` that automatically captures and persists
per-stage statistics for every data-transforming pipeline step.

Usage::

    from src.utils.stage_stats import track_stage

    with track_stage("clean", inputs=["data/interim/station_day.parquet"],
                              outputs=["data/interim/station_day_clean.parquet"]) as st:
        df = pd.read_parquet(st.inputs[0])
        st.record_input_frame(df)
        # ... do work ...
        df_out = clean(df)
        st.drop(reason="missing AQI target", n=100)
        st.record_output_frame(df_out)

On exit the stats JSON is written to ``reports/metrics/<stage>_stats.json``
and a summary section is appended to ``docs/data_change_log.md``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Repo root (two levels up from this file: src/utils/ → src/ → repo/)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_METRICS_DIR = _REPO_ROOT / "reports" / "metrics"
_CHANGE_LOG = _REPO_ROOT / "docs" / "data_change_log.md"


# ── helpers ───────────────────────────────────────────────────────────────────


def _git_commit() -> str:
    """Return the current HEAD commit hash (short), or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        return result.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _params_hash(params_path: Path | None = None) -> str:
    """SHA-256 (truncated to 12 hex chars) of params.yaml, or 'unknown'."""
    p = params_path or (_REPO_ROOT / "params.yaml")
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    except FileNotFoundError:
        return "unknown"


def _file_info(path: str | Path) -> dict[str, Any]:
    """Return size and, for parquet/csv, row+column counts."""
    p = Path(path)
    info: dict[str, Any] = {"path": str(p), "exists": p.exists()}
    if not p.exists():
        return info
    info["size_bytes"] = p.stat().st_size
    ext = p.suffix.lower()
    try:
        if ext == ".parquet":
            df = pd.read_parquet(p)
            info["rows"] = len(df)
            info["columns"] = list(df.columns)
            info["n_columns"] = len(df.columns)
        elif ext == ".csv":
            df = pd.read_csv(p, nrows=0)
            # count rows cheaply
            with open(p, encoding="utf-8", errors="replace") as fh:
                info["rows"] = sum(1 for _ in fh) - 1
            info["columns"] = list(df.columns)
            info["n_columns"] = len(df.columns)
    except Exception as exc:  # noqa: BLE001
        info["read_error"] = str(exc)
    return info


def _null_pct(df: pd.DataFrame) -> dict[str, float]:
    """Return {column: null_percentage} rounded to 2 dp."""
    total = len(df)
    if total == 0:
        return {}
    return {col: round(df[col].isna().sum() / total * 100, 2) for col in df.columns}


# ── core data-class ───────────────────────────────────────────────────────────


@dataclass
class StageStats:
    """Accumulates statistics for a single pipeline stage run."""

    stage: str
    inputs: list[str]
    outputs: list[str]

    # populated automatically
    start_time: str = field(default="", init=False)
    end_time: str = field(default="", init=False)
    duration_s: float = field(default=0.0, init=False)
    git_commit: str = field(default="", init=False)
    params_hash: str = field(default="", init=False)

    # frame snapshots
    _input_frame_stats: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _output_frame_stats: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    # drop log
    _drops: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    # column diff
    _cols_before: list[str] = field(default_factory=list, init=False, repr=False)
    _cols_after: list[str] = field(default_factory=list, init=False, repr=False)

    def drop(self, reason: str, n: int) -> None:
        """Record that *n* rows were dropped for *reason*."""
        self._drops.append({"reason": reason, "n": n})
        logger.debug("drop(reason=%r, n=%d)", reason, n)

    def record_input_frame(self, df: pd.DataFrame) -> None:
        """Snapshot statistics of the input DataFrame."""
        self._cols_before = list(df.columns)
        self._input_frame_stats = {
            "rows": len(df),
            "n_columns": len(df.columns),
            "null_pct": _null_pct(df),
        }
        logger.debug("Input frame: %d rows × %d cols", len(df), len(df.columns))

    def record_output_frame(self, df: pd.DataFrame) -> None:
        """Snapshot statistics of the output DataFrame."""
        self._cols_after = list(df.columns)
        self._output_frame_stats = {
            "rows": len(df),
            "n_columns": len(df.columns),
            "null_pct": _null_pct(df),
        }
        logger.debug("Output frame: %d rows × %d cols", len(df), len(df.columns))

    # ── computed properties ───────────────────────────────────────────────────

    @property
    def cols_added(self) -> list[str]:
        before = set(self._cols_before)
        after = set(self._cols_after)
        return sorted(after - before)

    @property
    def cols_removed(self) -> list[str]:
        before = set(self._cols_before)
        after = set(self._cols_after)
        return sorted(before - after)

    @property
    def rows_in(self) -> int:
        return self._input_frame_stats.get("rows", 0)

    @property
    def rows_out(self) -> int:
        return self._output_frame_stats.get("rows", 0)

    @property
    def total_dropped(self) -> int:
        return sum(d["n"] for d in self._drops)

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_s": round(self.duration_s, 3),
            "git_commit": self.git_commit,
            "params_hash": self.params_hash,
            "inputs": [_file_info(p) for p in self.inputs],
            "outputs": [_file_info(p) for p in self.outputs],
            "frame_stats": {
                "input": self._input_frame_stats,
                "output": self._output_frame_stats,
            },
            "drops": self._drops,
            "total_dropped": self.total_dropped,
            "cols_added": self.cols_added,
            "cols_removed": self.cols_removed,
        }


# ── context manager ───────────────────────────────────────────────────────────


@contextmanager
def track_stage(
    stage: str,
    *,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
) -> Generator[StageStats, None, None]:
    """Context manager that times a stage and saves stats on exit.

    Parameters
    ----------
    stage:
        Short name matching the DVC stage (e.g. ``"clean"``).
    inputs:
        List of input file paths (resolved relative to cwd).
    outputs:
        List of output file paths (resolved relative to cwd).

    Yields
    ------
    StageStats
        Accumulator object; call ``.drop()``, ``.record_input_frame()`` etc.
    """
    st = StageStats(
        stage=stage,
        inputs=[str(p) for p in (inputs or [])],
        outputs=[str(p) for p in (outputs or [])],
    )
    st.git_commit = _git_commit()
    st.params_hash = _params_hash()
    st.start_time = datetime.now(tz=timezone.utc).isoformat()
    t0 = time.perf_counter()

    logger.info("Stage '%s' started (commit=%s, params_hash=%s)", stage, st.git_commit, st.params_hash)

    try:
        yield st
    finally:
        st.end_time = datetime.now(tz=timezone.utc).isoformat()
        st.duration_s = time.perf_counter() - t0
        logger.info(
            "Stage '%s' finished in %.1fs  rows: %d -> %d  dropped: %d",
            stage,
            st.duration_s,
            st.rows_in,
            st.rows_out,
            st.total_dropped,
        )

        # ── Write JSON metrics ────────────────────────────────────────────────
        _METRICS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path = _METRICS_DIR / f"{stage}_stats.json"
        data = st.to_dict()
        metrics_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info("Stage stats written -> %s", metrics_path)

        # ── Append to data_change_log.md ─────────────────────────────────────
        _append_change_log(st)


def _append_change_log(st: StageStats) -> None:
    """Append a Markdown section to docs/data_change_log.md."""
    _CHANGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    drop_lines = "\n".join(
        f"  - `{d['reason']}`: {d['n']:,} rows" for d in st._drops
    ) or "  - (none)"

    section = f"""
## {st.stage} — {st.end_time[:10]}

| Field | Value |
|---|---|
| Stage | `{st.stage}` |
| Start | {st.start_time} |
| End | {st.end_time} |
| Duration | {st.duration_s:.1f} s |
| Git commit | `{st.git_commit}` |
| Params hash | `{st.params_hash}` |
| Rows in | {st.rows_in:,} |
| Rows out | {st.rows_out:,} |
| Total dropped | {st.total_dropped:,} |

**Rows dropped by reason:**
{drop_lines}

**Columns added:** {", ".join(f"`{c}`" for c in st.cols_added) or "none"}
**Columns removed:** {", ".join(f"`{c}`" for c in st.cols_removed) or "none"}

---
"""
    with _CHANGE_LOG.open("a", encoding="utf-8") as fh:
        fh.write(section)
    logger.info("data_change_log.md updated -> %s", _CHANGE_LOG)
