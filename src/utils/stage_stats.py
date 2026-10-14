"""
src/utils/stage_stats.py
Context manager that tracks pipeline stage statistics.
Records input/output file sizes, row/column counts, drop reasons, and emits a JSON report.
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


def _get_git_commit() -> str:
    """Return the current HEAD commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _file_stats(path: Path) -> dict[str, Any]:
    """Return size, row count, and column count for a file."""
    stats: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return stats
    stats["size_bytes"] = path.stat().st_size
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            df = pd.read_parquet(path)
            stats["rows"] = len(df)
            stats["columns"] = list(df.columns)
            stats["n_columns"] = len(df.columns)
        elif suffix in {".csv", ".gz"}:
            df = pd.read_csv(path, nrows=0)
            stats["columns"] = list(df.columns)
            stats["n_columns"] = len(df.columns)
            # Count rows without loading everything into memory
            with open(path, "rb") as f:
                stats["rows"] = sum(1 for _ in f) - 1  # subtract header
        elif suffix == ".jsonl":
            with open(path, encoding="utf-8") as f:
                stats["rows"] = sum(1 for _ in f)
    except Exception as exc:
        stats["read_error"] = str(exc)
    return stats


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while data := f.read(chunk):
                h.update(data)
        return h.hexdigest()
    except OSError:
        return "error"


@dataclass
class StageTracker:
    """Accumulates statistics for a pipeline stage during its run."""

    stage: str
    _drops: list[dict[str, Any]] = field(default_factory=list)
    _imputations: list[dict[str, Any]] = field(default_factory=list)
    _notes: list[str] = field(default_factory=list)

    def drop(self, reason: str, n: int) -> None:
        """Record rows dropped with the given reason."""
        self._drops.append({"reason": reason, "n": n})

    def impute(self, column: str, n: int, method: str = "") -> None:
        """Record imputation counts."""
        self._imputations.append({"column": column, "n": n, "method": method})

    def note(self, message: str) -> None:
        """Record a free-text note."""
        self._notes.append(message)


@contextmanager
def track_stage(
    stage: str,
    inputs: list[Path] | None = None,
    outputs: list[Path] | None = None,
    metrics_dir: Path | None = None,
) -> Generator[StageTracker, None, None]:
    """
    Context manager that records stage statistics and writes a JSON report.

    Usage::

        with track_stage("clean", inputs=[raw], outputs=[clean]) as st:
            # ... do work ...
            st.drop(reason="negative PM2.5", n=42)

    Args:
        stage: Pipeline stage name.
        inputs: List of input file paths to stat.
        outputs: List of output file paths to stat (after stage completes).
        metrics_dir: Directory for the JSON report. Defaults to reports/metrics/.

    Yields:
        StageTracker instance for recording drops, imputations, and notes.
    """
    if metrics_dir is None:
        metrics_dir = Path(__file__).resolve().parents[2] / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    git_commit = _get_git_commit()
    start_time = time.time()
    start_iso = datetime.now(tz=timezone.utc).isoformat()

    # Stat inputs before stage runs
    input_stats = [_file_stats(p) for p in (inputs or [])]

    tracker = StageTracker(stage=stage)
    try:
        yield tracker
    finally:
        end_time = time.time()
        end_iso = datetime.now(tz=timezone.utc).isoformat()

        # Stat outputs after stage completes
        output_stats = [_file_stats(p) for p in (outputs or [])]

        report: dict[str, Any] = {
            "stage": stage,
            "git_commit": git_commit,
            "start_time": start_iso,
            "end_time": end_iso,
            "duration_seconds": round(end_time - start_time, 2),
            "inputs": input_stats,
            "outputs": output_stats,
            "drops": tracker._drops,
            "total_dropped": sum(d["n"] for d in tracker._drops),
            "imputations": tracker._imputations,
            "notes": tracker._notes,
        }

        out_path = metrics_dir / f"{stage}_stats.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Print summary to stdout
        total_in = sum(s.get("rows", 0) for s in input_stats)
        total_out = sum(s.get("rows", 0) for s in output_stats)
        print(
            f"\n[stage_stats] {stage} | "
            f"duration={report['duration_seconds']}s | "
            f"rows in={total_in} out={total_out} | "
            f"dropped={report['total_dropped']} | "
            f"commit={git_commit}"
        )
        print(f"[stage_stats] Report written to: {out_path}")

        # Append to data_change_log.md
        _append_data_change_log(stage, report)


def _append_data_change_log(stage: str, report: dict[str, Any]) -> None:
    """Append a summary section to docs/data_change_log.md."""
    log_path = Path(__file__).resolve().parents[2] / "docs" / "data_change_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    total_in = sum(s.get("rows", 0) for s in report.get("inputs", []))
    total_out = sum(s.get("rows", 0) for s in report.get("outputs", []))
    drops_summary = "; ".join(
        f"{d['reason']}: {d['n']}" for d in report.get("drops", [])
    ) or "none"
    imputations_summary = "; ".join(
        f"{i['column']} ({i['method']}): {i['n']}" for i in report.get("imputations", [])
    ) or "none"

    entry = (
        f"\n## {report['stage']} — {report['end_time'][:10]}\n"
        f"- **Stage:** `{stage}`\n"
        f"- **Duration:** {report['duration_seconds']}s\n"
        f"- **Rows in → out:** {total_in} → {total_out}\n"
        f"- **Rows dropped (by reason):** {drops_summary}\n"
        f"- **Imputations:** {imputations_summary}\n"
        f"- **Notes:** {'; '.join(report.get('notes', [])) or 'none'}\n"
        f"- **Git commit:** `{report['git_commit']}`\n"
    )

    with log_path.open("a", encoding="utf-8") as f:
        f.write(entry)
