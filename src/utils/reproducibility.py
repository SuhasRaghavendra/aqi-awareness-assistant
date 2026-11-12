"""Deterministic timestamps for reproducible artifacts.

The risk card, the reference statistics and the champion ``MANIFEST.json`` all
carry a build timestamp, and their SHA-256 digests are recorded in the bundle
manifest. Wall-clock time (``datetime.now()``) or the ``HEAD`` committer date
makes every rebuild produce different bytes, so DVC reports the outputs as
changed and the registry gains a spurious model version even when no input
changed. Committing the artifact then moves ``HEAD`` and breaks it again.

Artifact timestamps are therefore derived from the data itself: the newest
observation in the split the artifact describes. Rebuilding the same frozen
data yields the same bytes on any machine, at any commit.

``build_timestamp`` (``SOURCE_DATE_EPOCH`` -> ``HEAD`` -> now) is retained for
logs and human-readable provenance, but it is not used for hashed artifacts.
"""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def git_commit_timestamp(root: Path) -> str | None:
    """Return the ISO-8601 committer date of HEAD, or ``None`` without Git."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cI"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def build_timestamp(root: Path) -> str:
    """Return a reproducible build timestamp for artifacts built from *root*.

    Repeated builds of the same commit yield the same string, so the manifest
    and every digest recorded in it are stable.
    """
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is not None:
        return datetime.fromtimestamp(int(epoch), tz=UTC).isoformat()
    return git_commit_timestamp(root) or datetime.now().isoformat()


def frame_timestamp(frame: Any) -> str:
    """Return a reproducible ISO-8601 timestamp from a split's newest observation.

    The value is the last ``date`` in the frame at UTC midnight, so it depends
    only on the frozen data and not on when (or at which commit) the artifact
    was built. This is what hashed artifacts such as the risk card, the
    reference statistics, ``feature_list.json`` and ``MANIFEST.json`` record.
    """
    if frame is None or "date" not in getattr(frame, "columns", []):
        raise ValueError("frame_timestamp requires a frame with a 'date' column")
    dates = pd.to_datetime(pd.Series(frame["date"]))
    if dates.empty or dates.isna().all():
        raise ValueError("frame_timestamp requires at least one valid date")
    return f"{dates.max().strftime('%Y-%m-%d')}T00:00:00+00:00"


def path_timestamp(path: Path) -> str:
    """Return :func:`frame_timestamp` for a parquet split at *path*."""
    frame = pd.read_parquet(path, columns=["date"])
    return frame_timestamp(frame)
