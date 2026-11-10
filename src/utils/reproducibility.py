"""Deterministic timestamps for reproducible artifacts.

The risk card, the reference statistics and the champion ``MANIFEST.json`` all
carry a build timestamp, and their SHA-256 digests are recorded in the bundle
manifest. Using ``datetime.now()`` makes every rebuild produce different bytes,
so DVC reports the outputs as changed and the registry gains a spurious model
version even when no input changed.

Build timestamps are therefore derived from the build inputs instead:

1. ``SOURCE_DATE_EPOCH`` (the reproducible-builds standard) when set,
2. otherwise the committer date of ``HEAD``,
3. otherwise the current local time (exported source tree without Git).
"""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path


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
