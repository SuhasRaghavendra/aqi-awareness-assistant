#!/usr/bin/env python3
"""
scripts/check_logbook.py
Fails if commits since the last logbook entry changed src/, params.yaml, dvc.yaml or configs/
without a new logbook entry. Run in pre-push hook and `make test`.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGBOOK_DIR = ROOT / "docs" / "logbook"

WATCHED_PATTERNS = [
    r"^src/",
    r"^params\.yaml$",
    r"^dvc\.yaml$",
    r"^configs/",
]


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:
        return ""


def get_last_logbook_commit() -> str:
    """Return the commit hash of the most recent logbook entry commit."""
    entries = sorted(LOGBOOK_DIR.glob("[0-9]*.md"))
    if not entries:
        return ""
    # Get the commit that last touched the most recent entry
    latest = entries[-1]
    return run(["git", "log", "-1", "--format=%H", "--", str(latest.relative_to(ROOT))])


def get_changed_files_since(commit: str) -> list[str]:
    """List files changed since the given commit."""
    if not commit:
        return []
    out = run(["git", "diff", "--name-only", commit, "HEAD"])
    return [line.strip() for line in out.splitlines() if line.strip()]


def is_watched(path: str) -> bool:
    return any(re.match(p, path) for p in WATCHED_PATTERNS)


def main() -> None:
    last_commit = get_last_logbook_commit()
    if not last_commit:
        print("ℹ️  No logbook entries found yet — skipping check.")
        sys.exit(0)

    changed = get_changed_files_since(last_commit)
    watched_changed = [f for f in changed if is_watched(f)]

    if watched_changed:
        print(
            f"❌ check_logbook: The following watched files changed since the last logbook entry "
            f"(commit {last_commit[:8]}) but no new logbook entry was found:\n"
        )
        for f in watched_changed:
            print(f"   {f}")
        print(
            "\nRun: python scripts/log_step.py --step <NNN> --title '<title>' --phase <n>"
            "\nThen commit the logbook entry before pushing."
        )
        sys.exit(1)
    else:
        print("✅ check_logbook: All watched changes are covered by logbook entries.")
        sys.exit(0)


if __name__ == "__main__":
    main()
