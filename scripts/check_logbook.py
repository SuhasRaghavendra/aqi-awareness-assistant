#!/usr/bin/env python
"""
scripts/check_logbook.py
------------------------
Logbook completeness guard (Section 4.3, item 6).

Fails (exit code 1) if any commit since the last logbook entry has touched
files under ``src/``, ``params.yaml``, ``dvc.yaml``, or ``configs/`` without
a corresponding new entry in ``docs/logbook/``.

Run in:
  - ``make test``  (pre-merge quality gate)
  - ``.pre-commit-config.yaml`` pre-push hook

Exit codes:
  0 -- logbook is up to date (or no tracked files have changed)
  1 -- uncommitted code/config changes exist without a logbook entry
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOGBOOK_DIR = _REPO_ROOT / "docs" / "logbook"

# Files/directories that MUST have a logbook entry if changed
_TRACKED_PREFIXES = (
    "src/",
    "configs/",
    "params.yaml",
    "dvc.yaml",
    "scripts/",
)


def _run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, cwd=_REPO_ROOT, timeout=30,
        )
        return r.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _latest_logbook_commit() -> str | None:
    """Return the git commit hash of the most-recently added logbook entry, or None."""
    entries = sorted(_LOGBOOK_DIR.glob("[0-9][0-9][0-9]-*.md"))
    if not entries:
        return None
    latest = entries[-1]
    rel = latest.relative_to(_REPO_ROOT)
    out = _run(["git", "log", "-1", "--pretty=%H", "--", str(rel)])
    return out or None


def _changed_files_since(commit: str | None) -> list[str]:
    """List files changed since *commit* (or all tracked files if commit is None)."""
    if commit is None:
        out = _run(["git", "status", "--porcelain"])
    else:
        out = _run(["git", "diff", "--name-only", commit, "HEAD"])
    return [line.strip().lstrip("AMDRCU? ") for line in out.splitlines() if line.strip()]


def _is_tracked(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in _TRACKED_PREFIXES)


def _count_entries() -> int:
    return len(list(_LOGBOOK_DIR.glob("[0-9][0-9][0-9]-*.md")))


def main() -> None:
    n_entries = _count_entries()
    last_commit = _latest_logbook_commit()

    changed = _changed_files_since(last_commit)
    tracked_changed = [f for f in changed if _is_tracked(f)]

    if not tracked_changed:
        print(f"[OK] Logbook check passed - {n_entries} entries; no tracked files changed since last entry.")
        sys.exit(0)

    if last_commit:
        new_entries_out = _run(
            ["git", "log", "--pretty=%H", f"{last_commit}..HEAD",
             "--", "docs/logbook/[0-9][0-9][0-9]-*.md"]
        )
        new_entries = [line for line in new_entries_out.splitlines() if line.strip()]
    else:
        new_entries = list(_LOGBOOK_DIR.glob("[0-9][0-9][0-9]-*.md"))

    if new_entries:
        print(f"[OK] Logbook check passed - {n_entries} entries; new entry covers the changes.")
        sys.exit(0)

    # FAIL
    print("[FAIL] Logbook check FAILED")
    print(f"   {len(tracked_changed)} tracked file(s) changed since last logbook entry:")
    for f in tracked_changed[:20]:
        print(f"     * {f}")
    if len(tracked_changed) > 20:
        print(f"     ... and {len(tracked_changed) - 20} more")
    print()
    print("   Create a logbook entry with:")
    print("     python scripts/log_step.py --step NNN --title '...' --phase N")
    print("   or:")
    print("     make step-close STEP=NNN TITLE='...' PHASE=N")
    sys.exit(1)


if __name__ == "__main__":
    main()
