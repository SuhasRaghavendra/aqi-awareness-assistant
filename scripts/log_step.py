#!/usr/bin/env python
"""
scripts/log_step.py
-------------------
Step-Close Protocol helper (Section 4.3, item 4).

Usage::

    python scripts/log_step.py --step 001 --title "Repo scaffold and logging" --phase 2
    python scripts/log_step.py --step 002 --title "Station discovery" --phase 1 --mlflow-run abc123

What it does:
  1. Reads git status, git diff --stat, dvc status, dvc diff output.
  2. Reads the latest *_stats.json files from reports/metrics/.
  3. Creates docs/logbook/NNN-<slug>.md from the template, pre-filled.
  4. Writes reports/step_reports/NNN.json (machine-readable snapshot).
  5. Appends a line to docs/CHANGELOG.md under ## [Unreleased].
  6. Prints the filled entry to the terminal.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── repo root ─────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOGBOOK_DIR = _REPO_ROOT / "docs" / "logbook"
_STEP_REPORTS_DIR = _REPO_ROOT / "reports" / "step_reports"
_METRICS_DIR = _REPO_ROOT / "reports" / "metrics"
_CHANGELOG = _REPO_ROOT / "docs" / "CHANGELOG.md"


# ── git helpers ───────────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: Path | None = None) -> str:
    """Run a subprocess command and return its stdout (or error message)."""
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd or _REPO_ROOT,
            timeout=30,
        )
        return (r.stdout + r.stderr).strip()
    except FileNotFoundError:
        return f"(command not found: {cmd[0]})"
    except subprocess.TimeoutExpired:
        return "(timed out)"
    except Exception as exc:  # noqa: BLE001
        return f"(error: {exc})"


def _git_branch() -> str:
    return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def _git_commit() -> str:
    return _run(["git", "rev-parse", "HEAD"])


def _git_status() -> str:
    return _run(["git", "status", "--short"])


def _git_diff_stat() -> str:
    out = _run(["git", "diff", "--stat", "HEAD~1"])
    if "unknown revision" in out or not out:
        out = _run(["git", "diff", "--stat", "--cached"])
    if not out:
        out = _run(["git", "diff", "--stat"])
    return out or "(no diff — first commit or nothing staged)"


def _dvc_status() -> str:
    return _run(["dvc", "status"])


def _dvc_diff() -> str:
    return _run(["dvc", "diff"])


# ── metrics helpers ────────────────────────────────────────────────────────────

def _load_latest_metrics() -> dict[str, dict]:
    """Load all *_stats.json files from reports/metrics/."""
    result: dict[str, dict] = {}
    if not _METRICS_DIR.exists():
        return result
    for p in sorted(_METRICS_DIR.glob("*_stats.json")):
        try:
            result[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            result[p.stem] = {"error": "could not parse"}
    return result


def _summarise_metrics(metrics: dict[str, dict]) -> str:
    if not metrics:
        return "_No stage stats files found yet._"
    lines = []
    for name, data in metrics.items():
        rows_in = data.get("frame_stats", {}).get("input", {}).get("rows", "?")
        rows_out = data.get("frame_stats", {}).get("output", {}).get("rows", "?")
        dropped = data.get("total_dropped", "?")
        dur = data.get("duration_s", "?")
        lines.append(f"- **{name}**: rows {rows_in} -> {rows_out}  |  dropped {dropped}  |  {dur}s")
    return "\n".join(lines)


def _list_changed_files() -> dict[str, list[str]]:
    """Group changed files by type from git status."""
    status = _run(["git", "status", "--porcelain"])
    groups: dict[str, list[str]] = {
        "code": [], "config": [], "docs": [], "data": [], "tests": [], "other": [],
    }
    for line in status.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if path.startswith("src/") or path.startswith("scripts/"):
            groups["code"].append(path)
        elif path.startswith("configs/") or path in ("params.yaml", "dvc.yaml", "dvc.lock"):
            groups["config"].append(path)
        elif path.startswith("docs/") or path.startswith("reports/"):
            groups["docs"].append(path)
        elif path.startswith("data/") or path.startswith("models/"):
            groups["data"].append(path)
        elif path.startswith("tests/"):
            groups["tests"].append(path)
        else:
            groups["other"].append(path)
    return {k: v for k, v in groups.items() if v}


def _changed_files_table(groups: dict[str, list[str]]) -> str:
    if not groups:
        return "_No uncommitted changes detected._"
    rows = []
    for group, files in groups.items():
        for f in files:
            rows.append(f"| {group} | `{f}` | — |")
    header = "| Type | Path | Change summary |\n|---|---|---|"
    return header + "\n" + "\n".join(rows)


# ── slug helper ───────────────────────────────────────────────────────────────

def _slugify(title: str) -> str:
    s = title.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")[:60]


# ── CHANGELOG helper ──────────────────────────────────────────────────────────

def _append_changelog(step: str, title: str, phase: int, entry_path: Path) -> None:
    _CHANGELOG.parent.mkdir(parents=True, exist_ok=True)
    if not _CHANGELOG.exists():
        _CHANGELOG.write_text(
            "# Changelog\n\nAll notable changes to this project will be documented here.\n"
            "Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).\n\n"
            "## [Unreleased]\n\n",
            encoding="utf-8",
        )
    content = _CHANGELOG.read_text(encoding="utf-8")
    today = datetime.now().strftime("%Y-%m-%d")
    new_line = f"- **Step {step}** (Phase {phase}): {title} — [{entry_path.name}](logbook/{entry_path.name}) — {today}\n"
    if "## [Unreleased]" in content:
        content = content.replace("## [Unreleased]\n", f"## [Unreleased]\n\n### Added\n{new_line}", 1)
    else:
        content += f"\n## [Unreleased]\n\n### Added\n{new_line}"
    _CHANGELOG.write_text(content, encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Step-Close Protocol: create logbook entry + step report")
    parser.add_argument("--step", required=True, help="Zero-padded step number, e.g. 001")
    parser.add_argument("--title", required=True, help="Short step title")
    parser.add_argument("--phase", type=int, default=2, help="Phase number (1–9)")
    parser.add_argument("--mlflow-run", default="", help="MLflow run ID(s), if any")
    args = parser.parse_args()

    step = args.step.zfill(3)
    title = args.title
    phase = args.phase
    mlflow_run = args.mlflow_run or "N/A"

    now_utc = datetime.now(tz=timezone.utc)
    now_ist = now_utc.strftime("%Y-%m-%dT%H:%M:%S+05:30")

    # Gather information
    branch = _git_branch()
    commit = _git_commit()
    git_status = _git_status()
    git_diff_stat = _git_diff_stat()
    dvc_status = _dvc_status()
    dvc_diff = _dvc_diff()
    metrics = _load_latest_metrics()
    metrics_summary = _summarise_metrics(metrics)
    changed_groups = _list_changed_files()
    changed_table = _changed_files_table(changed_groups)

    slug = _slugify(title)
    entry_filename = f"{step}-{slug}.md"
    entry_path = _LOGBOOK_DIR / entry_filename

    # ── Build the logbook entry ────────────────────────────────────────────────
    entry = f"""# Step {step} — {title}
- **Date/time:** {now_ist}
- **Author:** <student-A>      **Reviewer:** <student-B>
- **Phase:** {phase}           **Branch / PR:** `{branch}` / #<n>
- **Git commit(s):** `{commit}`      **Data version:** <git-tag> / <dvc-md5>
- **MLflow run IDs:** {mlflow_run}

## 1. Objective
<!-- TODO: 1–3 sentences describing what this step achieves and why. -->

## 2. What changed

{changed_table}

<details>
<summary>git diff --stat</summary>

```
{git_diff_stat}
```
</details>

<details>
<summary>git status --short</summary>

```
{git_status}
```
</details>

<details>
<summary>dvc status</summary>

```
{dvc_status}
```
</details>

<details>
<summary>dvc diff</summary>

```
{dvc_diff}
```
</details>

### Stage statistics summary
{metrics_summary}

## 3. Commands run (verbatim)
```bash
# TODO: paste the exact commands run in this step
```

## 4. Results / evidence
<!-- TODO: key numbers, table excerpts, figure paths. Quote real output only. -->

## 5. Decisions and rationale
<!-- TODO: design decisions, alternatives considered, why chosen. Copy ADR-worthy ones to docs/decisions.md -->

## 6. Problems encountered and fixes
<!-- TODO: errors, surprises, data issues and how they were resolved. -->

## 7. Verification
<!-- TODO: commands + expected output to reproduce this step; tests added/passing. -->

## 8. Next step
<!-- TODO: one-line description of the next step. -->
"""

    # Write logbook entry
    _LOGBOOK_DIR.mkdir(parents=True, exist_ok=True)
    entry_path.write_text(entry, encoding="utf-8")

    # ── Write step report JSON ─────────────────────────────────────────────────
    _STEP_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "step": step,
        "title": title,
        "phase": phase,
        "timestamp_ist": now_ist,
        "branch": branch,
        "commit": commit,
        "mlflow_run": mlflow_run,
        "git_status": git_status,
        "git_diff_stat": git_diff_stat,
        "dvc_status": dvc_status,
        "dvc_diff": dvc_diff,
        "metrics_summary": metrics,
        "changed_files": changed_groups,
        "logbook_entry": str(entry_path.relative_to(_REPO_ROOT)),
    }
    report_path = _STEP_REPORTS_DIR / f"{step}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # ── Append to CHANGELOG ────────────────────────────────────────────────────
    _append_changelog(step, title, phase, entry_path)

    # ── Print summary ──────────────────────────────────────────────────────────
    sep = "-" * 70
    print(f"\n{sep}")
    print(f"  STEP-CLOSE: Step {step} - {title}")
    print(sep)
    print(f"  Logbook entry : {entry_path}")
    print(f"  Step report   : {report_path}")
    print(f"  CHANGELOG     : {_CHANGELOG}")
    print(f"  Branch        : {branch}")
    print(f"  Commit        : {commit[:12]}")
    print("\n-- git diff --stat --")
    print(git_diff_stat or "(none)")
    print("\n-- git status --")
    print(git_status or "(clean)")
    print("\n-- Stage metrics summary --")
    print(metrics_summary)
    print(f"\n{sep}")
    print("  [OK] Logbook entry created. Complete the TODO sections, then commit.")
    print(sep + "\n")


if __name__ == "__main__":
    main()
