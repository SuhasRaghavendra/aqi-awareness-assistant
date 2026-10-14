#!/usr/bin/env python3
"""
scripts/log_step.py
Creates a logbook entry and step report for the Step-Close Protocol.

Usage:
    python scripts/log_step.py --step 001 --title "Repo scaffold" --phase 2
    python scripts/log_step.py --step 007 --title "Train XGBoost" --phase 6 --mlflow-run abc123
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGBOOK_DIR = ROOT / "docs" / "logbook"
STEP_REPORTS_DIR = ROOT / "reports" / "step_reports"
CHANGELOG_PATH = ROOT / "docs" / "CHANGELOG.md"
METRICS_DIR = ROOT / "reports" / "metrics"


def run(cmd: list[str], cwd: Path = ROOT) -> str:
    """Run a shell command and return stdout; return empty string on error."""
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, cwd=cwd
        ).stdout.strip()
    except Exception:
        return ""


def slugify(text: str) -> str:
    """Convert title to a safe slug."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:50]


def load_latest_metrics() -> dict:
    """Load and merge all *_stats.json files from reports/metrics/."""
    combined: dict = {}
    if METRICS_DIR.exists():
        for f in sorted(METRICS_DIR.glob("*_stats.json")):
            try:
                data = json.loads(f.read_text())
                combined[f.stem] = {
                    "duration_seconds": data.get("duration_seconds"),
                    "total_dropped": data.get("total_dropped"),
                    "notes": data.get("notes", []),
                }
            except Exception:
                pass
    return combined


def create_logbook_entry(step: str, title: str, phase: int, mlflow_run: str) -> Path:
    """Create a logbook markdown file from the template."""
    LOGBOOK_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify(title)
    entry_path = LOGBOOK_DIR / f"{step}-{slug}.md"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    git_branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    git_commit = run(["git", "rev-parse", "--short", "HEAD"])
    git_diff_stat = run(["git", "diff", "--stat", "HEAD~1"]) or "(first commit)"
    git_status = run(["git", "status", "--short"])
    dvc_status = run(["dvc", "status"]) or "(no DVC changes)"

    metrics_summary = json.dumps(load_latest_metrics(), indent=2)

    content = f"""# Step {step} — {title}
- **Date/time:** {now} (IST)
- **Author:** <student>      **Reviewer:** <other student>
- **Phase:** {phase}           **Branch / PR:** {git_branch} / #<n>
- **Git commit(s):** {git_commit}      **Data version:** <git tag + dvc md5>
- **MLflow run IDs (if any):** {mlflow_run or "N/A"}

## 1. Objective
<!-- What this step is meant to achieve and why (1–3 sentences). -->

## 2. What changed
| Type | Path / object | Change summary |
|---|---|---|
| code | src/... | |

### git diff --stat
```
{git_diff_stat}
```

### git status
```
{git_status}
```

### dvc status
```
{dvc_status}
```

### Stage metrics summary
```json
{metrics_summary}
```

## 3. Commands run (verbatim)
```bash
# Fill in the exact commands run
```

## 4. Results / evidence
<!-- Key numbers, small tables, figure paths. Quote real output only. -->

## 5. Decisions and rationale
<!-- Each decision, alternatives considered, why chosen. -->

## 6. Problems encountered and fixes
<!-- Errors, surprises, data issues, and how they were resolved. -->

## 7. Verification
```bash
# How to reproduce/check this step
pytest -q
```

## 8. Next step
<!-- One line. -->
"""
    entry_path.write_text(content, encoding="utf-8")
    return entry_path


def append_changelog(step: str, title: str, phase: int) -> None:
    """Append a line under [Unreleased] in CHANGELOG.md."""
    CHANGELOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CHANGELOG_PATH.exists():
        CHANGELOG_PATH.write_text(
            "# Changelog\n\nAll notable changes to this project will be documented here.\n"
            "\n## [Unreleased]\n",
            encoding="utf-8",
        )

    content = CHANGELOG_PATH.read_text(encoding="utf-8")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    new_line = f"- **[Step {step}]** Phase {phase}: {title} ({now})\n"

    if "## [Unreleased]" in content:
        content = content.replace("## [Unreleased]\n", f"## [Unreleased]\n{new_line}", 1)
    else:
        content += f"\n## [Unreleased]\n{new_line}"

    CHANGELOG_PATH.write_text(content, encoding="utf-8")


def write_step_report(step: str, title: str, entry_path: Path) -> Path:
    """Write a machine-readable JSON step report."""
    STEP_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "step": step,
        "title": title,
        "logbook_entry": str(entry_path),
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": load_latest_metrics(),
    }
    report_path = STEP_REPORTS_DIR / f"{step}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a logbook entry for a pipeline step.")
    parser.add_argument("--step", required=True, help="Step number, e.g. 001")
    parser.add_argument("--title", required=True, help="Short step title")
    parser.add_argument("--phase", required=True, type=int, help="Phase number 1–9")
    parser.add_argument("--mlflow-run", default="", help="MLflow run ID(s), if any")
    args = parser.parse_args()

    entry_path = create_logbook_entry(args.step, args.title, args.phase, args.mlflow_run)
    report_path = write_step_report(args.step, args.title, entry_path)
    append_changelog(args.step, args.title, args.phase)

    print(f"\n✅ Logbook entry created: {entry_path}")
    print(f"✅ Step report written:   {report_path}")
    print(f"✅ CHANGELOG updated:     {CHANGELOG_PATH}")
    print("\nNext: fill in the narrative sections (Objective, Decisions, Problems) in the logbook entry.")
    print("Then run: git add -A && git commit -m 'docs(step-{step}): logbook'".format(step=args.step))


if __name__ == "__main__":
    main()
