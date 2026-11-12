#!/usr/bin/env python
"""Evaluate the frozen champion on the holdout split exactly once (Plan A1).

Usage
-----
    python scripts/final_test_eval.py
    python scripts/final_test_eval.py --check   # re-derive and diff

The script aborts if any split md5 differs from ``params.yaml:integrity``.  A
report for a different candidate is never overwritten silently: pass
``--allow-new-candidate`` (the DVC stage does) after freezing a new champion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import model_risk as mr  # noqa: E402
from src.models.final_eval import (  # noqa: E402
    FINAL_EVAL_RELPATH,
    compare_reports,
    compute_final_test_eval,
    write_final_test_eval,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute and compare against the committed report instead of writing",
    )
    parser.add_argument(
        "--allow-new-candidate",
        action="store_true",
        help="permit replacing a report that belongs to a different candidate",
    )
    args = parser.parse_args()

    root = args.root or mr.repo_root()
    output = args.output or root / FINAL_EVAL_RELPATH
    if not output.is_absolute():
        output = root / output

    if args.check:
        existing = mr.read_json(output)
        fresh = compute_final_test_eval(root)
        problems = compare_reports(existing, fresh)
        if problems:
            print("FINAL TEST EVAL DRIFT:")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print(
            f"[OK] final_test_eval.json matches a fresh evaluation "
            f"({fresh['counts']['rows']} rows, mae {fresh['metrics']['mae']:.3f})"
        )
        return 0

    if output.exists():
        existing = mr.read_json(output)
        current_name = mr.select_champion(mr.read_json(root / "reports/metrics/val_metrics.json"))[
            "model"
        ]
        if (
            existing.get("candidate", {}).get("name") != current_name
            and not args.allow_new_candidate
        ):
            print(
                f"Refusing to overwrite a holdout report for "
                f"{existing['candidate']['name']!r} with {current_name!r}. "
                "Freeze the candidate, then pass --allow-new-candidate.",
                file=sys.stderr,
            )
            return 1

    report, path = write_final_test_eval(root, output)
    print("\n" + "=" * 78)
    print("FINAL HOLDOUT EVALUATION (single shot)")
    print("=" * 78)
    print(
        f"  candidate:  {report['candidate']['name']} ({report['candidate']['artifact_md5'][:12]})"
    )
    print(f"  rows:       {report['counts']['rows']} across {report['counts']['cities']} cities")
    print(
        f"  test:       MAE {report['metrics']['mae']:.3f}  RMSE {report['metrics']['rmse']:.3f}  "
        f"R2 {report['metrics']['r2']:.4f}"
    )
    print(
        f"  skill:      {report['skill_vs_persistence']:.4f} "
        f"[{report['skill_ci_lo_95']:.4f}, {report['skill_ci_hi_95']:.4f}]"
    )
    print(f"  test md5:   {report['splits']['test']}")
    print(f"  report:     {path}")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
