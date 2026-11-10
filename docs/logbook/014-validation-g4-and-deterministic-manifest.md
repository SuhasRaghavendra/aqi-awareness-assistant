# Step 014 — Validation-selected G4 and deterministic manifest

- **Date/time:** 2026-11-10T23:58:00+05:30
- **Author:** <student-A>      **Reviewer:** <student-B>
- **Phase:** 1 / 2           **Branch / PR:** `main` / #<n>
- **Git commit(s):** `a58763273a3d86e664994002809500b7d9389879` (fix applied on top)      **Data version:** `train.parquet` md5 `49bd4c88748327afa12057c2fe71b0e7`
- **MLflow run IDs:** `4c42dda852f444fdae1356026b109229` (model version 9)

## 1. Objective

Remove two measurement-integrity defects found in the Phase 1/2 audit:

1. Gate G4 selected the classifier with test metrics and then reported the gate
   on the same test split, so the gate was optimistic.
2. Rebuilding the bundle changed `MANIFEST.json` on every run because the build
   timestamp came from `datetime.now()`, producing spurious DVC output changes
   and registry versions without any input change.

## 2. What changed

| Type | Path | Change summary |
|---|---|---|
| code | `src/models/model_risk.py` | `select_classifier()` now consumes validation metrics; G4 records the validation selection and evaluates the chosen artifact on test; new `resolve_classifier_artifact()` maps the hybrid leaderboard name to its artifact |
| code | `src/models/model_risk.py` | Risk card and reference stats timestamps use `build_timestamp(root)` |
| code | `src/models/package_model.py` | `MANIFEST.created_at` uses `build_timestamp(base)` |
| code | `src/utils/reproducibility.py` | New helper: `SOURCE_DATE_EPOCH`, else HEAD committer date, else local time |
| tests | `tests/test_model_risk.py` | G4 provenance, selection-scoring, artifact-mapping and reference-stats determinism tests |
| tests | `tests/test_reproducibility.py` | `SOURCE_DATE_EPOCH` override, per-commit stability, no-Git fallback, git parity |
| pipeline | `dvc.lock` | `model_risk` and `register` re-run with the corrected selection and timestamps |
| docs | `docs/decisions.md` | ADR D-026 and D-027 |

## 3. Commands run (verbatim)

```bash
dvc repro register
python -m src.models.register_model --promote-only
python -m src.models.register_model --verify
python -m pytest -q
ruff check src/ scripts/ tests/
black --check src/ scripts/ tests/
```

## 4. Results / evidence

**G4 selection.** `reports/metrics/val_metrics.json` now drives the choice:

| Candidate (validation) | val within-one | val macro F1 |
|---|---:|---:|
| `regress_then_bucket_lightgbm_reg` | **0.9991** | 0.6490 |
| `random_forest_clf` | 0.9973 | 0.6311 |
| `xgboost_clf` | 0.9973 | 0.6417 |
| `logistic_regression` | 0.9970 | 0.6927 |
| `lightgbm_clf` | 0.9964 | 0.6941 |

- Selected artifact: `models/regress_then_bucket.joblib` (validation entry
  `regress_then_bucket_lightgbm_reg`).
- G4 on the holdout test split: **0.9951 ≥ 0.95 → pass**, unchanged from before.
- The risk card records `selected_on: validation`, the leaderboard entry, and the
  validation scores next to the test rate.

**Manifest determinism.** Two consecutive `python -m src.models.package_model`
runs produced byte-identical artifacts:

| File | MD5 of both builds |
|---|---|
| `MANIFEST.json` | `5b944f942d2bf287b3a1cb97c1f8f6f2` |
| `MANIFEST.sha256` | `6006facec811394654a6f5372dfd6ed6` |
| `bundle.joblib` | `69ebdebb91528aced765636017611acf` |

- `created_at` is now the HEAD committer date (`2026-11-10T22:53:52+05:30`)
  instead of wall-clock time.
- Current bundle: **811,359 bytes**, MANIFEST SHA-256
  `d03e40f021b26756c3737caf6a2e7d50676153c398d90b1514d389098c3c8f43`.
- MLflow `Indian-AQI-Predictor` version **9**, `challenger=9` and `champion=9`;
  registry parity **max diff 0.000e+00** on the 423-row fixture.
- Tests: **83 passed**; `ruff` and `black` clean; `dvc status` reports the
  pipeline up to date.

## 5. Decisions and rationale

- **D-026:** select the classifier on validation and report G4 on test. The
  previous implementation chose the winner by test within-one rate and then
  scored that same split, so G4 measured its own selection set. Validation
  selection leaves the test split untouched until the gate is reported.
- **D-027:** derive build timestamps from `SOURCE_DATE_EPOCH` or the HEAD
  committer date. `datetime.now()` made `MANIFEST.json` — and therefore
  `MANIFEST.sha256` — change on every build, which breaks DVC output hashing and
  creates empty registry versions.
- The risk card and reference statistics timestamps use the same helper because
  both files are hashed into the bundle manifest (`risk_card_sha`,
  `reference_stats_sha`); fixing only `created_at` would leave the manifest
  unstable across a full `dvc repro`.

## 6. Problems encountered and fixes

- `train.py` records the hybrid classifier as `regress_then_bucket_<regressor>`
  while the artifact is always `regress_then_bucket.joblib`, so a naive
  validation-selected name resolved to a missing file. `resolve_classifier_artifact()`
  handles the alias and the risk card records both names.
- The `register` stage writes `created_at` into `MANIFEST.json`, which
  `MANIFEST.sha256` covers, so any wall-clock value invalidated the digest.
- Known remaining churn (not fixed in this step): G5 stores measured wall-clock
  latency values in the risk card, so re-running the `model_risk` stage itself
  still changes the card, and therefore the manifest, without a data or code
  change.

## 7. Verification

```bash
dvc status
python -m src.models.register_model --verify
python -m pytest -q
ruff check src/ scripts/ tests/
black --check src/ scripts/ tests/
python scripts/check_logbook.py
```

Expected: DVC up to date; registry parity ≤ 1e-6; all tests pass; lint and
format clean; logbook check passes.

## 8. Next step

Phase 3: build the FastAPI inference service on top of the byte-stable,
reproducibly packaged `artifacts/champion/` bundle.
