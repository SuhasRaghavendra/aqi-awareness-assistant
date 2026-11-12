# Step 017 — CI repair and hermetic quality gates

- **Date/time:** 2026-11-12T12:10:00+05:30
- **Author:** <student-A>      **Reviewer:** <student-B>
- **Phase:** 0–3           **Branch / PR:** `main` / #<n>
- **Git commit(s):** `<hash>`      **Data version:** frozen splits unchanged (`val` md5 `462b7712b4cea342fe36ca80868a29f4`); the same bytes are now committed as `tests/fixtures/val_snapshot.parquet`
- **MLflow run IDs (if any):** N/A — parity is checked in a throwaway SQLite registry, not the local `mlflow.db`

## 1. Objective

Make CI able to prove the Phase 0–3 invariants on a clean clone. Before this
step the workflow could not run: `cd.yml` referenced non-existent
`actions/docker-*` action namespaces, Trivy was configured with
`exit-code: "0"` so findings could never fail a build, and the service and model
tests needed `data/processed` + `models/`, which a git checkout does not contain.
The registry was never exercised in CI at all because `mlflow.db` is local-only.

## 2. What changed

| Type | Path / object | Change summary |
|---|---|---|
| ci | `.github/workflows/ci.yml` | Rewritten as seven jobs: `lint`, `unit`, `service`, `model-gate`, `registry`, `docker`, `dvc-consistency`. Pinned `PYTHON_VERSION: "3.12"`, pip cache, concurrency cancellation. |
| ci | `.github/workflows/cd.yml` | `actions/docker-*` → `docker/*` (setup-buildx, login, metadata, build-push); Trivy `exit-code: "0"` → `"1"`. |
| fixtures | `tests/fixtures/val_snapshot.parquet` (new) | The frozen validation split, 717,507 bytes, md5 `462b7712b4cea342fe36ca80868a29f4`, committed for hermetic service/model jobs. |
| tests | `tests/conftest.py` | `resolve_val_path()`, `VAL_SNAPSHOT_MD5`, and bundle fallbacks: `champion_bundle`, `feature_manifest`, `scaler`, `champion`, `val_frame`. Missing DVC outputs fall back to `artifacts/champion`; a missing bundle is still a failure. |
| tests | `tests/test_service.py` | `pytestmark = pytest.mark.service`; reads the snapshot; new test asserting the snapshot md5 cannot silently drift. |
| tests | `tests/test_bundle.py`, `tests/test_model_risk.py`, `tests/test_build_features.py` | DVC-dependent cases marked `data`; seasonal and history-seed tests now use `val_frame` instead of `data/processed/val.parquet`. |
| tests | `tests/test_registry.py` (new) | Two `registry`-marked tests over the parity report. |
| script | `scripts/verify_registry.py` (new) | Registers the committed bundle in a temporary SQLite tracking store, sets the `champion` alias, loads `models:/…@champion` back and writes `reports/metrics/registry_parity.json`. |
| script | `scripts/audit_integrity.py` | `--bundle-only` mode skips `risk_card.file_md5_matches.*` checks that describe DVC-managed outputs; the 13 bundle/risk-card checks still run. |
| docs | `docs/setup_howto.md` | Test-marker table, the exact per-job commands, and the CI/branch-protection section. |
| docs | `docs/decisions.md` | D-035, D-036, D-037. |

## 3. Commands run (verbatim)

```bash
python -m pytest tests -q
python -m ruff check src tests scripts
python -m black --check .
python scripts/audit_integrity.py
python scripts/verify_registry.py
dvc status

# Hermetic CI simulation: temporarily move the DVC outputs aside
pytest -m "not model and not data and not registry and not slow and not service"
pytest -m service tests/test_service.py
pytest -m "model and not data"
python scripts/audit_integrity.py --bundle-only
```

## 4. Results / evidence

Full checkout:

```
149 passed in 18.43s                 # python -m pytest tests -q
All checks passed!                   # ruff
68 files would be left unchanged     # black
Checks: 64  failed: 0                # scripts/audit_integrity.py
Data and pipelines are up to date.   # dvc status
REGISTRY PARITY (temporary sqlite)
  model:      Indian-AQI-Predictor-Parity v1 @champion
  rows:       423
  max |diff|: 0.000e+00 (tolerance 1e-06)
```

Hermetic simulation, with `data/processed/` and `models/` moved aside:

```
46 passed, 103 deselected            # unit
43 passed                            # service (snapshot only)
45 passed, 104 deselected            # model gate (bundle + snapshot only)
Checks: 13  failed: 0                # audit_integrity.py --bundle-only
```

These are the exact commands the six always-available CI jobs run, so the gate
behaviour is demonstrated locally rather than assumed.

**Negative control.** A temporary `tests/test_gate_canary.py` marked `model`
with an unconditional `assert False` was added, and the model-gate command
returned:

```
FAILED tests/test_gate_canary.py::test_deliberate_failure - AssertionError
1 failed, 45 passed, 104 deselected          # exit code 1
```

The canary was then deleted, so a genuine model regression cannot merge while
the gate is green.

## 5. Decisions and rationale

- **D-035** — hermetic service/model gates. Parity, latency, bundle integrity and
  metamorphic behaviour only ever needed the validation split and the bundle.
  Committing the snapshot makes those jobs runnable by anyone, instead of being
  skipped without DVC credentials.
- **D-036** — temporary-registry parity. A throwaway SQLite store exercises the
  real log → register → alias → load path without committing `mlflow.db`.
- **D-037** — six mandatory checks, seven jobs. The conditional full-data job is
  skipped without secrets, and a skipped required check blocks merges on forks.

## 6. Problems encountered and fixes

1. **Windows file lock on the temporary registry.** `TemporaryDirectory`
   cleanup raised `PermissionError: [WinError 32]` because MLflow's SQLAlchemy
   engine pool still held `mlflow.db`. Fixed by calling the store's
   `_dispose_engine()` inside the context and passing
   `ignore_cleanup_errors=True`.
2. **`unit` job failed on `test_scaler_matches_train_split`.** The test reads
   `data/processed/train.parquet`, so it was marked `data`. Likewise
   `test_resolve_classifier_artifact_maps_the_hybrid_entry` needs `models/`.
3. **`--bundle-only` audit failed on `risk_card.file_md5_matches.scaler`.** Those
   five checks describe DVC-managed outputs, so they are skipped in bundle-only
   mode; the bundle, manifest and risk-card checks still run.
4. **Black had to reflow three files after the ruff fixes**, which re-ran
   `dvc repro` for `register`; the second run was a no-op, so byte stability
   held.

## 7. Verification

```bash
pytest -q                                  # 149 passed
python -m ruff check src tests scripts      # clean
python -m black --check .                   # clean
python scripts/audit_integrity.py           # 64/64
python scripts/verify_registry.py           # 423 rows, max diff 0.0
dvc status                                  # up to date
```

To reproduce the hermetic gate locally, move `data/processed` and `models`
aside and run the four commands from §3; every one must exit 0.

## 8. Next step

Step 018 — Gate 3: train the pre-registered monotone-constrained LightGBM,
apply the uniform per-city validation calibration (D-030/D-031), freeze the
candidate, then evaluate the holdout exactly once.
