# Step 016 — Integrity, atomic artifacts and serving safety

- **Date/time:** 2026-11-12T11:30:00+05:30
- **Author:** <student-A>      **Reviewer:** <student-B>
- **Phase:** 0–3           **Branch / PR:** `main` / #<n>
- **Git commit(s):** `<hash>`      **Data version:** frozen splits, md5 `train 49bd4c88…` / `val 462b7712…` / `test bd1de9ea…` (no data change in this step)
- **MLflow run IDs (if any):** N/A — serving consumes the regenerated champion bundle; MLflow v16 is promoted to `@champion`

## 1. Objective

Make the "no corruption" and "no leakage" claims testable rather than
asserted. Step 015 left four gaps: `AqiBundle.load()` unpickled the payload
before checking anything, artifact writes were not atomic, nothing froze the
split hashes, and the serving history store had no safety tests. This step
closes all four without changing the model or the data.

## 2. What changed

| Type | Path / object | Change summary |
|---|---|---|
| code | `src/serving/bundle.py` | `load(verify=True)` verifies `MANIFEST.sha256` → `MANIFEST.json` → `bundle.joblib` **before unpickling**, then `reference_stats.json` and `history_seed.parquet`; `verify_structure()` checks feature count, `feature_names_sha256`, scaler `n_features_in_`, LightGBM `booster_.num_feature()`, residual-quantile ordering/coverage and seed columns. `save()` is atomic and writes `MANIFEST.sha256` last. Bundle format `1.1` stores the manifest externally and adds `bundle_sha256` / `bundle_size_bytes`. |
| code | `src/utils/io.py` (new) | `atomic_write_bytes/text/json/parquet/joblib` (temp file + `fsync` + `os.replace`) and `replace_directory` with `.bak` rollback. |
| code | `src/models/package_model.py` | Publishes through `artifacts/.champion.staging` + `replace_directory`; history seed drops `target_aqi`/`target_category`; `created_at` is data-derived. |
| code | `src/models/model_risk.py` | Timestamps are data-derived; latency percentiles move to `reports/metrics/latency.json`, the hashed card keeps `threshold_ms` + pointer. |
| code | `src/utils/reproducibility.py` | Added `frame_timestamp()` / `path_timestamp()`. |
| code | `src/features/build_features.py`, `src/utils/stage_stats.py` | Deterministic timestamps and atomic JSON writes; change-log append is opt-in via `AQI_STAGE_CHANGELOG=1`. |
| code | `src/models/register_model.py` | Registration consumes the verifying loader, so a corrupt bundle cannot be registered. |
| params | `params.yaml` | new `integrity:` block with the three frozen split md5s; `integrity` added to the `register` stage params. |
| dvc | `dvc.yaml` | new `reports/metrics/latency.json` metric. |
| tests | `tests/test_io.py`, `tests/test_bundle.py`, `tests/test_data_integrity.py`, `tests/test_service.py` | 26 new tests: atomic failure injection, tamper detection, audit + drift detection, and history-store safety. |
| docs | `docs/decisions.md` | D-030…D-034. |

## 3. Commands run (verbatim)

```bash
python -m pytest tests -q
python -m ruff check src tests scripts
python -m black --check .
python scripts/audit_integrity.py
dvc repro
dvc status
python -m src.models.register_model --promote-only
python -m src.models.register_model --verify
```

## 4. Results / evidence

```
146 passed in 9.99s                 # python -m pytest tests -q
All checks passed!                  # ruff
66 files would be left unchanged    # black
INTEGRITY AUDIT (full)
Checks: 64  failed: 0               # scripts/audit_integrity.py
Data and pipelines are up to date.  # dvc status
Promoted Indian-AQI-Predictor v16 to @champion
Loaded models:/Indian-AQI-Predictor@champion on 423 rows; max diff 0.000e+00
```

**Byte-stability proof.** After the integrity changes, every downstream output
was deleted (`artifacts/champion`, the splits, `feature_list.json`,
`scaler.joblib`, risk card, reference stats) and rebuilt with `dvc repro`.
All 12 hashed artifacts were **byte-identical** to the pre-deletion build, and
a further `dvc repro` reported every stage as "didn't change, skipping".

**Tamper evidence.** Flipping one byte of `bundle.joblib`, appending a space to
`MANIFEST.json`, altering `reference_stats.json` or `history_seed.parquet`, or
removing `MANIFEST.sha256` each raises `BundleIntegrityError` or
`FileNotFoundError` before deserialisation.

**Leak evidence.** The audit recomputes `target_aqi[t] == aqi[t+1]` and
`aqi_lag_1[t] == aqi[t-1]` per city on all three splits, confirms the scaler
matches a fresh train-only fit within `1e-6`, and confirms the history seed has
no `target_*` columns.

## 5. Decisions and rationale

- **D-032** — hash before unpickle, atomic writes, staging swap. A hash check
  after `joblib.load()` is useless because the payload has already executed.
- **D-033** — frozen split md5s in `params.yaml`, enforced by the audit and by
  tests. `dvc repro --force` had already demonstrated how easily the splits can
  change underneath the project.
- **D-034** — data-derived timestamps, latency out of the hashed card. With
  `HEAD`-derived times a rebuild produced new bytes at every commit, so
  `dvc repro` never converged.
- **D-030 / D-031** — pre-registered, not executed here: the monotone candidate
  and the validation-only G3 remediation are recorded before Gate 3 so their
  acceptance criteria cannot be adjusted after seeing test.

## 6. Problems encountered and fixes

1. **`dvc repro --force` re-ran the live download stages** and changed all three
   splits (`val` md5 became `f27638c4…`). Recovered with
   `git checkout HEAD -- dvc.lock` + `dvc checkout`; the frozen md5s were
   restored and verified. This incident is why D-033 exists. `--force` is now
   never used on this pipeline.
2. **`OSError: [Errno 9] Bad file descriptor` on Windows** when `fsync`-ing a
   temp parquet/joblib opened with `"rb"`. Windows cannot fsync a read-only
   handle; switched to `"rb+"`.
3. **`feature_list.json` embedded `datetime.now()`**, so a fresh rebuild
   invalidated `train`/`evaluate`/`model_risk`/`register` even when nothing
   changed. Fixed with `frame_timestamp(df_test)`.
4. **`MANIFEST.json` changed after the first re-run** only because `params.yaml`
   gained the integrity block, i.e. a real provenance change, not
   nondeterminism. Confirmed by re-running: zero diffs.

## 7. Verification

```bash
python -m pytest tests -q                      # 146 passed
python scripts/audit_integrity.py              # 64/64, exit 0
dvc repro && dvc status                        # up to date
python -m src.models.register_model --verify   # max diff 0.0
```

To reproduce the tamper test: copy `artifacts/champion` to a temp dir, flip a
byte in `bundle.joblib`, and call `AqiBundle.load(tempdir)` — it must raise
`BundleIntegrityError` without unpickling.

## 8. Next step

Step 017 — CI repair: valid Docker action namespaces, Trivy `exit-code: "1"`,
the committed validation snapshot, and the six required jobs.
