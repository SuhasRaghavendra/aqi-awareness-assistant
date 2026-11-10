# Step 012 — Registry and bundling

- **Date/time:** 2026-11-10T21:43:00+05:30
- **Author:** <student-A>      **Reviewer:** <student-B>
- **Phase:** 2           **Branch / PR:** `main` / #<n>
- **Git commit(s):** `4b9146c71d6611c02ccc0db060ba326d7135b1bb`      **Data version:** `train.parquet` md5 `49bd4c88748327afa12057c2fe71b0e7`
- **MLflow run IDs:** `182e1d19532f4485bae1f3b789cb0d21` (model version 5)

## 1. Objective

Package the Phase 1 champion into a small serving bundle that lives in Git, and
register the same bundle in MLflow with `challenger`/`champion` aliases. This
gives Phase 3 a stable artifact contract and fixes the old `dvc_data_version =
unknown` lineage tags.

## 2. What changed

| Type | Path | Change summary |
|---|---|---|
| code | `src/models/package_model.py` | Champion selection check, gate policy, residual quantiles, 45-day history seed, MANIFEST, DVC-cache-independent bundle build |
| code | `src/models/register_model.py` | MLflow registration, `challenger`/`champion` aliases, consistency checks, MLflow 2.x/3.x `log_model` compatibility |
| code | `src/serving/bundle.py` | `AqiBundle` contract: model, scaler, features, residual mode, intervals, reference stats, history seed, manifest |
| pipeline | `dvc.yaml` / `dvc.lock` | New `register` stage; `artifacts/champion/` is tracked with `cache: false` so it can be committed to Git |
| tests | `tests/test_bundle.py` | Bundle, manifest, DVC-md5, params-hash, history-seed, interval, fallback and round-trip tests |
| docs | `docs/decisions.md` | ADR D-023 links to this step |

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

- Champion: `residual_lightgbm` (persistence-anchored LightGBM).
- Validation RMSE: **20.099** (persistence 22.584); test RMSE **23.821**, MAE **14.114**, R² **0.8779**.
- Feature count: **103** (read from `data/processed/feature_list.json`; never hard-coded).
- Feature manifest SHA-256: `d3cec166e3db1cfd4194dfa4477c73d077580e913eaf036dd65dfe340206216f`.
- History seed: **405 rows / 45 days** across the nine cities.
- Bundle size: **812,212 bytes** (5 files) in `artifacts/champion/`.
- MANIFEST SHA-256: `4378697963835df9cebf499b75a3fd9055de1694401d4e21f297b8fe1cbd1cf8`.
- Risk-card SHA-256: `36f99565ea0ca88626dec953856a67242cdd98df4effecb487cbe8462a8febf5`.
- Params SHA-256: `d2497f4f50083b7ea68ee0f4be28eadb1e40ba27db7f7d768317863f7a74f178`.
- Gates: G1 pass (val skill 0.1100, test skill 0.0665), G2 pass, **G3 fail**
  (Kolkata ratio 1.0922, waived under ADR D-023), G4 pass (0.9951), G5 pass
  (p95 8.7475 ms).
- MLflow: `Indian-AQI-Predictor` version **5**; aliases `challenger=5` and
  `champion=5`; tags include `features_count=103`, `git_commit`,
  `dvc_data_md5`, `dvc_data_version`, `params_sha256`, `target=next_day_aqi`
  and `risk_card_sha`.
- Registry parity: loading `models:/Indian-AQI-Predictor@champion` reproduced
  the bundle predictions on the 423-row fixture with **max diff 0.000e+00**
  (limit 1e-6).
- Superseded by [Step 013](013-bundle-byte-stability.md): the bundle is now
  811,362 bytes with LF-only text artifacts, MANIFEST SHA-256
  `f765a61da001da89b646d5bf7989827494ce99a4580e8bfd0b3f79898532fac1`, and the
  promoted version is **7**.

## 5. Decisions and rationale

- The bundle is committed to Git and the `register` DVC output uses
  `cache: false`, so CI and Docker can read it without DagsHub credentials.
- Registration always moves `challenger`; only `--promote-only` moves
  `champion`, and only after the MANIFEST SHA, risk-card SHA, feature manifest,
  params hash and gate policy all validate.
- G3 remains a visible failure. ADR D-023 accepts that red gate for this
  bundle; the 1.05 threshold is unchanged.
- The history seed was raised from 31 to 45 days to match the Phase 3 serving
  contract (lags plus 30-day rolling windows).
- `MANIFEST.json` records the `train.parquet` md5 from `dvc.lock` and a
  `params_sha256`, so a bundle cannot silently drift from its data or params.

## 6. Problems encountered and fixes

- The repo pinned MLflow 2.13 in `requirements.txt` at the time, but the working
  environment has MLflow 3.17. `mlflow.pyfunc.log_model` moved from
  `artifact_path` to `name`; the registration helper now selects the supported
  keyword at runtime. The pins were aligned with the runtime in ADR D-024.
- `artifacts/.gitignore` hid the champion bundle. It was changed so
  `artifacts/champion/` is visible to Git, as required for the CI gate.

## 7. Verification

```bash
dvc repro register
python -m src.models.register_model --promote-only
python -m src.models.register_model --verify
python -m pytest -q
ruff check src/ scripts/ tests/
black --check src/ scripts/ tests/
```

Expected: DVC stage green, promotion succeeds, registry parity max diff ≤ 1e-6,
all tests pass and lint/format checks are clean.

## 8. Next step

Phase 3: build the FastAPI inference service on top of `artifacts/champion/`.
