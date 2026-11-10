# AQI Awareness Assistant — MLOps Milestone 2 Implementation Plan (v2)

*Built from the MLOPS\_Project repo snapshot (pipeline last run 2026-11-07) and your v1 plan. Supersedes docs/milestone2\_implementation\_plan.md. Figures tagged (sandbox) come from a quick, untuned run I did on the validation split to size problems. They are diagnostics, not reportable results.*

## 1. Summary

V1 covers all seven required areas (model testing, risk/quality checks, registry, FastAPI, Docker, CI/CD, deployment) and v2 keeps that structure. Reading the actual repo, though, three problems would make v1's gates fail or become meaningless, and several design points do not match how the code works. V2 therefore adds a **Phase 0** (make the baseline credible), rewrites the quality gates around the real metrics, and redesigns serving around the real feature pipeline.

The three blockers:

1. The current test split is **83 rows, Hyderabad only (2026-01-01 to 2026-03-26)**. Per-city bias testing and any meaningful gate are impossible on it.
2. **Every trained model loses to persistence** on validation and test. V1's gate (R² > 0.65, MAE < 30, skill > 0) would reject all learned models, so there would be nothing to register.
3. The 102 features contain about 45 lag/rolling features, so `/predict` cannot work from "weather + fire counts" alone. It needs per-city history.

## 2. Findings from the repo

| # | Finding | Evidence | Fix |
| --- | --- | --- | --- |
| F1 | Test split is 83 rows, one city | test.parquet: Hyderabad only. Other 8 cities end 2025-08-31 and fall in val (8,325 rows). params.yaml has train\_end 2022-12-31 / val\_end 2025-12-31, but ADR D-004 says train ≤ 2023, val 2024, test 2025+ | Phase 0.1 |
| F2 | Models lose to persistence | val R²: persistence 0.868 vs ridge 0.619, RF 0.622, XGB 0.683, LGBM 0.691. Test R²: persistence 0.696; every learned model is negative (-0.15 to -0.91) | Phase 0.2–0.4 |
| F3 | Likely cause: today's AQI is not a clean feature | feature\_list.json has aqi\_lag\_1…14, rolling stats shifted by 1, and aqi\_diff\_1d, but not `aqi` (day t). On val, aqi ≠ aqi\_diff\_1d + aqi\_lag\_1, so t is not recoverable. (sandbox) adding `aqi` lifts val R² from 0.640 to 0.878 (Ridge) and 0.642 to 0.884 (LightGBM, MAE 24.8 to 14.1) versus persistence 0.868 / MAE 14.5. A LightGBM predicting the change from persistence reached R² 0.889 / MAE 13.7 | Phase 0.2 |
| F4 | Scaler is fitted in three places | build\_features.py saves data/processed/scaler.joblib; train.py and evaluate.py each refit a StandardScaler. All learned models (trees too) were trained on scaled input; baselines take raw frames | Phase 0.3 |
| F5 | Serving needs history | lags 1/3/7/14, rolling 7/14/30-day mean/std/max for AQI and fires, weather lags, grap\_lag\_1, plus derived interactions (fire\_x\_low\_wind, inversion\_x\_fire, festival\_burning\_overlap, diesel\_vehicle\_intensity, winter\_stagnation\_index), season/city one-hots and static city attributes | Phase 3 |
| F6 | Models and data are not in Git | .gitignore excludes models/ and data/processed/; DVC remote is DagsHub (.dvc/config). CI and Docker cannot see them unless they pull from DagsHub or a small bundle is committed | Phases 2, 5 |
| F7 | MLflow registry details | Registry is the gitignored sqlite file (sqlite:///mlflow.db). Stages (Staging/Production) are deprecated in favour of aliases; pinned mlflow 2.13.0 supports aliases. All runs log dvc\_data\_version = unknown because train.py never passes it | Phase 2 |
| F8 | No CI, thin test coverage | .github/ holds only pull\_request\_template.md. 29 test functions cover download, ingest/merge, fires, build\_features. Nothing tests baselines, train, evaluate, aqi\_utils, clean or validate | Phases 1, 5 |
| F9 | Tooling differs from v1 | Repo uses ruff + black (line length 100) + pre-commit, not flake8. requirements.txt pins numpy 1.26.4, pandas 2.2.2, scikit-learn 1.5.0, xgboost 2.0.3, lightgbm 4.3.0, mlflow 2.13.0, with unpinned datagovindia. Many files use CRLF line endings | Phases 0.6, 4 |
| F10 | Docs describe the old pipeline | decisions.md still says OpenAQ (D-001), local DVC remote (D-008), Delhi-only weather (D-013). Real pipeline: XKDR AQI, NASA FIRMS, Open-Meteo ERA5, DagsHub remote. Vehicle data is annual (interpolated to daily), IIP/power are manual downloads | Phase 0.7 |
| F11 | Secrets in the zip | .env and .dvc/config.local were in the archive (I did not open them). They must stay out of Git, Docker context and any shared zip | Phases 4, 5 |

**Already solid:** an 11-stage DVC pipeline with DagsHub remote and dvc.lock; MLflow runs tagged with git\_commit/step\_id/logbook\_entry; 12 saved models; a 102-feature manifest; ruff/black/pre-commit; the Step-Close logbook protocol (steps 001–010) and ADR register. V2 reuses all of these.

## 3. Target architecture

1. **Offline (DVC):** data → features → train → evaluate → **risk** → **register** (new stages). The register stage outputs `artifacts/champion/` with DVC `cache: false`, so this small bundle lives in Git.
2. **Bundle:** model + the single fitted scaler + feature\_names + residual quantiles per city + reference\_stats.json + a history seed (last 45 days per city) + MANIFEST.json (git commit, md5 of train.parquet from dvc.lock, params hash, metrics).
3. **Serving:** FastAPI loads the bundle, builds features with the same code as training, predicts, and exposes /metrics.
4. **Delivery:** GitHub Actions (CI gate, then CD) builds the Docker image from the committed bundle, scans it, pushes to GHCR, deploys to the host, smoke-tests it.
5. **Lineage:** MLflow registers the same bundle as `Indian-AQI-Predictor` with aliases `challenger` and `champion`.

## 4. Phased plan

### Phase 0 — Make the baseline credible (prerequisite)

Objective: a test set that can support gates, and a model that can honestly beat persistence.

- **0.1 Restore the documented split.** In params.yaml set train\_end 2023-12-31 and val\_end 2024-12-31 (per ADR D-004), giving a test set of 2025-01-01 onward. From date coverage that is roughly 2,000 rows across all 9 cities (Hyderabad to March 2026). Record as ADR D-018 and bump the data version in the logbook. Pune has only 553 train rows (2021-01 to 2022-07); note it as a known weak city.
- **0.2 Fix the feature set.** Add day-t `aqi` as an explicit feature (feature count becomes 103) and find out why aqi\_diff\_1d + aqi\_lag\_1 ≠ aqi. Re-run train, evaluate and ablations. From now on **never hard-code 102**: every artifact reads n\_features from feature\_list.json.
- **0.3 One scaler.** train.py and evaluate.py load data/processed/scaler.joblib instead of refitting. Add a test that the saved scaler's mean\_/scale\_ match the train split.
- **0.4 Candidate models.** Keep the 7 regressors. Add a residual variant (target = target\_aqi − aqi, prediction = aqi + model output) for LightGBM and Ridge, chosen only by TimeSeriesSplit CV on train and a single val check, never by test.
- **0.5 Evaluation protocol.** Report per-city MAE/RMSE, skill vs persistence, and a bootstrap 95% CI on skill (resampling by week). Add xgboost\_clf, which train.py's docstring lists but no joblib exists for.
- **0.6 Hygiene.** Run `make fmt` once so the lint gate starts green. Split requirements into requirements.txt (training), requirements-serve.txt (exact pins of pandas, numpy, scikit-learn, lightgbm, joblib, pyyaml, pyarrow only if needed, plus fastapi, uvicorn, pydantic ≥ 2, prometheus-client) and requirements-dev.txt (pytest, pytest-cov, httpx, ruff, black). Add .gitattributes (`*.sh text eol=lf`). Confirm the Python version of your .venv and use the same in CI/Docker (3.11 or 3.12, not both).
- **0.7 Docs.** Add ADRs D-018 onward, refresh data\_card.md and decisions.md for the XKDR/FIRMS pipeline, and close each phase with `make step-close STEP=011…` so the logbook stays complete. The README is updated last.

**Acceptance:** `dvc repro` is green; test covers all 9 cities; at least one candidate beats persistence on val RMSE; ruff and black pass.

### Phase 1 — Model testing and risk/quality audit

Files: `src/models/model_risk.py`, `tests/test_model_risk.py`, `tests/conftest.py`, `tests/fixtures/val_sample.parquet` (about 500 rows), `reports/metrics/model_risk_card.json`, `reports/metrics/reference_stats.json`. Mark these tests `@pytest.mark.model`; if the bundle is missing they must **fail**, not skip.

**1.1 Data and schema checks**

- Parquet columns vs feature\_list.json: names, order, numeric dtypes, no NaN/Inf, n\_features taken from the manifest.
- Range checks against train min/max with a documented tolerance (flag, do not crash).

**1.2 Behavioural tests (revised from v1)**

- Metamorphic tests must perturb **raw inputs and rebuild features**. Editing one column is invalid because fire\_count\_500km also feeds lags, rolling stats, fire\_x\_low\_wind and inversion\_x\_fire.
- *Fires:* take about 500 val rows from burning-season months, scale raw fire inputs ×2 and ×5, rebuild features. Assert mean ΔAQI ≥ 0 and at most 5% of rows decrease by more than 2 AQI points. Tree models are not monotone, so a tolerance is needed; alternatively train LightGBM with `monotone_constraints` on fire and wind features and assert strict direction.
- *Wind/stagnation:* lower wind (same row, wind ÷ 3) must not lower AQI on average; the same tolerance applies.
- *Festival overlap:* replace v1's cross-season comparison (Diwali vs mid-monsoon changes dozens of features at once) with a paired counterfactual: same row with festival\_flag 0 vs 1 during burning season. A separate data-level check asserts mean predicted AQI for Oct–Nov exceeds Jul–Aug for Delhi and Lucknow.
- *Determinism/consistency:* the same input gives the same output; batch prediction equals single prediction; row order does not matter.
- *Boundary:* 0 vs 5,000 fires, extreme temperatures, wind 0 and 40 m/s. Assert finite output, clipped to the 0–500 AQI scale used in training (train max 500).

**1.3 Quality gates (data-driven)** V1's absolute R² > 0.65 / MAE < 30 are kept only as a floor, because persistence on val already has MAE 14.5, so those numbers would pass a weak model. Gates:

- G1: skill vs persistence > 0 on val **and** test (RMSE basis), with the bootstrap CI reported.
- G2: provisional floor R² ≥ 0.65 and MAE ≤ 30 on test. Freeze the final numbers in an ADR after Phase 0 and before looking at test.
- G3 (per city): model MAE ≤ persistence MAE in each of the 9 cities, or within +5%. V1's rule (no city above 1.5× the national mean error) would fail Delhi for any model: (sandbox) Delhi MAE was 27.6 against 13.7 overall, 2.0×, because Delhi is intrinsically harder. Use per-city comparison to persistence, plus MAE as a share of city mean AQI.
- G4 (category): within-one-category rate ≥ 0.95 (val values are 0.98–0.99) and recall for Poor-and-above reported per city.
- G5: p95 single-row latency < 50 ms measured on the bundle, with the Random Forest excluded as champion on size.

**1.4 Risk card** (`model_risk_card.json`): model id, data md5, git commit, split dates, all gate results with thresholds, per-city table, test counts, latency, known limitations (Pune data thinness; ERA5 reanalysis in training vs forecast data in production; annual vehicle data interpolated).

**Acceptance:** all tests pass in CI; risk card regenerated by `dvc repro`.

### Phase 2 — Registry and bundling

Files: `src/models/package_model.py`, `src/models/register_model.py`, a DVC stage `register`, `artifacts/champion/`.

- Champion rule: lowest validation RMSE among models that pass G1–G5. Persistence and other baselines are never champions but ship as **fallback**.
- Bundle class `AqiBundle` (in `src/serving/bundle.py`) contains: model, scaler loaded from data/processed/scaler.joblib, feature\_names, a flag for residual mode, per-city residual quantiles (10/50/90) from val for prediction intervals, reference\_stats, history seed, and MANIFEST.json.
- Register in MLflow as `Indian-AQI-Predictor` with tags features\_count (from manifest), git\_commit, dvc\_data\_md5 (read from dvc.lock), target=next\_day\_aqi, risk\_card\_sha. Fix the existing dvc\_data\_version = unknown tags by passing the md5 into start\_run.
- Use **aliases**, not stages. Registration sets `challenger`; `champion` is set only by the promote step and only if the risk card shows all gates passed and the bundle's MANIFEST sha matches.
- Use relative paths from src/config.py, not v1's absolute C:\\Users file links.
- Because mlflow.db is not shared with CI, the CI gate reads the committed risk card and bundle; MLflow is the lineage record. Optional upgrade: point tracking at DagsHub's MLflow (ADR needed, since D-003 chose local sqlite).

**Acceptance:** `mlflow` shows the version with tags and aliases; loading `models:/Indian-AQI-Predictor@champion` reproduces the bundle's predictions on the fixture within 1e-6.

### Phase 3 — FastAPI inference service

Files: `src/features/serving_features.py` (pure functions refactored from build\_features.py and imported by both training and serving), `src/service/{app,schemas,predictor,history,middleware}.py`, `tests/test_service.py`.

**Design**

- **Request** (`POST /predict`): city (enum of the 9), date, and today's observations: aqi (required), the 9 weather variables, fire\_count\_500km, fire\_frp\_sum\_500km, fire\_dist\_min\_km, and optional overrides such as grap\_stage and festival\_flag.
- **Server-side fill:** calendar and season features from date; city one-hots and static attributes (lat, lon, elevation, population) from configs; vehicle fractions and IIP from the latest reference tables; GRAP stage from configs/grap\_calendar.csv when not supplied.
- **History:** the service keeps a rolling window per city, initialised from the bundle's seed (45 days) and extended by every accepted request. Rolling and lag features then match training (min\_periods=1 behaviour). If fewer than 30 days exist, the response carries `degraded: true` and `history_days`. A stateless alternative is an optional `history` array in the request.
- **Response:** aqi\_pred, aqi\_interval\_80 (from residual quantiles), category, category\_probs, model\_version, alias, warnings (out-of-range inputs), latency\_ms. V1 promised confidence probabilities, but the regression models produce none, so probabilities come from the calibrated classifier or from interval mass over CPCB cut-offs; state which in the model card.
- **Fallback:** if the model or history is unusable, return the persistence value with `model_used: fallback_persistence`.
- **Endpoints:** /health, /ready (model + scaler + history seed loaded), /metadata, /predict, /predict/batch (9 cities, vectorised), /metrics, /docs.
- **Errors:** 422 for schema errors, 400 for impossible values (negative wind, humidity > 100), 503 when not ready. Request-id middleware and JSON logs through src/utils/logging\_utils.py.
- **Prometheus:** request count, latency histogram, prediction histogram, out-of-range counter, drift gauge (Phase 6).

**Tests** (TestClient needs httpx): normal 200; missing field 422; unknown city; out-of-bounds values; /health and /ready; batch of 9; **parity test:** the service's output for 50 val rows equals the offline pipeline within 1e-6 (this catches training/serving skew, the most common deployment bug); latency budget check.

**Acceptance:** p95 < 50 ms single row, < 100 ms for the 9-city batch (measured, not assumed).

### Phase 4 — Docker

Files: `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `scripts/test_container.sh` and `.ps1`.

- Multi-stage on python:3.12-slim-bookworm (match Phase 0.6). Build stage makes wheels from requirements-serve.txt; runtime installs them with `--no-cache-dir`.
- Non-root user appuser (uid 10001); copy only src/service, src/features, src/serving, src/config.py, params.yaml, configs/ and artifacts/champion/.
- HEALTHCHECK uses a Python urllib probe (slim images have no curl).
- Entrypoint: uvicorn src.service.app:app with `--workers 1` on a free-tier host with 512 MB RAM, `--workers 2` elsewhere; make it an env var.
- Image labels: git SHA and bundle version.
- `.dockerignore` must exclude: .env, .dvc/config.local, .git, .dvc, data/raw, data/interim, logs, mlflow.db, mlruns, models/ (full model set), reports/figures, notebooks, docs, tests, .venv.
- Size: aim for under 500 MB and measure. If exceeded, drop xgboost from the serve requirements unless the champion needs it.
- docker-compose: service `aqi-api`; optional profiles `monitoring` (Prometheus, Grafana) and `mlflow` (UI on the sqlite file).
- Test script builds, runs, waits for health, calls /predict, checks the non-root user, and tears down.

**Acceptance:** build is clean; container runs as UID 10001; health passes; Trivy shows no fixable CRITICAL/HIGH findings.

### Phase 5 — CI/CD (GitHub Actions)

**ci.yml** (pull requests and pushes to main). Jobs:

1. **lint:** `ruff check` and `black --check` (what the repo uses), plus pre-commit.
2. **test:** `pytest -m "not model" --cov` using fixtures only (no DVC pull).
3. **model-gate:** runs `pytest -m model` against the committed bundle and fixture sample, and verifies the risk card shows all gates passed and MANIFEST sha matches. Failing blocks the merge.
4. **docker:** builds the image, runs it, runs the smoke test.
5. **dvc-consistency (optional, on main only):** `dvc pull` with DagsHub credentials and `dvc status -c`.

Make lint, test, model-gate and docker **required status checks** in branch protection.

**cd.yml** (tags `v*`, or main after CI):

1. Log in to GHCR with GITHUB\_TOKEN (permissions: packages write).
2. Build with buildx; tags `sha-<commit>`, semver and `latest`. linux/amd64 is enough for the free hosts; add arm64 only if needed (QEMU adds build time).
3. Trivy scan (fail on fixable CRITICAL/HIGH).
4. Push.
5. **Promote:** run `register_model --promote` so alias `champion` moves only after the gates pass.
6. Trigger the Render deploy hook, then run the post-deploy smoke test against the live URL. Failure triggers redeploy of the previous tag.

**Secrets:** DAGSHUB\_USER, DAGSHUB\_TOKEN (optional job), RENDER\_DEPLOY\_HOOK\_URL, SERVICE\_URL. Pin action versions at major tags (checkout, setup-python, build-push-action, trivy-action) and check the current majors when writing the files.

**Acceptance:** a deliberately broken PR (a model that violates a fire-direction test) is blocked; a green PR merges; a tag deploys end to end.

### Phase 6 — Deployment and monitoring

- **Target (decide now):** Render as a Docker web service via render.yaml (healthCheckPath /health). Fallback: Google Cloud Run. Free tiers sleep when idle and have small memory, so check current limits and expect cold starts; run one worker.
- `scripts/smoke_test.py --url`: GET /health, GET /ready, POST /predict with a valid payload, POST with an invalid one (expects 422), GET /docs.
- **Monitoring (what v1's title promised but did not specify):**
  - Prometheus scrapes /metrics (local via compose profile).
  - *Input drift:* PSI per feature over a rolling window of 500 requests against reference\_stats.json; gauge per feature; warn at PSI > 0.2.
  - *Live accuracy:* clients send day t's aqi daily, so the service can compare it with its stored prediction for that city and expose realized absolute error and a rolling MAE. Alert rule: rolling 14-day MAE above 1.5× the validation MAE.
  - *Retraining trigger:* drift or MAE alert opens a retrain (`dvc repro` then re-run the gates). Document it in a runbook.
  - Container filesystems are ephemeral on PaaS, so keep metrics in memory and logs on stdout.
- Rollback: redeploy the previous image tag (kept in GHCR).

**Acceptance:** live URL serves /docs and valid predictions; drift and error gauges visible on /metrics.

## 5. Verification criteria

| ID | Criterion | How measured |
| --- | --- | --- |
| V1 | Test split covers all 9 cities and at least 1,500 rows | row counts by city in the risk card |
| V2 | Champion beats persistence (RMSE skill > 0) on val and test | val\_metrics.json, test\_metrics.json, bootstrap CI |
| V3 | All metamorphic, boundary, determinism and per-city gates pass | pytest -m model |
| V4 | Registry holds champion with tags and aliases; loaded model reproduces the bundle within 1e-6 | MLflow + parity test |
| V5 | Service matches offline pipeline within 1e-6 on 50 val rows | test\_service.py parity test |
| V6 | p95 latency < 50 ms single, < 100 ms batch of 9 | latency test and /metrics |
| V7 | Image builds, runs as non-root, healthy, no fixable CRITICAL/HIGH vulns | test\_container + Trivy |
| V8 | CI green and required; broken-model PR is blocked | Actions history |
| V9 | Live deployment passes smoke test; /docs reachable | smoke\_test.py output |
| V10 | Logbook steps, ADRs, data card and README updated | check\_logbook.py |

## 6. Timeline and ownership (estimates)

Based on the logbook, Suhas (Student A) handled infrastructure and data stages and Srijan (Student B) handled modelling. A suggested split, adjust as you like:

| Phase | Work | Hours | Owner | Depends on |
| --- | --- | --- | --- | --- |
| 0 | Split, feature fix, scaler, candidates, hygiene, ADRs | 12–16 | B (0.1–0.5), A (0.6–0.7) | none |
| 1 | Risk module, tests, gates, risk card | 10–12 | B | 0 |
| 2 | Bundle, registry, aliases, DVC stage | 6–8 | B | 1 |
| 3 | Serving features, API, tests, parity | 14–16 | A | 2 (bundle format) |
| 4 | Dockerfile, compose, scripts | 5–6 | A | 3 |
| 5 | ci.yml, cd.yml, branch protection | 7–8 | A | 1, 4 |
| 6 | Deploy, smoke test, monitoring, runbook | 6–8 | A + B | 5 |
| Report | Milestone 2 report, demo script, README | 6–8 | both | all |

Total about 66–82 hours, so roughly 35–40 hours each. Critical path: 0 → 1 → 2 → 3 → 4 → 5 → 6. Phases 3 and 4 can start on a stub bundle while Phase 1 finishes. Each phase ends with a Step-Close entry (steps 011 onward) and its ADRs.

## 7. Risks

| Risk | Mitigation |
| --- | --- |
| Even after Phase 0 no model beats persistence | Ship the residual model only if G1 holds; otherwise present persistence + features as an honest finding and keep the full MLOps path with persistence as champion |
| Restoring the split changes Milestone 1 numbers | ADR D-018; keep the old results in the M1 report and tag the data versions |
| Training/serving skew (ERA5 reanalysis vs live weather; annual vehicle data) | Parity test; state it in the model card; plan a forecast-weather source as future work |
| Free host limits (memory, cold start) | One worker, small bundle, smoke test with retry |
| DagsHub credentials in CI | Optional job only; committed bundle keeps the main path independent |
| Package pins drift between training and serving | Serve requirements use the exact training pins |

## 8. Decisions I need from you

1. Python version of your .venv (3.11 or 3.12), so CI and Docker match the pickled models.
2. OK to restore the ADR D-004 split (train ≤ 2023, val 2024, test 2025+)? It changes Milestone 1 numbers.
3. Deployment target: Render (my default) or Cloud Run.
4. The Milestone 2 deadline, so the hour estimates can be turned into dates.
5. Whether the Milestone 1 release tag and report must stay frozen.
