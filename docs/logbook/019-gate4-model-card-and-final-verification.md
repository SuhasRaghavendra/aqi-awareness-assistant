# Step 019 — Gate 4: model card, final ADRs and end-to-end verification

- **Date/time:** 2026-11-12T13:00:00+05:30
- **Author:** <student-A>      **Reviewer:** <student-B>
- **Phase:** 0–3           **Branch / PR:** `main` / #<n>
- **Git commit(s):** `<hash>`      **Data version:** frozen splits unchanged (`train` `49bd4c88748327afa12057c2fe71b0e7`, `val` `462b7712b4cea342fe36ca80868a29f4`, `test` `bd1de9ea97083402bd9ccaceb1072e3c`)
- **MLflow run IDs (if any):** `Indian-AQI-Predictor` v23 `@champion` (unchanged by this step)
- **Tag:** `phase3-final`

## 1. Objective

Close Gate 4 of the integrity plan:

1. **D1** — write `docs/model_card.md` with intended use, data, metrics, the
   prediction-interval and `category_probs` methods, gate status and
   limitations.
2. **D2** — finish the ADR pass (D-030…D-038 present and consistent).
3. **D3** — logbook, CHANGELOG and README updated with real hashes.
4. Final verification: full pipeline, tests, audit, registry parity, live
   `uvicorn` latency probe, container smoke; then tag `phase3-final`.

No model, data or threshold changed in this step. The champion, the holdout
report and the G3 waiver are exactly as frozen in step 018.

## 2. What changed

| Type | Path / object | Change summary |
|---|---|---|
| docs | `docs/model_card.md` (new) | Model card: intended/out-of-scope use, family and windows, sources and licensing, leakage controls, val/test/per-city/CV metrics, interval and `category_probs` derivation, gate table with the G3 waiver, limitations, safety/failure modes, reproduction commands and pinned hashes. |
| docs | `docs/decisions.md` | Removed a blank line that split the ADR table; D-001…D-038 now render as one table. No decision content changed. |
| tooling | `scripts/latency_probe.py` (new) | Starts a real `uvicorn` subprocess on a free port, waits for `/ready`, warms up, then measures single-row server latency and nine-city batch wall-clock latency; writes `reports/metrics/latency_probe.json` and fails when p95 exceeds 50 ms / 100 ms. |
| tests | `tests/test_latency_probe.py` (new) | Nearest-rank percentile arithmetic, budgets and in-range probe payloads (12 tests, no server). |
| docs | `README.md`, `docs/CHANGELOG.md` | Champion description, live latency numbers, model-card link, test count 174, logbook/ADR counts, step table. |
| evidence | `reports/metrics/latency_probe.json` (new) | Live probe report from this step. |

## 3. Commands run (verbatim)

```bash
python -m pytest tests -q
python -m ruff check src tests scripts
python -m black --check .
dvc repro
dvc status
python scripts/audit_integrity.py
python scripts/verify_registry.py
python -m src.models.register_model --verify
python scripts/latency_probe.py --iterations 100 --batch-iterations 30
```

## 4. Results / evidence

### Live latency (real `uvicorn`, after warm-up)

```
single row, 100 iterations, server-reported latency_ms
  p50 12.42 ms   p95 13.20 ms   p99 13.85 ms       budget p95 < 50 ms
nine-city batch, 30 iterations, wall clock
  p50 68.68 ms   p95 76.36 ms   p99 79.17 ms       budget p95 < 100 ms
```

The in-process G5 measurement (7.90 ms p95, `reports/metrics/latency.json`)
and the service test budgets are unchanged. Report:
`reports/metrics/latency_probe.json`.

### Verification matrix

| Check | Command | Result |
|---|---|---|
| Test suite | `pytest tests -q` | **174 passed** |
| Lint | `ruff check src tests scripts` | clean |
| Format | `black --check .` | 77 files unchanged |
| Pipeline | `dvc repro` | all stages up to date |
| DVC | `dvc status` | clean |
| Integrity | `scripts/audit_integrity.py` | **64 checks, 0 failed** |
| Registry parity | `scripts/verify_registry.py` | 423 rows, max \|diff\| `0.0` |
| Champion alias | `register_model --verify` | 423 rows, max diff `0.0` |
| Hermetic CI | unit 56, service 43, model 48, bundle-only audit 13/13 | pass with `data/processed` and `models/` removed |

### Container smoke

`docker` is not installed on the development host, so
`scripts/test_container.ps1` / `.sh` could not be executed locally. The
container path is **not** left unverified by assumption: the CI job
`Container Build & Smoke` (`.github/workflows/ci.yml`) builds
`aqi-predictor:test` with `docker build` and runs `scripts/test_container.sh`
(health, ready, metadata, predict, metrics) on every push. This limitation is
stated here and in the model card rather than silently skipped.

## 5. Decisions and rationale

- No new model decision. D-038 remains the record of the frozen champion and
  the retained G3 waiver.
- The live probe is a separate script rather than a pytest test because it
  needs an out-of-process server; keeping it out of the suite avoids a
  port-binding, subprocess-spawning test in every CI job. The percentile
  arithmetic is unit-tested, and CI enforces the same budgets in-process.
- The model card explicitly labels `category_probs` as a normal-residual
  approximation derived from `q10`/`q90`, as Plan Phase 3 requires.

## 6. Problems encountered and fixes

1. **`docker` is unavailable locally.** Documented as an environment
   limitation; enforced in CI instead of being reported as passed.
2. **Percentile mismatch in the new script test.** The first implementation
   used round-half-even over `n-1`, which reports 51.0 for the median of
   1…100. Replaced with the standard nearest-rank definition
   (`ceil(q*n)-1`) and aligned the test.

## 7. Verification

```bash
python -m pytest tests -q                      # 174 passed
python -m ruff check src tests scripts         # clean
python -m black --check .                      # 77 files unchanged
dvc repro && dvc status                        # up to date, clean
python scripts/audit_integrity.py              # 64/64
python scripts/verify_registry.py              # 423 rows, max diff 0.0
python scripts/latency_probe.py                # p95 13.20 ms / 76.36 ms
```

## 8. Next step

Tag `phase3-final`. Phases 0–3 are closed: integrity gates, hermetic CI,
validation-only model remediation, a single-shot holdout, a model card and
end-to-end verification.
