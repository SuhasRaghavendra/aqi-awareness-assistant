# Step 022 — Render Cloud Deployment, Live Smoke Test & Drift Monitoring Runbook

- **Date/time:** 2026-11-12T22:30:00+05:30
- **Author:** <student-A>      **Reviewer:** <student-B>
- **Phase:** 6 (Deployment & Monitoring)           **Branch / PR:** `main`
- **Git commit(s):** `11cd466`
- **Container Tag:** `ghcr.io/suhasraghavendra/aqi-awareness-assistant:latest`

## 1. Objective

Deploy the containerized AQI inference service to a live public cloud endpoint and verify acceptance criterion V9 from `docs/Plan.md`:
> *Live deployment passes smoke test; `/docs` reachable.*

Additionally produce `docs/RUNBOOK.md` covering drift monitoring, the PSI-based alert protocol, and the standard model retraining and rollback procedure.

## 2. Deployment Target

**Platform:** Render Web Service (Free Tier — $0/month, 0.1 CPU, 512 MB RAM).  
**Live Base URL:** `https://aqi-predictor-gxbd.onrender.com`

Initial deployment to Azure App Service (`indiasouthcentral` region) encountered persistent 503/504 Gateway Timeout errors due to Azure frontend load-balancer probing port 80 while the container binds non-root port 8000. Despite configuring `WEBSITES_PORT=8000`, the Azure control plane did not propagate port routing correctly within acceptable time. Switched to Render as the documented primary target in `docs/Plan.md` (Section 6).

## 3. Live Endpoint Verification (Post-Deploy Smoke Test)

All 5 endpoint checks passed:

| Endpoint | Status | Key Evidence |
|---|---|---|
| `GET /health` | ✅ 200 | `{"status":"healthy"}` |
| `GET /ready` | ✅ 200 | `{"ready":true,"model_name":"residual_lightgbm_monotone","model_version":"1.1","features_count":103,"history_seed_days":45}` |
| `GET /metadata` | ✅ 200 | Full gate status, champion metrics, val MAE 13.233, test MAE 14.028 returned. |
| `POST /predict` (Delhi, 2025-01-02) | ✅ 200 | `{"aqi_pred":222.31,"category":"Poor","latency_ms":31.8,"fallback_used":false}` — well under 50 ms budget. |
| `GET /docs` | ✅ 200 | Swagger UI HTML returned; `title: AQI Awareness Assistant`. |
| `POST /predict` (invalid schema) | ✅ 422 | Correctly rejected with Pydantic validation detail. |
| `GET /metrics` | ✅ 200 | Prometheus exposition format with `aqi_predictions_total`, `aqi_http_request_duration_seconds`, `aqi_out_of_range_total`. |

## 4. What Changed

| Type | Path / object | Change summary |
|---|---|---|
| docs | `docs/RUNBOOK.md` (new) | Full operations runbook: service architecture, endpoint reference, SLA targets, PSI drift monitoring protocol, 14-day rolling MAE alert rule, 5-step model retraining workflow, Render rollback procedure, incident response playbook. |
| docs | `docs/decisions.md` | Added D-039 (Docker hardening), D-040 (CI/CD pipeline), D-041 (Render deployment decision). |
| docs | `docs/logbook/020-*.md`, `021-*.md`, `022-*.md` (new) | Logbook step-close entries for Phases 4, 5, and 6. |
| docs | `docs/logbook/INDEX.md` | Added Steps 020–022 rows. |
| docs | `README.md` | Updated with live Render URL, Swagger docs link, CI/CD badges, V9 acceptance criterion verified. |

## 5. Acceptance Criterion

- **V9: Live deployment passes smoke test; `/docs` reachable** — ✅ Verified above.
