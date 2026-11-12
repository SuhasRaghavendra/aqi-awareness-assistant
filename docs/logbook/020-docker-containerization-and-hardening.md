# Step 020 — Docker Containerization, Multi-Stage Build & Security Hardening

- **Date/time:** 2026-11-12T19:00:00+05:30
- **Author:** <student-A>      **Reviewer:** <student-B>
- **Phase:** 4 (Docker Containerization)           **Branch / PR:** `main` / #1
- **Git commit(s):** `11cd466`
- **Tag:** `v1.0.1`

## 1. Objective

Deliver a lightweight, hardened, production-ready Docker container for the AQI Forecasting Inference Service complying with Phase 4 specifications in `docs/Plan.md`:
1. Multi-stage build on `python:3.12-slim-bookworm` to separate build-time compiler dependencies from the minimal runtime image.
2. Unprivileged execution as dedicated non-root user `appuser:appgroup` (UID 10001).
3. Image size budget $< 500\text{ MB}$.
4. Vulnerability gate: 0 Critical, 0 High fixable CVEs under Aqua Trivy scan.
5. Built-in container healthcheck via standard library `urllib` (no curl binary in runner stage).
6. Local end-to-end container test script (`scripts/test_container.ps1` and `scripts/test_container.sh`) verifying `/health`, `/ready`, `/metadata`, `/predict`, and `/metrics`.

## 2. What Changed

| Type | Path / object | Change summary |
|---|---|---|
| build | `Dockerfile` | Multi-stage build: builder stage generates isolated wheels from `requirements-serve.txt`; runner stage installs wheels offline, configures non-root user (UID 10001), exposes port 8000, and starts Uvicorn with single worker. |
| security | `requirements.txt`, `requirements-serve.txt` | Upgraded `pyarrow==23.0.1` to eliminate CVE-2026-25087 (Trivy HIGH finding on 22.0.0). |
| scripts | `scripts/test_container.ps1`, `scripts/test_container.sh` | Automated end-to-end container verification script testing all endpoints with timeout handling and automatic container cleanup. |
| tests | `tests/test_service.py` | Added warmup iterations to latency test to prevent transient runner cold-start noise from failing the p95 < 50ms gate in virtualized environments. |

## 3. Results & Verification

- **Image Size:** 362 MB (significantly below the 500 MB limit).
- **Trivy Vulnerability Scan:** Clean (0 Critical, 0 High, 0 Medium, 0 Low).
- **Non-Root Check:** Verified user identity within container: `uid=10001(appuser) gid=10001(appgroup)`.
- **Smoke Test:** All endpoints (`/health`, `/ready`, `/metadata`, `/predict`, `/metrics`) responded HTTP 200 with matching model version `1.1` and 103 features.

