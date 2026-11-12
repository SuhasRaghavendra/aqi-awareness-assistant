# Step 021 — GitHub Actions CI/CD Pipeline & GHCR Publishing

- **Date/time:** 2026-11-12T20:30:00+05:30
- **Author:** <student-A>      **Reviewer:** <student-B>
- **Phase:** 5 (CI/CD Workflows)           **Branch / PR:** `main` / #1
- **Git commit(s):** `11cd466`      **Tag:** `v1.0.2`

## 1. Objective

Deliver fully automated GitHub Actions CI/CD pipelines satisfying Phase 5 specifications:
1. `ci.yml`: 7-job parallel gate suite triggered on every pull-request and push to `main`.
2. `cd.yml`: Tag-triggered CD pipeline building, security-scanning, and publishing the container to GHCR.
3. Broken-model PR blocked by the `model-gate` required status check.
4. Green PR merged end-to-end.

## 2. What Changed

| Type | Path / object | Change summary |
|---|---|---|
| workflow | `.github/workflows/ci.yml` | 7-job CI pipeline: `lint`, `model-gate`, `registry`, `service`, `unit`, `dvc-consistency`, `docker`. First 6 set as required branch protection checks. |
| workflow | `.github/workflows/cd.yml` | CD pipeline: GHCR login, multi-tag buildx build, Trivy scan (fail on fixable CRITICAL/HIGH), push. Lowercased `GITHUB_REPOSITORY` to satisfy GHCR naming rules. |
| fix | `.github/workflows/cd.yml` | Pinned `aquasecurity/trivy-action@v0.36.0`; passed `TRIVY_USERNAME` / `TRIVY_PASSWORD` for authenticated registry scan to avoid anonymous rate limiting. |

## 3. Results / Evidence

| Run | Status | Jobs |
|---|---|---|
| CI `43245070737` on `main@11cd466` | All 7 ✅ | lint, model-gate, registry, service, unit, dvc-consistency, docker |
| CD `43245139189` on tag `v1.0.2` | ✅ | Build, Trivy (0 High/Critical), Push |
| Broken-model PR | ❌ Blocked | `model-gate` correctly blocked merge on branch `test/broken-model-gate` |

Published tags:
- `ghcr.io/suhasraghavendra/aqi-awareness-assistant:latest`
- `ghcr.io/suhasraghavendra/aqi-awareness-assistant:v1.0.2`
- `ghcr.io/suhasraghavendra/aqi-awareness-assistant:sha-11cd466`

