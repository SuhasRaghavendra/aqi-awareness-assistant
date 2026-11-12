# AQI Forecasting Service — Operations Runbook

> **Target Audience:** MLOps Engineers, On-Call Engineers, Model Maintainers.  
> **Service Name:** `aqi-predictor` (`aqi-awareness-assistant`)  
> **Repository:** `https://github.com/SuhasRaghavendra/aqi-awareness-assistant`  
> **Live Production Base URL:** `https://aqi-predictor-gxbd.onrender.com`  
> **Documentation (Swagger UI):** `https://aqi-predictor-gxbd.onrender.com/docs`  
> **Metrics (Prometheus):** `https://aqi-predictor-gxbd.onrender.com/metrics`  
> **Container Registry:** `ghcr.io/suhasraghavendra/aqi-awareness-assistant:latest`  

---

## 1. System Overview & Architecture

The Air Quality Index (AQI) Forecasting Inference Service serves continuous next-day AQI forecasts and CPCB health risk classifications for 9 major Indian metropolitan cities (Ahmedabad, Bengaluru, Chennai, Delhi, Hyderabad, Kolkata, Lucknow, Mumbai, Pune).

### Architecture Diagram
```
                     +----------------------------------------------+
                     |             Client / Edge Caller             |
                     +----------------------------------------------+
                                            |
                                  HTTPS (Cloudflare / TLS)
                                            v
                     +----------------------------------------------+
                     |           Render Web Service (PaaS)          |
                     |  - Host: 0.0.0.0:8000                        |
                     |  - User: appuser (UID 10001, non-root)       |
                     |  - Process: Uvicorn + FastAPI                |
                     |  - Engine: LightGBM Monotone + Calibrator    |
                     +----------------------------------------------+
                             /            |             \
                            v             v              v
                     +------------+ +------------+ +------------+
                     |  /health   | |  /predict  | |  /metrics  |
                     |  /ready    | | (Inference)| |(Prometheus)|
                     |  /metadata | |            | |            |
                     +------------+ +------------+ +------------+
```

---

## 2. Production Service Endpoints

| Endpoint | Method | Expected Status | Purpose |
|---|---|---|---|
| `/health` | `GET` | `200 OK` | Liveness probe: returns `{"status":"healthy"}`. Used by Render health checker. |
| `/ready` | `GET` | `200 OK` | Readiness probe: asserts model, scaler, feature manifest, and 45-day history seed are in RAM. |
| `/metadata`| `GET` | `200 OK` | Model lineage: returns model name, version, champion alias, feature count, validation metrics, and gate status. |
| `/predict` | `POST` | `200 OK` | Single-city prediction: returns `aqi_pred`, `category`, `category_probs`, `aqi_interval_80`, and latency. |
| `/predict/batch` | `POST` | `200 OK` | Vectorized batch prediction for all 9 cities. |
| `/metrics` | `GET` | `200 OK` | Prometheus scrape endpoint (requests total, latency histograms, prediction distribution, out-of-range counts). |
| `/docs` | `GET` | `200 OK` | Interactive OpenAPI / Swagger documentation. |

---

## 3. SLA & Performance Targets

- **Single-City Prediction Latency:** $p95 < 50\text{ ms}$ (Production observed: $\sim 13\text{ ms}$).
- **9-City Batch Latency:** $p95 < 100\text{ ms}$ (Production observed: $\sim 76\text{ ms}$).
- **Availability:** $\ge 99.5\%$ (Render free-tier instances sleep after 15 min of inactivity; cold-start wake-up takes $\approx 30\text{ s}$).
- **Error Budget:** Zero unhandled 500 errors; invalid schemas must return 422 Unprocessable Entity; unphysical bounds return 400 Bad Request.

---

## 4. Drift Monitoring & Alerting Protocol

### 4.1 Input Feature Drift (Population Stability Index - PSI)
Feature distributions are baselined against `reports/metrics/reference_stats.json` (computed from the training split).

$$\text{PSI} = \sum \left( P_i - Q_i \right) \times \ln\left( \frac{P_i}{Q_i} \right)$$

*Where $P_i$ is the production request distribution and $Q_i$ is the training reference distribution.*

| PSI Range | Severity | Action Required |
|---|---|---|
| $\text{PSI} < 0.1$ | **Green (Normal)** | No action needed. Natural seasonal fluctuation. |
| $0.1 \le \text{PSI} \le 0.2$ | **Amber (Moderate Shift)** | Log warning in monitoring dashboard; observe feature trends over 7 days. |
| $\text{PSI} > 0.2$ | **Red (Significant Drift)** | Alert on-call; trigger data validation audit; initiate Model Retraining Workflow. |

### 4.2 Out-of-Range Metric (`aqi_out_of_range_total`)
When a feature value falls outside $[\min_{\text{train}}, \max_{\text{train}}]$, the inference service increments `aqi_out_of_range_total{field="<feature_name>"}` and attaches a warning in the response.
- **Alert Rule:** If $\text{rate}(aqi\_out\_of\_range\_total[1h]) > 5\%$ of total requests, investigate sensor calibration drift or regional anomaly.

### 4.3 Realized Model Performance Drift (Rolling 14-Day MAE)
Clients report realized AQI for day $t$ when querying day $t+1$. The service tracks realized prediction errors.
- **Validation MAE Baseline:** $13.23\text{ AQI points}$.
- **Alert Rule:** If 14-day rolling realized MAE exceeds $1.5 \times \text{Validation MAE}$ ($> 19.85\text{ AQI points}$), trigger **P1 Retraining Alert**.

---

## 5. Model Retraining & Release Workflow

When drift or performance alerts fire, execute the standard retraining protocol:

### Step 1: Ingest Fresh Data & Update Splits
```bash
# 1. Activate project environment
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# 2. Run DVC pipeline reproduction
dvc repro
```

### Step 2: Validate Data Integrity & Gates
```bash
# Verify train/val/test split hashes and schema integrity
python scripts/audit_integrity.py

# Run unit tests and regression quality gates
pytest tests/ -m "not model" -q
pytest tests/ -m model -q
```

### Step 3: Verify Model Acceptance Gates
Ensure the retrained champion passes the 5 production quality gates:
1. **Gate 1 (Skill Score):** Skill vs. persistence $> 0$ on validation and holdout test.
2. **Gate 2 (Floor):** Test $R^2 \ge 0.65$ and $\text{MAE} \le 30$.
3. **Gate 3 (Per-City Robustness):** Model MAE $\le 1.05 \times$ persistence MAE across cities (or documented formal waiver).
4. **Gate 4 (Category Recall):** Within-one-category accuracy $\ge 0.95$.
5. **Gate 5 (Latency Budget):** $p95$ single-row latency $< 50\text{ ms}$.

### Step 4: Package and Promote Champion
```bash
# Package champion bundle with atomic manifest verification
python -m src.models.package_model

# Promote model in MLflow model registry
python -m src.models.register_model --promote
```

### Step 5: Trigger CI/CD Deployment
```bash
# Commit updated artifacts and bump release tag
git add artifacts/champion/ docs/ reports/
git commit -m "feat(model): retrain champion model following drift alert"
git tag v1.0.3
git push origin main --tags
```
GitHub Actions `cd.yml` will automatically:
1. Run linting, unit tests, and model gate checks.
2. Build and scan the Docker container using Aqua Trivy.
3. Push `ghcr.io/suhasraghavendra/aqi-awareness-assistant:latest` and `ghcr.io/suhasraghavendra/aqi-awareness-assistant:v1.0.3`.
4. Render detects the new image tag and redeploys without downtime.

---

## 6. Rollback Procedures

If an incident or regression occurs post-deployment:

### 1-Click Rollback via Render Dashboard
1. Open **[dashboard.render.com](https://dashboard.render.com)** $\to$ select `aqi-predictor`.
2. Navigate to **Settings** $\to$ **Image URL**.
3. Change image tag from `:latest` to the previous stable commit SHA tag:
   ```text
   ghcr.io/suhasraghavendra/aqi-awareness-assistant:sha-11cd466
   ```
4. Click **Save Changes**; Render will restart the service with the previous verified container within 20 seconds.

---

## 7. Incident Response Playbook

| Symptom | Diagnostic Step | Immediate Fix |
|---|---|---|
| **502 / 504 Bad Gateway** | Check Render Logs tab for unhandled container crash or out-of-memory. | Check if RAM exceeded 512 MB. Verify `WORKERS=1` in environment. Restart instance. |
| **503 Service Unavailable** | Query `/ready` endpoint. | Model bundle or history seed failed to load from disk. Verify bundle integrity and file permissions. |
| **High Latency ($> 100\text{ ms}$)** | Inspect Prometheus `aqi_http_request_duration_seconds`. | Check CPU usage. If cold start, send synthetic keepalive pings every 10 min using a cron monitor. |
| **422 Validation Errors** | Inspect client payload structure against `/docs`. | Client payload missing mandatory pollutant/weather features or bad date format (`YYYY-MM-DD`). |

