"""
src/service/dashboard.py
------------------------
Built-in live HTML performance & drift dashboard served at GET /dashboard.

Auto-refreshes from GET /monitoring/summary and GET /metadata with zero
external CDN dependencies, so it works both locally and on Render PaaS.
"""

from __future__ import annotations

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AQI MLOps — Live Performance &amp; Drift Dashboard</title>
<style>
  :root {
    --bg: #0b1120;
    --panel: #111827;
    --panel-alt: #1e293b;
    --border: #334155;
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --green: #22c55e;
    --amber: #f59e0b;
    --red: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    padding: 1.5rem;
  }
  .wrap { max-width: 1240px; margin: 0 auto; }
  .topbar {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    padding: 1.25rem 1.5rem;
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 1.25rem;
  }
  .topbar h1 { font-size: 1.45rem; font-weight: 700; }
  .topbar .sub { color: #cbd5e1; font-size: 0.9rem; margin-top: 0.2rem; }
  .controls { display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; }
  .pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.03em;
  }
  .pill.green { background: rgba(34, 197, 94, 0.2); color: #86efac; border: 1px solid var(--green); }
  .pill.amber { background: rgba(245, 158, 11, 0.2); color: #fde68a; border: 1px solid var(--amber); }
  .pill.red { background: rgba(239, 68, 68, 0.22); color: #fca5a5; border: 1px solid var(--red); }
  .pill.neutral { background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid var(--border); }
  button.btn {
    background: var(--accent);
    color: #0f172a;
    border: none;
    border-radius: 8px;
    padding: 0.45rem 0.95rem;
    font-weight: 700;
    font-size: 0.85rem;
    cursor: pointer;
  }
  button.btn:hover { opacity: 0.9; }
  a.navlink {
    color: #bae6fd;
    text-decoration: none;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 0.35rem 0.65rem;
    border: 1px solid rgba(186, 230, 253, 0.3);
    border-radius: 6px;
  }
  a.navlink:hover { background: rgba(255, 255, 255, 0.1); }
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 1rem;
    margin-bottom: 1.25rem;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.15rem 1.25rem;
  }
  .kpi-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    font-weight: 600;
  }
  .kpi-val {
    font-size: 1.75rem;
    font-weight: 800;
    margin: 0.35rem 0 0.2rem;
    color: var(--text);
  }
  .kpi-sub { font-size: 0.82rem; color: var(--muted); }
  .grid-2 {
    display: grid;
    grid-template-columns: 1.15fr 1fr;
    gap: 1.25rem;
    margin-bottom: 1.25rem;
  }
  @media (max-width: 920px) { .grid-2 { grid-template-columns: 1fr; } }
  .card h2 {
    font-size: 1.05rem;
    margin-bottom: 0.85rem;
    color: #e2e8f0;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  th, td { padding: 0.6rem 0.65rem; border-bottom: 1px solid var(--border); text-align: left; }
  th { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
  .bar-track {
    width: 100%;
    min-width: 90px;
    height: 10px;
    background: var(--panel-alt);
    border-radius: 999px;
    overflow: hidden;
  }
  .bar-fill { height: 100%; border-radius: 999px; transition: width 0.3s ease; }
  .bar-green { background: var(--green); }
  .bar-amber { background: var(--amber); }
  .bar-red { background: var(--red); }
  .bar-neutral { background: #64748b; }
  .alert-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0.85rem;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    background: var(--panel-alt);
    border-left: 4px solid var(--amber);
    font-size: 0.88rem;
  }
  .alert-item.red { border-left-color: var(--red); }
  .footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.8rem;
    margin-top: 1rem;
  }
</style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div>
      <h1>AQI Awareness Assistant &mdash; Performance &amp; Drift Dashboard</h1>
      <div class="sub" id="model-subtitle">Loading champion metadata...</div>
    </div>
    <div class="controls">
      <span class="pill neutral" id="status-badge">CHECKING</span>
      <button class="btn" onclick="refreshDashboard()">Refresh Now</button>
      <a class="navlink" href="/docs">Swagger /docs</a>
      <a class="navlink" href="/monitoring/summary">JSON Summary</a>
      <a class="navlink" href="/metrics">Prometheus /metrics</a>
    </div>
  </div>

  <div class="kpi-grid">
    <div class="card">
      <div class="kpi-label">Total Predictions Served</div>
      <div class="kpi-val" id="kpi-total">0</div>
      <div class="kpi-sub" id="kpi-oor">Out-of-range: 0</div>
    </div>
    <div class="card">
      <div class="kpi-label">Live Single-Row p95 Latency</div>
      <div class="kpi-val" id="kpi-latency">&mdash;</div>
      <div class="kpi-sub">SLA Budget: &lt; 50.0 ms</div>
    </div>
    <div class="card">
      <div class="kpi-label">Global Rolling 14-Day MAE</div>
      <div class="kpi-val" id="kpi-mae">&mdash;</div>
      <div class="kpi-sub" id="kpi-mae-sub">Alert Threshold: &gt; 19.85 AQI</div>
    </div>
    <div class="card">
      <div class="kpi-label">Max Feature Drift (PSI)</div>
      <div class="kpi-val" id="kpi-psi">&mdash;</div>
      <div class="kpi-sub" id="kpi-pred-psi">Warn &ge; 0.10 &bull; Alert &gt; 0.20</div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>
        <span>Input Feature Drift (Rolling 500-Req PSI)</span>
        <span class="pill neutral" id="pred-psi-pill">Output PSI: &mdash;</span>
      </h2>
      <table>
        <thead>
          <tr>
            <th>Feature</th>
            <th>PSI</th>
            <th>Gauge (0 &rarr; 0.40)</th>
            <th>Z-Shift</th>
            <th>Window</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody id="drift-tbody">
          <tr><td colspan="6" style="color: var(--muted);">Waiting for predictions...</td></tr>
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2>
        <span>Per-City Realized 14-Day MAE vs. Holdout</span>
        <span class="pill neutral" id="mae-limit-pill">Limit: 19.85</span>
      </h2>
      <table>
        <thead>
          <tr>
            <th>City</th>
            <th>Live 14d MAE</th>
            <th>Obs</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody id="city-tbody">
          <tr><td colspan="4" style="color: var(--muted);">No realized pairs yet (send /predict or /feedback)</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <h2>
      <span>Active Monitoring Alerts &amp; Retraining Signal</span>
      <span class="pill neutral" id="retrain-pill">Retrain: No</span>
    </h2>
    <div id="alerts-box" style="color: var(--muted); font-size: 0.9rem;">No active alerts.</div>
  </div>

  <div class="footer" id="last-updated">Auto-refresh every 10 seconds &bull; Powered by FastAPI + Prometheus</div>
</div>

<script>
const CITIES = ["Delhi", "Mumbai", "Bengaluru", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad", "Lucknow"];

function statusClass(status) {
  if (status === "RED") return "red";
  if (status === "AMBER") return "amber";
  if (status === "GREEN") return "green";
  return "neutral";
}

function fmt(val, digits = 3) {
  return (val === null || val === undefined || Number.isNaN(val)) ? "\u2014" : Number(val).toFixed(digits);
}

async function refreshDashboard() {
  try {
    const [sumResp, metaResp] = await Promise.all([
      fetch("/monitoring/summary"),
      fetch("/metadata").catch(() => null),
    ]);
    const summary = await sumResp.json();
    const meta = (metaResp && metaResp.ok) ? await metaResp.json() : null;

    if (meta) {
      document.getElementById("model-subtitle").textContent =
        `Model: ${meta.model_name} v${meta.model_version} (@${meta.alias}) \u2022 ` +
        `Features: ${meta.features_count} \u2022 Val MAE: ${fmt(meta.val_metrics?.mae, 2)} \u2022 ` +
        `Test MAE: ${fmt(meta.test_metrics?.mae, 2)} \u2022 Test R\u00b2: ${fmt(meta.test_metrics?.r2, 4)}`;
    } else {
      document.getElementById("model-subtitle").textContent = "Service readiness: " + (summary.ready ? "READY" : "DEGRADED");
    }

    const badge = document.getElementById("status-badge");
    if (!summary.ready) {
      badge.className = "pill red";
      badge.textContent = "NOT READY";
    } else if (summary.retrain_recommended) {
      badge.className = "pill red";
      badge.textContent = "RETRAIN REQUIRED";
    } else if ((summary.active_alerts || []).length > 0) {
      badge.className = "pill amber";
      badge.textContent = "DRIFT WARNING";
    } else {
      badge.className = "pill green";
      badge.textContent = "HEALTHY";
    }

    document.getElementById("kpi-total").textContent = summary.total_predictions ?? 0;
    document.getElementById("kpi-oor").textContent = `Out-of-range predictions: ${summary.out_of_range_predictions ?? 0}`;
    document.getElementById("kpi-latency").textContent =
      summary.p95_latency_ms !== null && summary.p95_latency_ms !== undefined
        ? `${fmt(summary.p95_latency_ms, 1)} ms`
        : "\u2014";
    document.getElementById("kpi-mae").textContent =
      summary.rolling_14d_mae !== null && summary.rolling_14d_mae !== undefined
        ? `${fmt(summary.rolling_14d_mae, 2)} AQI`
        : "\u2014";
    const maeLimit = summary.mae_alert_threshold ?? 19.85;
    document.getElementById("kpi-mae-sub").textContent = `Baseline: ${fmt(summary.val_mae_baseline, 2)} \u2022 Alert > ${fmt(maeLimit, 2)}`;
    document.getElementById("mae-limit-pill").textContent = `Limit: ${fmt(maeLimit, 2)}`;

    document.getElementById("kpi-psi").textContent = fmt(summary.max_feature_psi, 3);
    const predPsi = summary.prediction_drift?.psi;
    const predStatus = summary.prediction_drift?.status;
    const predPill = document.getElementById("pred-psi-pill");
    predPill.className = `pill ${statusClass(predStatus)}`;
    predPill.textContent = `Output PSI: ${fmt(predPsi, 3)}`;

    const driftBody = document.getElementById("drift-tbody");
    const features = summary.feature_drift || {};
    driftBody.innerHTML = Object.entries(features).map(([name, info]) => {
      const psi = info.psi;
      const pct = psi !== null && psi !== undefined ? Math.min(100, (psi / 0.4) * 100) : 0;
      const cls = statusClass(info.status);
      return `<tr>
        <td><code>${name}</code></td>
        <td>${fmt(psi, 3)}</td>
        <td><div class="bar-track"><div class="bar-fill bar-${cls}" style="width:${pct.toFixed(1)}%"></div></div></td>
        <td>${fmt(info.z_score, 2)}</td>
        <td>${info.window_size ?? 0}</td>
        <td><span class="pill ${cls}">${info.status ?? "WARMING"}</span></td>
      </tr>`;
    }).join("");

    const cityBody = document.getElementById("city-tbody");
    const perCity = summary.per_city_mae || {};
    const breached = new Set(summary.breached_cities || []);
    cityBody.innerHTML = CITIES.map((city) => {
      const row = perCity[city];
      const mae = row ? fmt(row.mae, 2) : "\u2014";
      const obs = row ? row.observations : 0;
      const cls = !row ? "neutral" : (breached.has(city) ? "red" : "green");
      const label = !row ? "NO DATA" : (breached.has(city) ? "ALERT" : "OK");
      return `<tr>
        <td><strong>${city}</strong></td>
        <td>${mae}</td>
        <td>${obs}</td>
        <td><span class="pill ${cls}">${label}</span></td>
      </tr>`;
    }).join("");

    const retrainPill = document.getElementById("retrain-pill");
    retrainPill.className = summary.retrain_recommended ? "pill red" : "pill green";
    retrainPill.textContent = summary.retrain_recommended ? "Retrain: RECOMMENDED" : "Retrain: Not Needed";

    const alertsBox = document.getElementById("alerts-box");
    const alerts = summary.active_alerts || [];
    if (alerts.length === 0) {
      alertsBox.innerHTML = `<span style="color: #86efac;">All monitored features and city MAE tracks are within acceptable bounds.</span>`;
    } else {
      alertsBox.innerHTML = alerts.map((a) => {
        const isRed = a.severity === "RED";
        const target = a.feature || a.city || a.scope || "prediction_output";
        return `<div class="alert-item ${isRed ? "red" : ""}">
          <span><strong>${a.type}</strong> &mdash; <code>${target}</code></span>
          <span class="pill ${isRed ? "red" : "amber"}">${a.severity}</span>
        </div>`;
      }).join("");
    }

    document.getElementById("last-updated").textContent =
      `Last updated: ${new Date().toLocaleTimeString()} \u2022 Auto-refresh every 10s`;
  } catch (err) {
    document.getElementById("last-updated").textContent = `Refresh error: ${err}`;
  }
}

refreshDashboard();
setInterval(refreshDashboard, 10000);
</script>
</body>
</html>
"""
