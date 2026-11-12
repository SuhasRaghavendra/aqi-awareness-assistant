#!/usr/bin/env bash
# ==============================================================================
# scripts/test_container.sh
# Automates Docker build, local run, endpoint smoke tests, and cleanup
# ==============================================================================
set -euo pipefail

echo "==> 1. Building Docker image..."
docker build -t aqi-predictor:test .

echo "==> 2. Starting container..."
docker run -d -p 8000:8000 --name aqi-test-container aqi-predictor:test

cleanup() {
    echo "==> Cleaning up container..."
    docker stop aqi-test-container >/dev/null 2>&1 || true
    docker rm aqi-test-container >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "==> 3. Waiting for container to initialize..."
sleep 5

echo "==> 4. Testing GET /health..."
curl -fsS http://localhost:8000/health | grep -q "healthy"
echo "    /health OK"

echo "==> 5. Testing GET /ready..."
curl -fsS http://localhost:8000/ready
echo ""

echo "==> 6. Testing GET /metadata..."
curl -fsS http://localhost:8000/metadata
echo ""

echo "==> 7. Testing POST /predict..."
curl -fsS -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Delhi",
    "date": "2026-01-15",
    "aqi": 180.0,
    "pm25": 95.0,
    "pm10": 150.0,
    "no2": 55.0,
    "so2": 12.0,
    "co": 1.4,
    "o3": 30.0,
    "temperature_2m_mean": 14.0,
    "wind_speed_10m_mean": 3.5,
    "relative_humidity_2m_mean": 72.0
  }'
echo ""

echo "==> 8. Testing GET /metrics..."
curl -fsS http://localhost:8000/metrics | head -n 10
echo ""

echo "SUCCESS: All container smoke tests passed!"

