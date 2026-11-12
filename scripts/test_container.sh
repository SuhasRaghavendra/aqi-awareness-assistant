#!/usr/bin/env bash
# ==============================================================================
# scripts/test_container.sh
# Automates Docker build, local run, endpoint smoke tests, and cleanup
# ==============================================================================
set -euo pipefail

IMAGE="aqi-api:test"
CONTAINER="aqi-test-container"

echo "==> 1. Building Docker image..."
docker build -t "$IMAGE" .

echo "==> 2. Starting container..."
docker run -d -p 8000:8000 --name "$CONTAINER" "$IMAGE"

cleanup() {
    echo "==> Cleaning up container..."
    docker stop "$CONTAINER" >/dev/null 2>&1 || true
    docker rm   "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "==> 3. Waiting for /health to become ready (up to 30s)..."
deadline=$(( $(date +%s) + 30 ))
until curl -fsS http://localhost:8000/health >/dev/null 2>&1; do
    if [ "$(date +%s)" -gt "$deadline" ]; then
        echo "ERROR: container did not become healthy within 30s" >&2
        exit 1
    fi
    sleep 2
done
echo "    /health OK"

echo "==> 4. Checking non-root user (must be UID 10001)..."
UID_IN_CONTAINER=$(docker exec "$CONTAINER" id -u)
if [ "$UID_IN_CONTAINER" != "10001" ]; then
    echo "ERROR: container running as UID $UID_IN_CONTAINER, expected 10001" >&2
    exit 1
fi
echo "    UID check passed: running as UID $UID_IN_CONTAINER"

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
    "date": "2025-01-02",
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

echo "==> 7b. Testing POST /predict error contract (422)..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"city": "Delhi", "date": "not-a-date", "aqi": 100.0}')
if [ "$STATUS" != "422" ]; then
    echo "ERROR: expected 422, got $STATUS" >&2
    exit 1
fi
echo "    422 OK"

echo "==> 8. Testing GET /metrics..."
curl -fsS http://localhost:8000/metrics | head -n 10
echo ""

echo "SUCCESS: All container smoke tests passed!"

