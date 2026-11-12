# ==============================================================================
# scripts/test_container.ps1
# Automates Docker build, local run, endpoint smoke tests, and cleanup
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "==> 1. Building Docker image..." -ForegroundColor Cyan
docker build -t aqi-predictor:test .

Write-Host "==> 2. Starting container..." -ForegroundColor Cyan
$containerId = docker run -d -p 8000:8000 --name aqi-test-container aqi-predictor:test

try {
    Write-Host "==> 3. Waiting for container to become healthy (up to 15s)..." -ForegroundColor Cyan
    Start-Sleep -Seconds 5

    Write-Host "==> 4. Testing GET /health..." -ForegroundColor Cyan
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
    Write-Host "    Response: $($health | ConvertTo-Json -Compress)" -ForegroundColor Green

    Write-Host "==> 5. Testing GET /ready..." -ForegroundColor Cyan
    $ready = Invoke-RestMethod -Uri "http://localhost:8000/ready" -Method Get
    Write-Host "    Response: $($ready | ConvertTo-Json -Compress)" -ForegroundColor Green

    Write-Host "==> 6. Testing GET /metadata..." -ForegroundColor Cyan
    $meta = Invoke-RestMethod -Uri "http://localhost:8000/metadata" -Method Get
    Write-Host "    Response: $($meta | ConvertTo-Json -Compress)" -ForegroundColor Green

    Write-Host "==> 7. Testing POST /predict..." -ForegroundColor Cyan
    $payload = @{
        city = "Delhi"
        date = "2025-01-02"
        aqi = 180.0
        pm25 = 95.0
        pm10 = 150.0
        no2 = 55.0
        so2 = 12.0
        co = 1.4
        o3 = 30.0
        temperature_2m_mean = 14.0
        wind_speed_10m_mean = 3.5
        relative_humidity_2m_mean = 72.0
    } | ConvertTo-Json

    $pred = Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $payload -ContentType "application/json"
    Write-Host "    Response: $($pred | ConvertTo-Json -Compress)" -ForegroundColor Green

    Write-Host "==> 8. Testing GET /metrics..." -ForegroundColor Cyan
    $metrics = Invoke-WebRequest -Uri "http://localhost:8000/metrics" -Method Get
    Write-Host "    Status code: $($metrics.StatusCode)" -ForegroundColor Green

    Write-Host "`nSUCCESS: All container smoke tests passed!" -ForegroundColor Green
}
finally {
    Write-Host "==> Cleaning up container..." -ForegroundColor Yellow
    docker stop aqi-test-container 2>$null
    docker rm aqi-test-container 2>$null
}

