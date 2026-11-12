# ==============================================================================
# scripts/test_container.ps1
# Automates Docker build, local run, endpoint smoke tests, and cleanup
# ==============================================================================

$ErrorActionPreference = "Stop"

# Auto-resolve docker in PATH if installed but not loaded in current terminal session
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    $candidatePaths = @(
        "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin",
        "C:\Program Files\Docker\Docker\resources\bin"
    )
    foreach ($p in $candidatePaths) {
        if (Test-Path "$p\docker.exe") {
            $env:PATH = "$p;$env:PATH"
            break
        }
    }
}

$IMAGE   = "aqi-api:test"
$CONTAINER = "aqi-test-container"

Write-Host "==> 1. Building Docker image..." -ForegroundColor Cyan
docker build -t $IMAGE .

Write-Host "==> 2. Starting container..." -ForegroundColor Cyan
docker run -d -p 8000:8000 --name $CONTAINER $IMAGE | Out-Null

try {
    Write-Host "==> 3. Waiting for /health to become ready (up to 30s)..." -ForegroundColor Cyan
    $deadline = (Get-Date).AddSeconds(30)
    $healthy  = $false
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -ErrorAction Stop
            $healthy = $true
            break
        } catch {
            Start-Sleep -Seconds 2
        }
    }
    if (-not $healthy) { throw "Container did not become healthy within 30s" }
    Write-Host "    /health OK: $($r | ConvertTo-Json -Compress)" -ForegroundColor Green

    Write-Host "==> 4. Checking non-root user (must be UID 10001)..." -ForegroundColor Cyan
    $uid = docker exec $CONTAINER id -u
    if ($uid.Trim() -ne "10001") { throw "Container is running as UID $uid, expected 10001" }
    Write-Host "    UID check passed: running as UID $uid" -ForegroundColor Green

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
    $metrics = Invoke-WebRequest -Uri "http://localhost:8000/metrics" -Method Get -UseBasicParsing
    Write-Host "    Status code: $($metrics.StatusCode)" -ForegroundColor Green

    Write-Host "`nSUCCESS: All container smoke tests passed!" -ForegroundColor Green
}
finally {
    Write-Host "==> Cleaning up container..." -ForegroundColor Yellow
    docker stop $CONTAINER 2>$null
    docker rm   $CONTAINER 2>$null
}

