# ==============================================================================
# Multi-stage Dockerfile for AQI Forecasting Inference Service
# Base: python:3.12-slim-bookworm
# Runs as non-root user (appuser:10001) for strict production security
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build virtual environment with production serving dependencies
# ------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-serve.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-serve.txt

# ------------------------------------------------------------------------------
# Stage 2: Minimal runtime image
# ------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runner

# Install OpenMP runtime library (required by LightGBM and scikit-learn) and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user (UID 10001)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Copy application source code, configs, and serving artifacts
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup configs/ /app/configs/
COPY --chown=appuser:appgroup params.yaml /app/params.yaml
COPY --chown=appuser:appgroup artifacts/champion/ /app/artifacts/champion/

# Switch to unprivileged non-root user
USER appuser

EXPOSE 8000

# Health check probe against /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production entrypoint with 2 Uvicorn workers
CMD ["uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

