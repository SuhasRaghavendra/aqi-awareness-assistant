# ==============================================================================
# Multi-stage Dockerfile for AQI Forecasting Inference Service
# Base: python:3.12-slim-bookworm
# Runs as non-root user (appuser:10001) for strict production security
# ==============================================================================

# Build-time args for image labels
ARG GIT_SHA=unknown
ARG BUNDLE_VERSION=unknown

# ------------------------------------------------------------------------------
# Stage 1: Build wheels from serving requirements (no network in runtime stage)
# ------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir -r requirements-serve.txt -w /build/wheels

# ------------------------------------------------------------------------------
# Stage 2: Minimal runtime image – no curl, no build tools
# ------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runner

# Re-declare ARGs after FROM so they are in scope for LABEL
ARG GIT_SHA=unknown
ARG BUNDLE_VERSION=unknown

# Image labels for traceability
LABEL org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.version="${BUNDLE_VERSION}" \
      org.opencontainers.image.title="aqi-api" \
      org.opencontainers.image.description="AQI Forecasting Inference Service"

# OpenMP runtime required by LightGBM and scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user (UID 10001)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Install wheels built in stage 1 – offline, no index needed
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl && \
    rm -rf /wheels

ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
# Number of Uvicorn workers; default 1 for free-tier (512 MB RAM), set to 2+ elsewhere
ENV WORKERS=1

# Copy application source code, configs, and serving artifacts
COPY --chown=appuser:appgroup src/               /app/src/
COPY --chown=appuser:appgroup configs/           /app/configs/
COPY --chown=appuser:appgroup params.yaml        /app/params.yaml
COPY --chown=appuser:appgroup artifacts/champion/ /app/artifacts/champion/

# Switch to unprivileged non-root user
USER appuser

EXPOSE 8000

# HEALTHCHECK uses Python urllib – slim images have no curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Workers controlled by WORKERS env var (default 1)
CMD ["sh", "-c", "uvicorn src.service.app:app --host 0.0.0.0 --port 8000 --workers ${WORKERS}"]

