# Multi-stage build for intent-classifier-sgd serving
# Phase 8: Deployment & Serving
# Model: intent-classifier-sgd v1.0.2

# =============================================================================
# Stage 1: Builder - Install dependencies
# =============================================================================
FROM python:3.12-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY configs/ ./configs/

# =============================================================================
# Stage 2: Runtime - Minimal production image
# =============================================================================
FROM python:3.12-slim as runtime

LABEL maintainer="mavyjimz"
LABEL version="1.0.2"
LABEL description="Multi-Model Orchestration - Intent Classifier SGD"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/artifacts/models/intent-classifier-sgd/1.0.2/model \
    VECTOR_INDEX_PATH=/app/artifacts/vector_index/faiss_index_v2.0 \
    CONFIG_PATH=/app/configs/production.yaml \
    MLFLOW_TRACKING_URI=file:/app/mlruns \
    PORT=8000 \
    WORKERS=2

RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app /app/logs /app/mlruns && \
    chown -R appuser:appuser /app

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /install /usr/local

COPY --from=builder --chown=appuser:appuser /build/src ./src
COPY --from=builder --chown=appuser:appuser /build/artifacts ./artifacts
COPY --from=builder --chown=appuser:appuser /build/configs ./configs

COPY --chown=appuser:appuser scripts/phase8/docker/entrypoint.sh /app/entrypoint.sh
COPY --chown=appuser:appuser scripts/phase8/docker/healthcheck.sh /app/healthcheck.sh
RUN chmod +x /app/entrypoint.sh /app/healthcheck.sh

USER appuser

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD /app/healthcheck.sh

ENTRYPOINT ["/app/entrypoint.sh"]

CMD uvicorn src.registry.api:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS}
