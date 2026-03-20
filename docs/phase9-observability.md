# Phase 9: Monitoring & Observability

## Overview
Complete observability stack for MLOps system including Prometheus metrics, structured logging, and health checks.

## Components

### Metrics (p9.1-p9.2)
- `/metrics` endpoint for Prometheus scraping
- Custom MLOps metrics: request count, latency, errors, model version

### Dashboard (p9.3)
- Streamlit dashboard at `scripts/phase9/dashboard.py`
- Real-time metrics visualization

### Logging (p9.4-p9.5)
- Structured JSON logs with correlation IDs
- Log aggregation and request tracing

### Alerting (p9.6-p9.7)
- P50/P95/P99 latency tracking
- Error rate alerting rules (>5% threshold)

### Health Checks (p9.8-p9.9)
- Model drift integration (PSI/KS)
- Deep health probes: MLflow, disk, models

### Operations (p9.10-p9.11)
- OpenTelemetry tracing (optional)
- Log retention and rotation policy

## Usage

### View Metrics
```bash
curl http://localhost:8000/metrics
