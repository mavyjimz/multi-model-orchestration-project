# Phase 7: CI/CD Pipeline Runbook

## Overview
Automated testing, quality gates, containerization, and deployment for MLOps Registry.

## Quick Start

### Run Full Test Suite
pytest tests/ -v --cov=src/registry

### Run Linting
ruff check src/
mypy src/
bandit -r src/

### Build Docker Image (Laptop-Optimized)
docker build -t mlops-registry:latest --target runtime .

### Deploy to Staging
bash scripts/deploy-staging.sh

### Trigger Manual Retraining
bash scripts/retrain-model.sh

## Troubleshooting

### Test Failures
pytest tests/ -v -s
pytest tests/unit/test_api.py::test_health_check -v

### Docker Build Issues
docker builder prune -f
docker build -t mlops-registry:latest --progress=plain .

### Deployment Errors
cat logs/deploy-staging.log
bash scripts/validate-env.sh

## Pipeline Status
| Job | Trigger | Output |
|-----|---------|--------|
| ci-cd.yml | Push/PR | Build + Test + Lint |
| deploy-staging.yml | Main branch | Staging deployment |
| retrain.yml | Weekly/manual | Model retraining |
| observability.yml | Push/PR | Coverage + Security reports |
| security-scan.yml | Weekly | Vulnerability scan |

## Next: Phase 8 (MLflow Integration)
- Connect MLflow tracking server
- Enable model versioning
- Add experiment comparison UI
