# Disaster Recovery Runbook

## Scenario 1: API Service Failure
1. Check health endpoint: curl http://localhost:8000/health
2. Restart container: docker-compose restart api
3. If failed, rollback to previous image tag.

## Scenario 2: Model Registry Corruption
1. Stop MLflow service.
2. Restore from backup: tar -xzf backups/mlflow_<timestamp>.tar.gz
3. Restart MLflow service.
4. Verify models: python scripts/phase6/p6.7-registry-api.py list

## Scenario 3: Data Loss
1. Identify last valid backup.
2. Restore data/processed/ from backup.
3. Re-run Phase 1-3 pipelines to regenerate artifacts.
4. Validate model performance before re-deployment.
