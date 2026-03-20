"""
Enhanced Health Checks - Phase 9.9
Deep probes: MLflow connectivity, disk space, model availability.
"""

import os
import shutil
import mlflow
from pathlib import Path

def check_mlflow_connectivity():
    """Check MLflow tracking server connectivity."""
    try:
        client = mlflow.tracking.MlflowClient()
        client.list_experiments()
        return {'service': 'mlflow', 'status': 'healthy'}
    except Exception as e:
        return {'service': 'mlflow', 'status': 'unhealthy', 'error': str(e)}

def check_disk_space(path=".", threshold_gb=1):
    """Check available disk space."""
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    status = 'healthy' if free_gb > threshold_gb else 'warning'
    return {'service': 'disk', 'status': status, 'free_gb': round(free_gb, 2)}

def check_model_availability(model_path="artifacts/models"):
    """Check if model artifacts are available."""
    path = Path(model_path)
    exists = path.exists()
    model_count = len(list(path.glob("**/*.pkl"))) if exists else 0
    status = 'healthy' if model_count > 0 else 'warning'
    return {'service': 'models', 'status': status, 'model_count': model_count}

def run_all_health_checks():
    """Run all health checks and return results."""
    results = [
        check_mlflow_connectivity(),
        check_disk_space(),
        check_model_availability()
    ]

    print("Health Check Results:")
    for result in results:
        print(f"  [{result['status'].upper()}] {result['service']}")

    all_healthy = all(r['status'] == 'healthy' for r in results)
    return {'all_healthy': all_healthy, 'checks': results}

if __name__ == "__main__":
    run_all_health_checks()
