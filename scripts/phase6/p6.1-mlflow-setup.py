#!/usr/bin/env python3
"""
p6.1-mlflow-setup.py
Phase 6.1: MLflow Tracking Server Setup & Schema Standardization

Establishes MLflow tracking infrastructure with:
- Local backend store (mlruns/) for development
- Artifact store configuration (S3-ready path structure)
- Experiment schema for multi-model orchestration
- Model registry initialization with lifecycle stages
- Integration hooks for existing SGD v1.0.1 model
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/p6.1-mlflow-setup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MLFLOWS_DIR = PROJECT_ROOT / 'mlruns'
REGISTRY_DIR = PROJECT_ROOT / 'models' / 'registry' / 'mlflow'
CONFIG_DIR = PROJECT_ROOT / 'config' / 'registry'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'phase6'

# Ensure directories exist
for dir_path in [MLFLOWS_DIR, REGISTRY_DIR, CONFIG_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# MLflow configuration
MLFLOW_TRACKING_URI = f"file:{MLFLOWS_DIR.resolve()}"
EXPERIMENT_NAME = "multi-model-orchestration"
MODEL_NAME = "intent-classifier-sgd"


def setup_mlflow_environment():
    """Configure MLflow tracking URI and environment variables."""
    logger.info(f"Setting MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    logger.info("MLflow environment configured successfully")
    return True


def create_or_get_experiment():
    """Create experiment if not exists, return experiment ID."""
    logger.info(f"Checking for experiment: {EXPERIMENT_NAME}")
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=f"file:{(PROJECT_ROOT / 'artifacts').resolve()}",
            tags={
                "project": "multi-model-orchestration",
                "phase": "6",
                "owner": "mavyjimz",
                "created": datetime.now().isoformat()
            }
        )
        logger.info(f"Created new experiment with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment ID: {experiment_id}")
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment_id


def initialize_model_registry():
    """Initialize MLflow Model Registry with schema validation."""
    logger.info("Initializing Model Registry client")
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    try:
        registered_model = client.get_registered_model(MODEL_NAME)
        logger.info(f"Model '{MODEL_NAME}' already registered")
        return registered_model
    except mlflow.exceptions.MlflowException:
        logger.info(f"Registering new model: {MODEL_NAME}")
        client.create_registered_model(
            name=MODEL_NAME,
            tags={
                "model_type": "text-classifier",
                "framework": "sklearn-sgd",
                "feature_engineering": "tfidf-5000",
                "intent_classes": "41",
                "phase6_registered": datetime.now().isoformat()
            },
            description="SGD-based intent classifier for multi-model orchestration pipeline"
        )
        logger.info(f"Model '{MODEL_NAME}' registered successfully")
        return client.get_registered_model(MODEL_NAME)


def generate_registry_schema_config():
    """Generate schema configuration for registry validation."""
    schema_config = {
        "registry_schema_version": "1.0.0",
        "model_requirements": {
            "required_metadata_fields": [
                "model_type", "framework", "accuracy", "latency_p95",
                "training_data_version", "git_commit", "intent_classes"
            ],
            "promotion_gates": {
                "development_to_staging": {
                    "min_accuracy": 0.70,
                    "max_latency_ms": 100,
                    "required_tests": ["unit", "integration"]
                },
                "staging_to_production": {
                    "min_accuracy": 0.75,
                    "max_latency_ms": 50,
                    "required_tests": ["unit", "integration", "load", "drift"]
                }
            },
            "retention_policy": {
                "development_days": 30,
                "staging_days": 90,
                "production_days": 365,
                "archived_retention": "indefinite"
            }
        },
        "artifact_storage": {
            "local_path": str((PROJECT_ROOT / 'artifacts').resolve()),
            "s3_ready_structure": "s3://bucket-name/models/{model_name}/{version}/",
            "supported_formats": ["joblib", "onnx", "pickle"]
        }
    }
    
    schema_path = CONFIG_DIR / 'registry_schema_v1.0.0.json'
    with open(schema_path, 'w') as f:
        json.dump(schema_config, f, indent=2)
    
    logger.info(f"Registry schema saved to: {schema_path}")
    return schema_path


def validate_setup():
    """Validate MLflow setup and registry configuration."""
    logger.info("Running setup validation checks")
    checks = []
    
    # Check 1: Tracking URI accessible
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        client.search_experiments()
        checks.append(("Tracking URI accessible", True))
    except Exception as e:
        checks.append(("Tracking URI accessible", False, str(e)))
    
    # Check 2: Experiment exists
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    checks.append(("Experiment created", exp is not None))
    
    # Check 3: Model registered
    try:
        client.get_registered_model(MODEL_NAME)
        checks.append(("Model registered", True))
    except:
        checks.append(("Model registered", False))
    
    # Check 4: Schema config exists
    schema_path = CONFIG_DIR / 'registry_schema_v1.0.0.json'
    checks.append(("Schema config generated", schema_path.exists()))
    
    # Log results
    all_passed = True
    for check in checks:
        status = "PASS" if check[1] else "FAIL"
        if len(check) == 3:
            logger.info(f"[{status}] {check[0]}: {check[2]}")
        else:
            logger.info(f"[{status}] {check[0]}")
        if not check[1]:
            all_passed = False
    
    return all_passed, checks


def main():
    """Main execution flow for p6.1 MLflow setup."""
    logger.info("=" * 60)
    logger.info("Phase 6.1: MLflow Tracking Server Setup")
    logger.info("=" * 60)
    
    try:
        # Step 1: Environment setup
        setup_mlflow_environment()
        
        # Step 2: Create/get experiment
        create_or_get_experiment()
        
        # Step 3: Initialize registry
        initialize_model_registry()
        
        # Step 4: Generate schema config
        generate_registry_schema_config()
        
        # Step 5: Validate setup
        success, checks = validate_setup()
        
        # Summary
        logger.info("=" * 60)
        if success:
            logger.info("Phase 6.1 Setup: SUCCESS - All validation checks passed")
        else:
            logger.warning("Phase 6.1 Setup: PARTIAL - Some checks failed (review logs)")
        logger.info("=" * 60)
        
        # Write setup summary
        summary = {
            "phase": "6.1",
            "timestamp": datetime.now().isoformat(),
            "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
            "experiment_name": EXPERIMENT_NAME,
            "model_name": MODEL_NAME,
            "validation_passed": success,
            "checks": checks
        }
        
        summary_path = RESULTS_DIR / 'p6.1_setup_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Setup summary saved: {summary_path}")
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Setup failed with error: {e}", exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
