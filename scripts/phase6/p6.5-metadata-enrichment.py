#!/usr/bin/env python3
"""
p6.5-metadata-enrichment.py
Phase 6.5: Model Metadata Enrichment Pipeline

Enriches model registry entries with production-grade metadata:
- Training data lineage (DVC version, feature schema, split ratios)
- Hyperparameter provenance (search space, best params, optimization method)
- Performance metrics breakdown (per-class F1, confusion matrix stats)
- Dependency manifest (Python packages, system libs, CUDA version)
- Operational metadata (owner, cost estimates, SLA targets, runbook links)
- Compliance tags (PII handling, data retention, audit requirements)
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/p6.5-metadata-enrichment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MLFLOWS_DIR = PROJECT_ROOT / 'mlruns'
CONFIG_DIR = PROJECT_ROOT / 'config' / 'registry'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'phase6'
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'

MLFLOW_TRACKING_URI = f"file:{MLFLOWS_DIR.resolve()}"
MODEL_NAME = "intent-classifier-sgd"


def get_git_metadata() -> Dict:
    """Extract comprehensive git metadata for lineage."""
    try:
        return {
            "commit_hash": subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
            "short_hash": subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip(),
            "branch": subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip(),
            "remote_url": subprocess.check_output(['git', 'remote', 'get-url', 'origin']).decode().strip(),
            "tags": [t for t in subprocess.check_output(['git', 'tag', '--points-at', 'HEAD']).decode().strip().split('\n') if t],
            "last_commit_msg": subprocess.check_output(['git', 'log', '-1', '--pretty=%B']).decode().strip()[:200],
            "commit_timestamp": subprocess.check_output(['git', 'log', '-1', '--format=%cI']).decode().strip()
        }
    except:
        return {"error": "Could not retrieve git metadata"}


def get_data_lineage() -> Dict:
    """Extract training data lineage from DVC and project structure."""
    lineage = {
        "data_version": "v1.0",
        "total_samples": 4786,
        "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "split_counts": {"train": 3341, "val": 716, "test": 717},
        "feature_engineering": {
            "method": "TF-IDF",
            "max_features": 5000,
            "ngram_range": [1, 2],
            "min_df": 2,
            "vectorizer_path": "data/final/embeddings_v2.0/vectorizer.pkl"
        },
        "intent_classes": 41,
        "class_distribution_file": "results/phase1/class_distribution.json",
        "dvc_tracked": True
    }
    
    # Try to load actual DVC metadata if available
    dvc_file = PROJECT_ROOT / 'data.dvc'
    if dvc_file.exists():
        try:
            with open(dvc_file, 'r') as f:
                lineage["dvc_metadata"] = json.load(f)
        except:
            pass
    
    return lineage


def get_hyperparameter_provenance() -> Dict:
    """Extract hyperparameter search and selection provenance."""
    return {
        "model_type": "SGDClassifier",
        "framework": "scikit-learn==1.3.0",
        "optimization_method": "grid_search",
        "hyperparameters": {
            "loss": "log_loss",
            "penalty": "l2",
            "alpha": 0.0001,
            "max_iter": 1000,
            "tol": 1e-3,
            "random_state": 42,
            "class_weight": "balanced"
        },
        "search_space": {
            "alpha": [0.0001, 0.001, 0.01],
            "penalty": ["l1", "l2", "elasticnet"],
            "loss": ["log_loss", "hinge"]
        },
        "selection_criteria": "highest_validation_accuracy",
        "cv_folds": 5,
        "best_cv_score": 0.7234
    }


def get_performance_breakdown() -> Dict:
    """Load detailed performance metrics from Phase 5 results."""
    metrics = {
        "overall": {
            "accuracy": 0.7169,
            "precision_macro": 0.7245,
            "recall_macro": 0.7103,
            "f1_macro": 0.7156,
            "latency_p50_ms": 1.72,
            "latency_p95_ms": 21.75,
            "latency_p99_ms": 45.30
        },
        "per_class_file": "results/phase5/per_class_metrics_v1.0.1.csv",
        "confusion_matrix_file": "results/phase5/confusion_matrix_v1.0.1.png",
        "weak_classes": ["ask_question", "cancel", "maybe"],  # From Phase 5 analysis
        "test_set_size": 717
    }
    
    # Try to load actual metrics if available
    report_path = PROJECT_ROOT / 'results' / 'phase5' / 'validation_report_v1.0.1.json'
    if report_path.exists():
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
                if 'metrics' in report:
                    metrics["overall"].update(report['metrics'])
        except:
            pass
    
    return metrics


def get_dependency_manifest() -> Dict:
    """Generate dependency manifest for reproducibility."""
    try:
        # Get pip freeze output
        pip_freeze = subprocess.check_output(['pip', 'freeze']).decode().strip()
        packages = {}
        for line in pip_freeze.split('\n'):
            if '==' in line:
                pkg, ver = line.split('==')
                packages[pkg] = ver
        
        # Get Python version
        python_version = subprocess.check_output(['python3', '--version']).decode().strip()
        
        # Get system info
        import platform
        system_info = {
            "platform": platform.platform(),
            "python": python_version,
            "cpu_count": os.cpu_count(),
            "memory_gb": round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024**3, 2)
        }
        
        return {
            "core_dependencies": {
                k: v for k, v in packages.items() 
                if k in ['mlflow', 'scikit-learn', 'pandas', 'numpy', 'fastapi', 'faiss-cpu']
            },
            "all_dependencies_count": len(packages),
            "system_info": system_info,
            "requirements_file": "requirements.txt",
            "environment_file": "environment.yaml"
        }
    except Exception as e:
        logger.warning(f"Could not generate full dependency manifest: {e}")
        return {"error": str(e)}


def get_operational_metadata() -> Dict:
    """Define operational metadata for production readiness."""
    return {
        "ownership": {
            "team": "mavyjimz-mlops",
            "owner": "Vanjunn Pongasi",
            "contact": "vanjunn.pongasi.mavy@gmail.com",
            "slack_channel": "#mlops-alerts"
        },
        "sla_targets": {
            "availability": "99.9%",
            "p95_latency_ms": 100,
            "max_error_rate": 0.01,
            "recovery_time_objective_minutes": 15
        },
        "cost_estimates": {
            "inference_cost_per_1k_requests_usd": 0.002,
            "monthly_budget_usd": 50,
            "scaling_trigger_cpu_percent": 70
        },
        "runbook_links": {
            "deployment": "docs/runbooks/deployment.md",
            "rollback": "docs/runbooks/rollback.md",
            "incident_response": "docs/runbooks/incident-response.md"
        },
        "monitoring": {
            "metrics_endpoint": "/metrics",
            "health_endpoint": "/health",
            "alert_rules": "config/alerts/prometheus-rules.yaml"
        }
    }


def get_compliance_tags() -> Dict:
    """Define compliance and governance metadata."""
    return {
        "data_classification": "internal",
        "pii_handling": {
            "contains_pii": False,
            "anonymization_applied": True,
            "retention_days": 365
        },
        "audit_requirements": {
            "log_all_predictions": True,
            "log_model_access": True,
            "retention_days": 90
        },
        "regulatory_tags": ["internal-use", "ml-experimental"],
        "approval_status": "staging-approved",
        "next_review_date": (datetime.now().replace(month=datetime.now().month + 3)).strftime('%Y-%m-%d')
    }


def enrich_model_version(client: MlflowClient, version: str, enrichment: Dict) -> bool:
    """Apply enriched metadata to model version in MLflow."""
    try:
        # Log as tags (MLflow registry metadata)
        for category, data in enrichment.items():
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (str, int, float, bool)):
                        tag_key = f"{category}.{key}"
                        client.set_model_version_tag(
                            name=MODEL_NAME,
                            version=version,
                            key=tag_key,
                            value=str(value)[:500]  # MLflow tag value limit
                        )
        
        # Also log to the associated run for full metadata
        mv = client.get_model_version(MODEL_NAME, version)
        if mv.run_id:
            with mlflow.start_run(run_id=mv.run_id):
                mlflow.log_params({
                    f"enrichment.{k}": str(v)[:500] 
                    for k, v in enrichment.items() if isinstance(v, (str, int, float))
                })
        
        logger.info(f"Enriched metadata for {MODEL_NAME} v{version}")
        return True
    except Exception as e:
        logger.error(f"Failed to enrich metadata: {e}")
        return False


def generate_enriched_manifest(enrichment: Dict, version: str) -> Path:
    """Generate comprehensive enriched manifest file."""
    manifest = {
        "model_name": MODEL_NAME,
        "version": version,
        "enriched_at": datetime.now().isoformat(),
        "enrichment_schema_version": "1.0.0",
        **enrichment
    }
    
    manifest_path = RESULTS_DIR / f'{MODEL_NAME}_v{version}_enriched_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Enriched manifest saved: {manifest_path}")
    return manifest_path


def validate_enrichment(enrichment: Dict) -> tuple:
    """Validate enrichment pipeline output."""
    checks = []
    required_sections = ['git', 'data_lineage', 'hyperparameters', 'performance', 'dependencies', 'operational', 'compliance']
    
    for section in required_sections:
        checks.append((f"{section} metadata", section in enrichment))
    
    # Check critical fields
    critical = [
        ("git.commit_hash", enrichment.get('git', {}).get('commit_hash')),
        ("data_lineage.intent_classes", enrichment.get('data_lineage', {}).get('intent_classes')),
        ("performance.overall.accuracy", enrichment.get('performance', {}).get('overall', {}).get('accuracy')),
        ("operational.sla_targets.p95_latency_ms", enrichment.get('operational', {}).get('sla_targets', {}).get('p95_latency_ms'))
    ]
    
    for name, value in critical:
        checks.append((f"Critical: {name}", value is not None))
    
    # Log results
    all_passed = True
    for check in checks:
        status = "PASS" if check[1] else "FAIL"
        logger.info(f"[{status}] {check[0]}")
        if not check[1]:
            all_passed = False
    
    return all_passed, checks


def main():
    """Main execution for p6.5 metadata enrichment."""
    parser = argparse.ArgumentParser(description='Phase 6.5: Metadata Enrichment Pipeline')
    parser.add_argument('--version', type=str, default='1',
                       help='Model version to enrich (MLflow integer version)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Generate metadata without applying to registry')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Phase 6.5: Model Metadata Enrichment Pipeline")
    logger.info("=" * 60)
    
    try:
        # Initialize
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        
        # Gather all enrichment data
        enrichment = {
            "git": get_git_metadata(),
            "data_lineage": get_data_lineage(),
            "hyperparameters": get_hyperparameter_provenance(),
            "performance": get_performance_breakdown(),
            "dependencies": get_dependency_manifest(),
            "operational": get_operational_metadata(),
            "compliance": get_compliance_tags()
        }
        
        logger.info(f"Collected enrichment data for {MODEL_NAME} v{args.version}")
        
        # Validate enrichment
        valid, checks = validate_enrichment(enrichment)
        
        # Generate manifest file
        manifest_path = generate_enriched_manifest(enrichment, args.version)
        
        # Apply to registry (unless dry-run)
        if args.dry_run:
            logger.info("[DRY RUN] Would apply enriched metadata to MLflow registry")
        else:
            success = enrich_model_version(client, args.version, enrichment)
            if success:
                logger.info("Metadata applied to MLflow Model Registry")
        
        # Summary
        summary = {
            "phase": "6.5",
            "timestamp": datetime.now().isoformat(),
            "model_name": MODEL_NAME,
            "version": args.version,
            "enrichment_sections": list(enrichment.keys()),
            "validation_passed": valid,
            "manifest_path": str(manifest_path),
            "dry_run": args.dry_run
        }
        
        summary_path = RESULTS_DIR / 'p6.5_enrichment_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved: {summary_path}")
        
        logger.info("=" * 60)
        logger.info("Phase 6.5 Complete")
        logger.info("=" * 60)
        
        return 0 if valid else 1
        
    except Exception as e:
        logger.error(f"Enrichment pipeline failed: {e}", exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
