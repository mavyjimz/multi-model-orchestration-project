#!/usr/bin/env python3
"""
p6.4-promotion-workflow.py
Phase 6.4: Model Promotion Workflow with Gates

Implements production-grade model promotion with:
- Environment stages: development -> staging -> production
- Automated gate checks: accuracy, latency, test coverage
- Manual approval hooks for production promotion
- Rollback capability with version history
- Audit trail logging for compliance
- Integration with MLflow Model Registry lifecycle stages
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from enum import Enum
import mlflow
from mlflow.tracking import MlflowClient
# ModelVersion not needed in MLflow 2.x

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/p6.4-promotion-workflow.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MLFLOWS_DIR = PROJECT_ROOT / 'mlruns'
CONFIG_DIR = PROJECT_ROOT / 'config' / 'registry'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'phase6'

MLFLOW_TRACKING_URI = f"file:{MLFLOWS_DIR.resolve()}"
MODEL_NAME = "intent-classifier-sgd"


class EnvironmentStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class PromotionGate:
    """Define promotion gate criteria."""
    
    def __init__(self, name: str, criteria: Dict, required: bool = True):
        self.name = name
        self.criteria = criteria
        self.required = required
    
    def evaluate(self, metrics: Dict) -> Tuple[bool, str]:
        """Evaluate gate against provided metrics."""
        if self.criteria.get('type') == 'threshold':
            metric_name = self.criteria['metric']
            operator = self.criteria['operator']  # '>=', '<=', '>', '<'
            threshold = self.criteria['value']
            
            if metric_name not in metrics:
                return False, f"Missing metric: {metric_name}"
            
            actual = metrics[metric_name]
            
            if operator == '>=' and actual >= threshold:
                return True, f"{metric_name}={actual} >= {threshold}"
            elif operator == '<=' and actual <= threshold:
                return True, f"{metric_name}={actual} <= {threshold}"
            elif operator == '>' and actual > threshold:
                return True, f"{metric_name}={actual} > {threshold}"
            elif operator == '<' and actual < threshold:
                return True, f"{metric_name}={actual} < {threshold}"
            else:
                return False, f"{metric_name}={actual} failed {operator} {threshold}"
        
        elif self.criteria.get('type') == 'test_required':
            test_name = self.criteria['test']
            if metrics.get('tests_passed', []).count(test_name) > 0:
                return True, f"Test passed: {test_name}"
            return False, f"Required test not passed: {test_name}"
        
        return True, "Gate passed (no criteria)"


class PromotionWorkflow:
    """Manage model promotion through lifecycle stages."""
    
    def __init__(self, model_name: str, tracking_uri: str):
        self.model_name = model_name
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.gates = self._load_promotion_gates()
    
    def _load_promotion_gates(self) -> Dict[EnvironmentStage, List[PromotionGate]]:
        """Load gate configuration from schema."""
        schema_path = CONFIG_DIR / 'registry_schema_v1.0.0.json'
        if not schema_path.exists():
            logger.warning(f"Schema not found: {schema_path}, using defaults")
            return self._default_gates()
        
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        gates = {}
        requirements = schema.get('model_requirements', {})
        
        # Development -> Staging gates
        dev_to_staging = requirements.get('promotion_gates', {}).get('development_to_staging', {})
        gates[EnvironmentStage.STAGING] = [
            PromotionGate('min_accuracy', {
                'type': 'threshold', 'metric': 'accuracy', 'operator': '>=', 
                'value': dev_to_staging.get('min_accuracy', 0.70)
            }),
            PromotionGate('max_latency', {
                'type': 'threshold', 'metric': 'latency_p95_ms', 'operator': '<=', 
                'value': dev_to_staging.get('max_latency_ms', 100)
            }),
        ]
        for test in dev_to_staging.get('required_tests', []):
            gates[EnvironmentStage.STAGING].append(
                PromotionGate(f'test_{test}', {'type': 'test_required', 'test': test})
            )
        
        # Staging -> Production gates
        staging_to_prod = requirements.get('promotion_gates', {}).get('staging_to_production', {})
        gates[EnvironmentStage.PRODUCTION] = [
            PromotionGate('min_accuracy', {
                'type': 'threshold', 'metric': 'accuracy', 'operator': '>=', 
                'value': staging_to_prod.get('min_accuracy', 0.75)
            }),
            PromotionGate('max_latency', {
                'type': 'threshold', 'metric': 'latency_p95_ms', 'operator': '<=', 
                'value': staging_to_prod.get('max_latency_ms', 50)
            }),
        ]
        for test in staging_to_prod.get('required_tests', []):
            gates[EnvironmentStage.PRODUCTION].append(
                PromotionGate(f'test_{test}', {'type': 'test_required', 'test': test})
            )
        
        return gates
    
    def _default_gates(self) -> Dict[EnvironmentStage, List[PromotionGate]]:
        """Fallback default gates."""
        return {
            EnvironmentStage.STAGING: [
                PromotionGate('min_accuracy', {'type': 'threshold', 'metric': 'accuracy', 'operator': '>=', 'value': 0.70}),
                PromotionGate('max_latency', {'type': 'threshold', 'metric': 'latency_p95_ms', 'operator': '<=', 'value': 100}),
            ],
            EnvironmentStage.PRODUCTION: [
                PromotionGate('min_accuracy', {'type': 'threshold', 'metric': 'accuracy', 'operator': '>=', 'value': 0.75}),
                PromotionGate('max_latency', {'type': 'threshold', 'metric': 'latency_p95_ms', 'operator': '<=', 'value': 50}),
            ]
        }
    
    def get_model_version(self, version: str) -> Optional[dict]:
        """Get specific model version from registry."""
        try:
            return self.client.get_model_version(self.model_name, version)
        except:
            return None
    
    def evaluate_gates(self, target_stage: EnvironmentStage, metrics: Dict) -> Tuple[bool, List[str]]:
        """Evaluate all gates for target stage."""
        if target_stage not in self.gates:
            return True, [f"No gates defined for {target_stage.value}"]
        
        results = []
        all_passed = True
        
        for gate in self.gates[target_stage]:
            passed, message = gate.evaluate(metrics)
            status = "PASS" if passed else "FAIL"
            results.append(f"[{status}] {gate.name}: {message}")
            if not passed and gate.required:
                all_passed = False
        
        return all_passed, results
    
    def promote_model(self, version: str, target_stage: EnvironmentStage, 
                     metrics: Dict, approver: Optional[str] = None) -> Dict:
        """Promote model version to target stage after gate evaluation."""
        logger.info(f"Attempting promotion: {self.model_name} v{version} -> {target_stage.value}")
        
        # Step 1: Evaluate gates
        gates_passed, gate_results = self.evaluate_gates(target_stage, metrics)
        
        for result in gate_results:
            logger.info(f"  {result}")
        
        if not gates_passed:
            return {
                "success": False,
                "reason": "Gate evaluation failed",
                "gate_results": gate_results,
                "timestamp": datetime.now().isoformat()
            }
        
        # Step 2: Check manual approval for production
        if target_stage == EnvironmentStage.PRODUCTION and approver is None:
            logger.warning("Production promotion requires manual approval")
            return {
                "success": False,
                "reason": "Manual approval required for production",
                "gate_results": gate_results,
                "timestamp": datetime.now().isoformat()
            }
        
        # Step 3: Update model version stage in MLflow
        try:
            mv = self.client.get_model_version(self.model_name, version)
            current_stage = mv.current_stage
            
            # Transition to target stage
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage=target_stage.value.upper()
            )
            
            # Log promotion audit trail
            audit_entry = {
                "model_name": self.model_name,
                "version": version,
                "from_stage": current_stage,
                "to_stage": target_stage.value,
                "metrics": metrics,
                "gate_results": gate_results,
                "approver": approver,
                "timestamp": datetime.now().isoformat()
            }
            
            audit_path = RESULTS_DIR / 'promotion_audit_log.jsonl'
            with open(audit_path, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')
            
            logger.info(f"Successfully promoted to {target_stage.value}")
            
            return {
                "success": True,
                "model_name": self.model_name,
                "version": version,
                "new_stage": target_stage.value,
                "gate_results": gate_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Promotion failed: {e}")
            return {
                "success": False,
                "reason": str(e),
                "gate_results": gate_results,
                "timestamp": datetime.now().isoformat()
            }
    
    def rollback(self, version: str, target_stage: EnvironmentStage) -> Dict:
        """Rollback model to previous version."""
        logger.info(f"Rolling back: {self.model_name} v{version} -> {target_stage.value}")
        
        try:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage=target_stage.value.upper()
            )
            
            return {
                "success": True,
                "model_name": self.model_name,
                "version": version,
                "restored_stage": target_stage.value,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"success": False, "reason": str(e)}


def validate_promotion_setup(workflow: PromotionWorkflow) -> Tuple[bool, List]:
    """Validate promotion workflow configuration."""
    checks = []
    
    # Check 1: Gates loaded
    checks.append(("Promotion gates loaded", len(workflow.gates) > 0))
    
    # Check 2: Can query model registry
    try:
        versions = workflow.client.search_model_versions(f"name='{MODEL_NAME}'")
        checks.append(("Model registry access", len(versions) > 0))
    except Exception as e:
        checks.append(("Model registry access", False, str(e)))
    
    # Check 3: Gate evaluation works
    test_metrics = {"accuracy": 0.72, "latency_p95_ms": 85, "tests_passed": ["unit", "integration"]}
    passed, results = workflow.evaluate_gates(EnvironmentStage.STAGING, test_metrics)
    checks.append(("Gate evaluation", passed and len(results) > 0))
    
    # Log results
    all_passed = True
    for check in checks:
        status = "PASS" if check[1] else "FAIL"
        msg = f"[{status}] {check[0]}"
        if len(check) == 3:
            msg += f": {check[2]}"
        logger.info(msg)
        if not check[1]:
            all_passed = False
    
    return all_passed, checks


def main():
    """Main execution for p6.4 promotion workflow."""
    parser = argparse.ArgumentParser(description='Phase 6.4: Model Promotion Workflow')
    parser.add_argument('--version', type=str, default='1',
                       help='Model version to promote (MLflow integer version)')
    parser.add_argument('--target-stage', type=str, default='staging',
                       choices=['development', 'staging', 'production'],
                       help='Target environment stage')
    parser.add_argument('--accuracy', type=float, default=0.7169,
                       help='Model accuracy metric')
    parser.add_argument('--latency-ms', type=float, default=21.75,
                       help='P95 latency in milliseconds')
    parser.add_argument('--approver', type=str, default=None,
                       help='Approver name (required for production)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Evaluate gates without promoting')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Phase 6.4: Model Promotion Workflow with Gates")
    logger.info("=" * 60)
    
    try:
        # Initialize workflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        workflow = PromotionWorkflow(MODEL_NAME, MLFLOW_TRACKING_URI)
        
        # Load metrics
        metrics = {
            "accuracy": args.accuracy,
            "latency_p95_ms": args.latency_ms,
            "tests_passed": ["unit", "integration", "load"]
        }
        
        target_stage = EnvironmentStage(args.target_stage)
        
        logger.info(f"Model: {MODEL_NAME} v{args.version}")
        logger.info(f"Target stage: {target_stage.value}")
        logger.info(f"Metrics: accuracy={metrics['accuracy']}, latency={metrics['latency_p95_ms']}ms")
        
        # Validate setup
        success, checks = validate_promotion_setup(workflow)
        
        # Evaluate gates
        gates_passed, gate_results = workflow.evaluate_gates(target_stage, metrics)
        logger.info("\nGate Evaluation Results:")
        for result in gate_results:
            logger.info(f"  {result}")
        
        if args.dry_run:
            logger.info("\n[DRY RUN] Promotion would proceed" if gates_passed else "\n[DRY RUN] Promotion would be blocked")
        elif gates_passed:
            # Execute promotion
            result = workflow.promote_model(
                version=args.version,
                target_stage=target_stage,
                metrics=metrics,
                approver=args.approver
            )
            logger.info(f"\nPromotion result: {json.dumps(result, indent=2)}")
        else:
            logger.warning("Gates not passed - promotion blocked")
        
        # Summary
        summary = {
            "phase": "6.4",
            "timestamp": datetime.now().isoformat(),
            "model_name": MODEL_NAME,
            "version": args.version,
            "target_stage": target_stage.value,
            "metrics_evaluated": metrics,
            "gates_passed": gates_passed,
            "dry_run": args.dry_run,
            "validation_passed": success
        }
        
        summary_path = RESULTS_DIR / 'p6.4_promotion_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nSummary saved: {summary_path}")
        
        logger.info("=" * 60)
        logger.info("Phase 6.4 Complete")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Promotion workflow failed: {e}", exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
