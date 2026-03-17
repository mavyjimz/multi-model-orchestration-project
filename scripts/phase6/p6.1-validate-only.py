#!/usr/bin/env python3
"""Quick validation check for p6.1 setup"""

import sys
import json
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).parent.parent.parent
MLFLOWS_DIR = PROJECT_ROOT / 'mlruns'
CONFIG_DIR = PROJECT_ROOT / 'config' / 'registry'
MLFLOW_TRACKING_URI = f"file:{MLFLOWS_DIR.resolve()}"
EXPERIMENT_NAME = "multi-model-orchestration"
MODEL_NAME = "intent-classifier-sgd"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

print("=" * 60)
print("Phase 6.1 Setup Validation (Fixed)")
print("=" * 60)

checks = []

# Check 1: Tracking URI accessible (FIXED)
try:
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

# Check 5: MLruns directory has content
mlruns_files = list(MLFLOWS_DIR.glob('*'))
checks.append(("MLflow data directory populated", len(mlruns_files) > 0))

# Results
all_passed = True
for check in checks:
    status = "PASS" if check[1] else "FAIL"
    if len(check) == 3:
        print(f"[{status}] {check[0]}: {check[2]}")
    else:
        print(f"[{status}] {check[0]}")
    if not check[1]:
        all_passed = False

print("=" * 60)
if all_passed:
    print("Phase 6.1 Setup: SUCCESS - All validation checks passed")
else:
    print("Phase 6.1 Setup: PARTIAL - Review failed checks")
print("=" * 60)

# Update summary
summary_path = PROJECT_ROOT / 'results' / 'phase6' / 'p6.1_setup_summary.json'
if summary_path.exists():
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    summary['validation_passed'] = all_passed
    summary['checks'] = checks
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Updated summary: {summary_path}")

sys.exit(0 if all_passed else 1)
