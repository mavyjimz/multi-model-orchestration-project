#!/usr/bin/env python3
"""Fix p6.1 validation check for MLflow 2.x API compatibility"""

import sys
from pathlib import Path

script_path = Path('scripts/phase6/p6.1-mlflow-setup.py')
content = script_path.read_text()

# Fix the validation function - replace list_experiments with search_experiments
old_code = '''    # Check 1: Tracking URI accessible
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        client.list_experiments()
        checks.append(("Tracking URI accessible", True))
    except Exception as e:
        checks.append(("Tracking URI accessible", False, str(e)))'''

new_code = '''    # Check 1: Tracking URI accessible
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        client.search_experiments()
        checks.append(("Tracking URI accessible", True))
    except Exception as e:
        checks.append(("Tracking URI accessible", False, str(e)))'''

content = content.replace(old_code, new_code)
script_path.write_text(content)

print("Validation function fixed for MLflow 2.x API")
