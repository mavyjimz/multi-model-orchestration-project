"""
Registry module configuration - iR&D Dev Mode Support
"""

import os
from pathlib import Path

# MLflow Tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# Registry API settings
API_HOST = os.getenv("REGISTRY_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("REGISTRY_API_PORT", "8000"))

# iR&D Dev Mode: Bypass strict MLflow source validation for local testing
# ⚠️ Production: Set REGISTRY_DEV_MODE=false or omit entirely
REGISTRY_DEV_MODE = os.getenv("REGISTRY_DEV_MODE", "true").lower() == "true"

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
DB_PATH = PROJECT_ROOT / "mlflow.db"


def get_mlflow_tracking_uri() -> str:
    """Return MLflow tracking URI with proper resolution"""
    uri = MLFLOW_TRACKING_URI
    if uri.startswith("sqlite") and "///" in uri:
        db_path = Path(uri.replace("sqlite:///", ""))
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path
            uri = f"sqlite:///{db_path}"
    return uri


def is_dev_mode() -> bool:
    """Check if dev mode is enabled"""
    return REGISTRY_DEV_MODE
