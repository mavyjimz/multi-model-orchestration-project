import pytest
import sys
from pathlib import Path

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

@pytest.fixture(scope="session")
def project_root():
    """Return project root path"""
    return PROJECT_ROOT

@pytest.fixture(scope="session")
def registry_src():
    """Return src/registry path"""
    return PROJECT_ROOT / "src" / "registry"

@pytest.fixture
def mock_mlflow_client(mocker):
    """Mock MLflow client for unit tests"""
    mock_client = mocker.MagicMock()
    mock_client.search_registered_models.return_value = []
    mock_client.get_registered_model.side_effect = Exception("Model not found")
    return mock_client

@pytest.fixture
def sample_model_register_request():
    """Sample request data for model registration"""
    return {
        "name": "test-model",
        "version": "1.0.0",
        "source_path": "/tmp/test-model",
        "run_id": "test-run-123",
        "description": "Test model for CI/CD"
    }
