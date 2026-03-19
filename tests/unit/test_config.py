
def test_project_root_config():
    """Test PROJECT_ROOT resolution"""
    from src.registry.config import PROJECT_ROOT
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "src").exists()

def test_mlflow_tracking_uri():
    """Test MLflow tracking URI configuration"""
    from src.registry.config import get_mlflow_tracking_uri
    uri = get_mlflow_tracking_uri()
    assert uri is not None
    assert isinstance(uri, str)
