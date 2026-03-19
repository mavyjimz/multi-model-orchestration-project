from datetime import datetime

import pytest
from pydantic import ValidationError

from src.registry.schemas import (
    ModelRegisterRequest,
    RegistryHealthResponse,
)


def test_model_register_request_valid():
    """Test valid ModelRegisterRequest creation"""
    req = ModelRegisterRequest(
        name="test-model",
        version="1.0.0",
        source_path="/tmp/model",
        run_id="run-123",
        description="Test"
    )
    assert req.name == "test-model"
    assert req.version == "1.0.0"

def test_model_register_request_missing_fields():
    """Test ModelRegisterRequest validation on missing required fields"""
    with pytest.raises(ValidationError):
        ModelRegisterRequest(name="test")

def test_registry_health_response():
    """Test RegistryHealthResponse schema"""
    response = RegistryHealthResponse(
        status="healthy",
        service="model-registry",
        mlflow_connected=True,
        timestamp=datetime.utcnow()
    )
    assert response.status == "healthy"
    assert response.mlflow_connected is True
