import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from src.registry.api import app
from src.registry.schemas import RegistryHealthResponse

client = TestClient(app)

def test_health_check_endpoint():
    """Test /health endpoint returns valid response"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "service" in data
    assert data["service"] == "model-registry"

def test_health_response_schema():
    """Test health response matches RegistryHealthResponse schema"""
    response = client.get("/health")
    data = response.json()
    assert all(k in data for k in ["status", "service", "timestamp"])
    assert data["status"] in ["healthy", "degraded"]
