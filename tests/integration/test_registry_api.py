import pytest
from fastapi.testclient import TestClient

from src.registry.api import app

client = TestClient(app)

@pytest.mark.integration
def test_register_model_endpoint_structure():
    """Test /register endpoint exists and accepts POST"""
    response = client.post("/register", json={})
    assert response.status_code in [422, 503]

@pytest.mark.integration
def test_promote_model_endpoint_exists():
    """Test /promote endpoint exists"""
    response = client.post("/promote", json={})
    assert response.status_code in [422, 503]

@pytest.mark.integration
def test_query_model_endpoint_exists():
    """Test /query endpoint exists"""
    response = client.post("/query", json={})
    assert response.status_code in [422, 503]
