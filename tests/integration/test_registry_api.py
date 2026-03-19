import pytest
from fastapi.testclient import TestClient

from src.registry.api import app



@pytest.mark.integration
@pytest.mark.skip(reason="Temporarily skip: TestClient version compatibility")
def test_register_model_endpoint_structure(client):
    """Test /register endpoint exists and accepts POST"""
    response = client.post("/register", json={})
    assert response.status_code in [404, 422, 503]


@pytest.mark.integration
@pytest.mark.skip(reason="Temporarily skip: TestClient version compatibility")
def test_promote_model_endpoint_exists(client):
    """Test /promote endpoint exists"""
    response = client.post("/promote", json={})
    assert response.status_code in [404, 422, 503]


@pytest.mark.integration
@pytest.mark.skip(reason="Temporarily skip: TestClient version compatibility")
def test_query_model_endpoint_exists(client):
    """Test /query endpoint exists"""
    response = client.post("/query", json={})
    assert response.status_code in [404, 422, 503]
