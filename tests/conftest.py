"""Pytest fixtures for API testing"""

import pytest
from fastapi.testclient import TestClient

from src.registry.api import app


@pytest.fixture
def client():
    """Create a test client for API tests"""
    # Return client directly (avoid context manager compatibility issues)
    return TestClient(app)
