"""Tests for API health and root endpoints."""

from fastapi.testclient import TestClient
from backend.main import app
from backend.config import get_settings

client = TestClient(app)
settings = get_settings()


def test_root_endpoint() -> None:
    """Verify that root endpoint GET / returns 200 and valid metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert settings.app_name in data["message"]
    assert data["status"] == "running"
    assert data["version"] == "1.0.0"


def test_health_endpoint() -> None:
    """Verify that health endpoint GET /health returns 200 and 'healthy' status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == settings.app_name
    assert data["environment"] == settings.app_env
