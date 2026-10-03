"""Offline FastAPI tests."""

from fastapi.testclient import TestClient

from app.main import app


def test_root() -> None:
    """Test the root endpoint."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["version"] == "0.1.0"


def test_health() -> None:
    """Test the health endpoint."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
