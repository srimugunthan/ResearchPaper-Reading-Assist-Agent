"""Tests for memory REST API endpoints."""
import pytest
from fastapi.testclient import TestClient

from src.api.memory_api import app, get_memory
from src.memory.memory import ConversationMemory


@pytest.fixture
def memory(tmp_path):
    filepath = str(tmp_path / "test_memory.json")
    return ConversationMemory(filepath=filepath)


@pytest.fixture
def client(memory):
    """Create a test client with injected ConversationMemory."""
    app.dependency_overrides[get_memory] = lambda: memory
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestMemoryAPI:
    """Tests for memory REST endpoints."""

    def test_get_memory_empty(self, client):
        """GET /memory on empty state returns empty history."""
        response = client.get("/memory")
        assert response.status_code == 200
        data = response.json()
        assert data["history"] == []
        assert data["count"] == 0

    def test_get_memory_with_data(self, client, memory):
        """GET /memory returns stored conversation history."""
        memory.save_exchange("Q1", "A1")
        memory.save_exchange("Q2", "A2")
        response = client.get("/memory")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["history"]) == 2

    def test_delete_memory(self, client, memory):
        """DELETE /memory clears all conversation history."""
        memory.save_exchange("Q1", "A1")
        response = client.delete("/memory")
        assert response.status_code == 200
        # Verify cleared
        get_resp = client.get("/memory")
        assert get_resp.json()["count"] == 0

    def test_delete_memory_already_empty(self, client):
        """DELETE /memory on empty state should still return 200."""
        response = client.delete("/memory")
        assert response.status_code == 200
