"""Tests for idea log REST API endpoints."""
import pytest
from fastapi.testclient import TestClient

from src.api.idea_log_api import app, get_idea_log
from src.memory.idea_log import IdeaLog


@pytest.fixture
def idea_log(tmp_path):
    filepath = str(tmp_path / "test_ideas.json")
    return IdeaLog(filepath=filepath)


@pytest.fixture
def client(idea_log):
    """Create a test client with injected IdeaLog."""
    app.dependency_overrides[get_idea_log] = lambda: idea_log
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestIdeaLogAPI:
    """Tests for idea log REST endpoints."""

    def test_post_idea(self, client):
        """POST /ideas should create an idea and return 201."""
        response = client.post("/ideas", json={"text": "Use LQ loss for image tasks"})
        assert response.status_code == 201
        data = response.json()
        assert data["id"]
        assert data["text"] == "Use LQ loss for image tasks"
        assert "tags" in data
        assert "created_at" in data

    def test_get_ideas(self, client):
        """GET /ideas should list all ideas."""
        client.post("/ideas", json={"text": "Idea 1"})
        client.post("/ideas", json={"text": "Idea 2"})
        response = client.get("/ideas")
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 2

    def test_search_ideas_by_query(self, client):
        """GET /ideas?q=LQ should return matching ideas."""
        client.post("/ideas", json={"text": "Use LQ loss for image tasks"})
        client.post("/ideas", json={"text": "Try contrastive learning"})
        response = client.get("/ideas?q=LQ")
        assert response.status_code == 200
        data = response.json()
        assert len(data["ideas"]) == 1
        assert "LQ" in data["ideas"][0]["text"]

    def test_search_ideas_by_tag(self, client):
        """GET /ideas?tag=NLP should return tag-filtered ideas."""
        client.post("/ideas", json={"text": "NLP idea", "tags": ["NLP"]})
        client.post("/ideas", json={"text": "CV idea", "tags": ["CV"]})
        response = client.get("/ideas?tag=NLP")
        assert response.status_code == 200
        data = response.json()
        assert len(data["ideas"]) == 1

    def test_put_idea(self, client):
        """PUT /ideas/{id} should update the idea."""
        create = client.post("/ideas", json={"text": "Original"})
        idea_id = create.json()["id"]
        response = client.put(f"/ideas/{idea_id}", json={"text": "Updated"})
        assert response.status_code == 200
        assert response.json()["text"] == "Updated"

    def test_put_idea_not_found(self, client):
        """PUT /ideas/{id} with invalid ID should return 404."""
        response = client.put("/ideas/bad-id", json={"text": "test"})
        assert response.status_code == 404

    def test_delete_idea(self, client):
        """DELETE /ideas/{id} should remove the idea."""
        create = client.post("/ideas", json={"text": "To delete"})
        idea_id = create.json()["id"]
        response = client.delete(f"/ideas/{idea_id}")
        assert response.status_code == 200
        get_resp = client.get("/ideas")
        assert get_resp.json()["total_count"] == 0

    def test_delete_idea_not_found(self, client):
        """DELETE /ideas/{id} with invalid ID should return 404."""
        response = client.delete("/ideas/bad-id")
        assert response.status_code == 404
