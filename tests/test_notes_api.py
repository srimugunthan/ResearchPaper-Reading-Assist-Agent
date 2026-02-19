"""Tests for notes REST API endpoints."""
import pytest
from fastapi.testclient import TestClient

from src.api.notes_api import app, get_notes_log
from src.memory.notes_log import NotesLog


@pytest.fixture
def notes_log(tmp_path):
    filepath = str(tmp_path / "test_notes.json")
    return NotesLog(filepath=filepath)


@pytest.fixture
def client(notes_log):
    """Create a test client with injected NotesLog."""
    app.dependency_overrides[get_notes_log] = lambda: notes_log
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestNotesAPI:
    """Tests for notes REST endpoints."""

    def test_post_note(self, client):
        """POST /notes should create a note and return 201."""
        response = client.post("/notes", json={"text": "Only for symmetric loss, MAE is useful"})
        assert response.status_code == 201
        data = response.json()
        assert data["id"]
        assert data["text"] == "Only for symmetric loss, MAE is useful"
        assert "tags" in data
        assert "created_at" in data

    def test_get_notes(self, client):
        """GET /notes should list all notes."""
        client.post("/notes", json={"text": "Note 1"})
        client.post("/notes", json={"text": "Note 2"})
        response = client.get("/notes")
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 2
        assert len(data["notes"]) == 2

    def test_get_notes_with_limit(self, client):
        """GET /notes?limit=1 should return at most 1 note."""
        client.post("/notes", json={"text": "Note 1"})
        client.post("/notes", json={"text": "Note 2"})
        response = client.get("/notes?limit=1")
        assert response.status_code == 200
        data = response.json()
        assert len(data["notes"]) == 1
        assert data["total_count"] == 2

    def test_search_notes(self, client):
        """GET /notes?q=MAE should return matching notes."""
        client.post("/notes", json={"text": "Only for symmetric loss, MAE is useful"})
        client.post("/notes", json={"text": "Cross-attention is great"})
        response = client.get("/notes?q=MAE")
        assert response.status_code == 200
        data = response.json()
        assert len(data["notes"]) == 1
        assert "MAE" in data["notes"][0]["text"]

    def test_put_note(self, client):
        """PUT /notes/{id} should update the note."""
        create = client.post("/notes", json={"text": "Original"})
        note_id = create.json()["id"]
        response = client.put(f"/notes/{note_id}", json={"text": "Updated"})
        assert response.status_code == 200
        assert response.json()["text"] == "Updated"

    def test_put_note_not_found(self, client):
        """PUT /notes/{id} with invalid ID should return 404."""
        response = client.put("/notes/bad-id", json={"text": "test"})
        assert response.status_code == 404

    def test_delete_note(self, client):
        """DELETE /notes/{id} should remove the note."""
        create = client.post("/notes", json={"text": "To delete"})
        note_id = create.json()["id"]
        response = client.delete(f"/notes/{note_id}")
        assert response.status_code == 200
        # Verify it's gone
        get_resp = client.get("/notes")
        assert get_resp.json()["total_count"] == 0

    def test_delete_note_not_found(self, client):
        """DELETE /notes/{id} with invalid ID should return 404."""
        response = client.delete("/notes/bad-id")
        assert response.status_code == 404
