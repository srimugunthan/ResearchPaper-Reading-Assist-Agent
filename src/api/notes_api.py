"""REST API for Notes & Insights log."""
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

from src.memory.notes_log import NotesLog

app = FastAPI(title="Notes & Insights API")

_notes_log: Optional[NotesLog] = None


def get_notes_log() -> NotesLog:
    """Dependency: return the global NotesLog instance."""
    global _notes_log
    if _notes_log is None:
        _notes_log = NotesLog(filepath="notes_and_insights.json")
    return _notes_log


class NoteCreate(BaseModel):
    text: str


class NoteUpdate(BaseModel):
    text: str


@app.post("/notes", status_code=201)
def create_note(body: NoteCreate, log: NotesLog = Depends(get_notes_log)):
    """Add a new note."""
    return log.add_note(body.text)


@app.get("/notes")
def list_notes(
    q: Optional[str] = None,
    limit: Optional[int] = None,
    log: NotesLog = Depends(get_notes_log),
):
    """List or search notes."""
    if q:
        notes = log.search_notes(query=q)
        return {"notes": notes, "total_count": len(notes)}
    return log.get_notes(limit=limit)


@app.put("/notes/{note_id}")
def update_note(note_id: str, body: NoteUpdate, log: NotesLog = Depends(get_notes_log)):
    """Edit a note."""
    try:
        return log.edit_note(note_id, new_text=body.text)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Note '{note_id}' not found")


@app.delete("/notes/{note_id}")
def delete_note(note_id: str, log: NotesLog = Depends(get_notes_log)):
    """Delete a note."""
    try:
        log.delete_note(note_id)
        return {"status": "deleted"}
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Note '{note_id}' not found")
