"""REST API for Idea Log."""
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

from src.memory.idea_log import IdeaLog

app = FastAPI(title="Idea Log API")

_idea_log: Optional[IdeaLog] = None


def get_idea_log() -> IdeaLog:
    """Dependency: return the global IdeaLog instance."""
    global _idea_log
    if _idea_log is None:
        _idea_log = IdeaLog(filepath="idea_log.json")
    return _idea_log


class IdeaCreate(BaseModel):
    text: str
    tags: Optional[List[str]] = None


class IdeaUpdate(BaseModel):
    text: str


@app.post("/ideas", status_code=201)
def create_idea(body: IdeaCreate, log: IdeaLog = Depends(get_idea_log)):
    """Add a new idea."""
    return log.add_idea(body.text, tags=body.tags)


@app.get("/ideas")
def list_ideas(
    q: Optional[str] = None,
    tag: Optional[str] = None,
    limit: Optional[int] = None,
    log: IdeaLog = Depends(get_idea_log),
):
    """List or search ideas."""
    if q or tag:
        ideas = log.search_ideas(query=q, tag=tag)
        return {"ideas": ideas, "total_count": len(ideas)}
    return log.get_ideas(limit=limit)


@app.put("/ideas/{idea_id}")
def update_idea(idea_id: str, body: IdeaUpdate, log: IdeaLog = Depends(get_idea_log)):
    """Edit an idea."""
    try:
        return log.edit_idea(idea_id, new_text=body.text)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Idea '{idea_id}' not found")


@app.delete("/ideas/{idea_id}")
def delete_idea(idea_id: str, log: IdeaLog = Depends(get_idea_log)):
    """Delete an idea."""
    try:
        log.delete_idea(idea_id)
        return {"status": "deleted"}
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Idea '{idea_id}' not found")
