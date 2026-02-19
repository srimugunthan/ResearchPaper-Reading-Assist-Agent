"""REST API for conversation memory."""
from typing import Optional

from fastapi import FastAPI, Depends

from src.memory.memory import ConversationMemory

app = FastAPI(title="Memory API")

_memory: Optional[ConversationMemory] = None


def get_memory() -> ConversationMemory:
    """Dependency: return the global ConversationMemory instance."""
    global _memory
    if _memory is None:
        _memory = ConversationMemory(filepath="conversation_memory.json")
    return _memory


@app.get("/memory")
def get_conversation_memory(mem: ConversationMemory = Depends(get_memory)):
    """Get stored conversation history."""
    history = mem.get_history()
    return {"history": history, "count": len(history)}


@app.delete("/memory")
def clear_conversation_memory(mem: ConversationMemory = Depends(get_memory)):
    """Clear all conversation memory."""
    mem.clear()
    return {"status": "cleared"}
