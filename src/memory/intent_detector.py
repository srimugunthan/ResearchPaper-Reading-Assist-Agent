"""Intent detector for routing chat commands to the right handler."""
import re
from typing import Any, Dict


def detect_intent(user_input: str) -> Dict[str, Any]:
    """Detect user intent from chat input.

    Supported intents:
        - add_idea: User wants to add to the idea log
        - add_note: User wants to add to notes & insights
        - show_ideas: User wants to see their idea log
        - show_notes: User wants to see their notes
        - search_ideas: User wants to search the idea log
        - search_notes: User wants to search notes
        - question: Regular question (default)

    Returns:
        Dict with keys: intent, extracted_text, params
    """
    text = user_input.strip()
    lower = text.lower()

    # --- Add to idea log ---
    # Pattern: "Add to idea log: <text>"
    m = re.match(r"add\s+to\s+idea\s*log\s*:\s*(.+)", lower, re.IGNORECASE)
    if m:
        extracted = text[m.start(1):m.end(1)].strip()
        return {"intent": "add_idea", "extracted_text": extracted, "params": {}}

    # Pattern: "... Add it in idea log" or "... add it to idea log"
    m = re.search(r"\.?\s*add\s+it\s+(?:in|to)\s+idea\s*log\s*$", lower)
    if m:
        # Extract everything before the "add it in idea log" part
        extracted = text[:m.start()].strip()
        # Try to extract the idea content from phrases like "I have this idea about ..."
        idea_match = re.search(r"(?:idea\s+about\s+)(.+)", extracted, re.IGNORECASE)
        if idea_match:
            extracted = idea_match.group(1).strip().rstrip(".")
        return {"intent": "add_idea", "extracted_text": extracted, "params": {}}

    # --- Add to notes ---
    # Pattern: "Add this to notes: <text>" or "Add to notes: <text>"
    m = re.match(r"add\s+(?:this\s+)?to\s+notes\s*:\s*(.+)", lower, re.IGNORECASE)
    if m:
        extracted = text[m.start(1):m.end(1)].strip()
        # Strip surrounding quotes if present
        extracted = extracted.strip("'\"")
        return {"intent": "add_note", "extracted_text": extracted, "params": {}}

    # Pattern: "Note this down: <text>"
    m = re.match(r"note\s+this\s+down\s*:\s*(.+)", lower, re.IGNORECASE)
    if m:
        extracted = text[m.start(1):m.end(1)].strip()
        extracted = extracted.strip("'\"")
        return {"intent": "add_note", "extracted_text": extracted, "params": {}}

    # --- Search notes ---
    m = re.match(r"search\s+my\s+notes\s+(?:for\s+)?(.+)", lower, re.IGNORECASE)
    if m:
        query = text[m.start(1):m.end(1)].strip()
        return {"intent": "search_notes", "extracted_text": "", "params": {"query": query}}

    # --- Search ideas ---
    m = re.match(r"search\s+my\s+ideas?\s+(?:for\s+)?(.+)", lower, re.IGNORECASE)
    if m:
        query = text[m.start(1):m.end(1)].strip()
        return {"intent": "search_ideas", "extracted_text": "", "params": {"query": query}}

    # --- Show notes ---
    if re.match(r"(show|display|list)\s+(my\s+)?notes(\s+and\s+insights)?", lower):
        return {"intent": "show_notes", "extracted_text": "", "params": {}}

    # --- Show ideas ---
    if re.match(r"(show|display|list)\s+(my\s+)?(idea\s*log|ideas)", lower):
        return {"intent": "show_ideas", "extracted_text": "", "params": {}}

    # --- List papers ---
    if re.match(r"(list|show|display|what)\s+(all\s+)?(the\s+)?(papers?|documents?)", lower):
        return {"intent": "list_papers", "extracted_text": "", "params": {}}
    if re.search(r"papers?\s+(ingested|indexed|in\s+the\s+collection|in\s+the\s+database|stored)", lower):
        return {"intent": "list_papers", "extracted_text": "", "params": {}}

    # --- Default: regular question ---
    return {"intent": "question", "extracted_text": text, "params": {}}
