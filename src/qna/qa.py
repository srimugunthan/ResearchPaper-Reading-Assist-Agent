"""RetrievalQA chain with citation support."""
from typing import Any, Dict, List

from langchain_core.documents import Document

from src.qna.prompts import build_qa_prompt, format_citations


def create_qa_chain(llm, retriever):
    """Create a QA chain that retrieves docs and generates cited answers.

    Args:
        llm: A LangChain chat model.
        retriever: A LangChain retriever.

    Returns:
        A dict with the chain components for use in ask().
    """
    prompt = build_qa_prompt()
    return {"llm": llm, "retriever": retriever, "prompt": prompt}


def ask(
    question: str,
    retriever,
    llm,
) -> Dict[str, Any]:
    """Ask a question and get an answer with source citations.

    Args:
        question: The user's question.
        retriever: A LangChain retriever instance.
        llm: A LangChain chat model instance.

    Returns:
        Dict with 'answer' (str) and 'sources' (list of dicts with
        title, authors, page).
    """
    # Retrieve relevant documents
    docs = retriever.invoke(question)

    if not docs:
        result = llm.invoke(
            "The user asked a question but no relevant documents were found. "
            "Respond by saying you don't have enough information.\n\n"
            f"Question: {question}"
        )
        return {
            "answer": result.content if hasattr(result, "content") else str(result),
            "sources": [],
        }

    # Build context from retrieved docs
    context_parts = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        authors = doc.metadata.get("authors", "Unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Source: {title}, {authors}, p.{page}]\n{doc.page_content}"
        )
    context = "\n\n".join(context_parts)

    # Build and invoke prompt
    prompt = build_qa_prompt()
    formatted = prompt.format(context=context, question=question)
    result = llm.invoke(formatted)
    answer_text = result.content if hasattr(result, "content") else str(result)

    # Build source metadata list
    sources = []
    seen = set()
    for doc in docs:
        key = (
            doc.metadata.get("title", ""),
            doc.metadata.get("authors", ""),
            doc.metadata.get("page", ""),
        )
        if key not in seen:
            seen.add(key)
            sources.append({
                "title": doc.metadata.get("title", "Unknown"),
                "authors": doc.metadata.get("authors", "Unknown"),
                "page": doc.metadata.get("page", 0),
            })

    return {
        "answer": answer_text,
        "sources": sources,
    }
