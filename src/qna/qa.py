"""RetrievalQA chain with citation support."""
from typing import Any, Dict, List

from langchain_core.documents import Document

from src.qna.prompts import build_qa_prompt, build_title_enforcing_qa_prompt, format_citations
from src.core.vectorstore import deduplicate_results


def create_qa_chain(llm, retriever, use_title_enforcement: bool = True):
    """Create a QA chain that retrieves docs and generates cited answers.

    Args:
        llm: A LangChain chat model.
        retriever: A LangChain retriever.
        use_title_enforcement: If True, use title-enforcing prompt for better handling
                              of queries that mention specific paper titles.

    Returns:
        A dict with the chain components for use in ask().
    """
    if use_title_enforcement:
        prompt = build_title_enforcing_qa_prompt()
    else:
        prompt = build_qa_prompt()

    return {"llm": llm, "retriever": retriever, "prompt": prompt}


def _detect_title_in_question(question: str) -> bool:
    """Simple heuristic to detect if question mentions a paper title.

    Args:
        question: The user's question.

    Returns:
        True if question likely mentions a paper title (contains quotes or specific keywords).
    """
    keywords = ["paper", "title", "published", "titled", "by the authors", "gradient-based"]
    lower_q = question.lower()
    return '"' in question or any(kw in lower_q for kw in keywords)


def ask(
    question: str,
    retriever,
    llm,
    use_title_enforcement: bool = True,
) -> Dict[str, Any]:
    """Ask a question and get an answer with source citations.

    Args:
        question: The user's question.
        retriever: A LangChain retriever instance.
        llm: A LangChain chat model instance.
        use_title_enforcement: If True, use title-enforcing prompt when question mentions titles.

    Returns:
        Dict with 'answer' (str), 'sources' (list of dicts with title, authors, page, section, chunk_type),
        and 'confidence' (float 0-1 indicating how confident the system is in the answer).
    """
    # Retrieve relevant documents (increased k=5 for more candidates in hybrid search)
    docs = retriever.invoke(question)

    # Deduplicate near-identical chunks
    docs = deduplicate_results(docs)

    if not docs:
        result = llm.invoke(
            "The user asked a question but no relevant documents were found. "
            "Respond by saying you don't have enough information.\n\n"
            f"Question: {question}"
        )
        return {
            "answer": result.content if hasattr(result, "content") else str(result),
            "sources": [],
            "confidence": 0.0,
        }

    # Build context from retrieved docs with section info
    context_parts = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        authors = doc.metadata.get("authors", "Unknown")
        page = doc.metadata.get("page", "?")
        chunk_type = doc.metadata.get("chunk_type", "")
        section = doc.metadata.get("section", "")

        section_info = ""
        if section:
            section_info = f", Section: {section}"
        elif chunk_type and chunk_type != "general":
            section_info = f", Section: {chunk_type.capitalize()}"

        context_parts.append(
            f"[Source: {title}, {authors}, p.{page}{section_info}]\n{doc.page_content}"
        )
    context = "\n\n".join(context_parts)

    # Choose prompt based on question and configuration
    should_enforce = use_title_enforcement and _detect_title_in_question(question)
    if should_enforce:
        prompt = build_title_enforcing_qa_prompt()
    else:
        prompt = build_qa_prompt()

    # Invoke LLM with formatted prompt
    formatted = prompt.format(context=context, question=question)
    result = llm.invoke(formatted)
    answer_text = result.content if hasattr(result, "content") else str(result)

    # Build source metadata list (deduplicate by title)
    sources = []
    seen_titles = {}
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        if title not in seen_titles:
            seen_titles[title] = {
                "title": title,
                "authors": doc.metadata.get("authors", "Unknown"),
                "pages": {doc.metadata.get("page", 0)},
            }
        else:
            seen_titles[title]["pages"].add(doc.metadata.get("page", 0))
    for info in seen_titles.values():
        pages = sorted(info["pages"])
        sources.append({
            "title": info["title"],
            "authors": info["authors"],
            "page": ", ".join(str(p) for p in pages),
            "section": "",
            "chunk_type": "",
        })

    # Simple confidence scoring: higher if answer contains citations and mentions sources
    confidence = 0.5
    if "I don't have enough information" in answer_text or "not in my knowledge base" in answer_text:
        confidence = 0.3
    elif len(docs) >= 3 and "[" in answer_text:  # Good retrieval + citations
        confidence = 0.8
    elif len(docs) >= 1 and "[" in answer_text:  # Some retrieval + citations
        confidence = 0.6

    return {
        "answer": answer_text,
        "sources": sources,
        "confidence": confidence,
    }
