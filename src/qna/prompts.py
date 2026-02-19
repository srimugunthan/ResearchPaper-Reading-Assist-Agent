"""Citation-aware prompt templates for Q&A."""
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


QA_TEMPLATE = """Use the following context from research papers to answer the question.
Always cite your sources using the format [Title, Authors, p.PageNumber] for each claim.
If you cannot answer from the provided context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer (with citations):"""


def build_qa_prompt() -> PromptTemplate:
    """Build the QA prompt template with citation instructions.

    Returns:
        A PromptTemplate with 'context' and 'question' variables.
    """
    return PromptTemplate(
        template=QA_TEMPLATE,
        input_variables=["context", "question"],
    )


def format_citations(docs: List[Document]) -> str:
    """Format a list of documents into citation strings.

    Args:
        docs: List of Document objects with metadata.

    Returns:
        A formatted citation string, one citation per line.
    """
    if not docs:
        return ""

    citations = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        authors = doc.metadata.get("authors", "Unknown")
        page = doc.metadata.get("page", "?")
        citations.append(f"[{title}, {authors}, p.{page}]")

    return "\n".join(citations)
