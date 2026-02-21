"""Citation-aware prompt templates for Q&A."""
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


QA_TEMPLATE = """Use the following context from research papers to answer the question.
Do NOT include inline citations or references in your answer. Just provide a clear, direct answer.
Sources will be listed separately.
If you cannot answer from the provided context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""


TITLE_ENFORCING_QA_TEMPLATE = """You are a research paper Q&A assistant. Your task is to answer questions
about research papers using ONLY the provided context.

CRITICAL INSTRUCTION: If the user's question explicitly mentions a paper title or author,
PRIORITIZE retrievals matching that title/author. If the retrieved context contains the paper
mentioned in the question, extract and highlight its main ideas, contributions, or findings explicitly.

If you find the paper mentioned in the question but the provided excerpts are insufficient to answer
the full question, say you found the paper but the provided excerpts don't contain enough information.

If the paper mentioned in the question is NOT in the retrieved context, say you don't have that paper
in your knowledge base.

Do NOT include inline citations or references like [Title, Authors, p.X] in your answer.
Just provide a clear, direct answer. Sources will be listed separately.
Never make up information not in the context.

Context:
{context}

Question: {question}

Answer:"""


def build_qa_prompt() -> PromptTemplate:
    """Build the QA prompt template with citation instructions.

    Returns:
        A PromptTemplate with 'context' and 'question' variables.
    """
    return PromptTemplate(
        template=QA_TEMPLATE,
        input_variables=["context", "question"],
    )


def build_title_enforcing_qa_prompt() -> PromptTemplate:
    """Build a title-enforcing QA prompt that prioritizes exact title/author matches.

    This prompt is designed for queries that mention specific paper titles or authors.
    It instructs the LLM to:
    1. Prioritize papers matching the mentioned title/author
    2. Explicitly extract main ideas from matching papers
    3. Provide helpful fallback messages if the paper is found but excerpts are insufficient
    4. Indicate if the paper is not in the knowledge base

    Returns:
        A PromptTemplate with 'context' and 'question' variables.
    """
    return PromptTemplate(
        template=TITLE_ENFORCING_QA_TEMPLATE,
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
