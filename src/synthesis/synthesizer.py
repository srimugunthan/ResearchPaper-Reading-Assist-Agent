"""Multi-paper synthesis using map-reduce and refine strategies."""
from collections import defaultdict
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

MAP_TEMPLATE = """You are a research paper analyst. Summarize the following excerpt from the paper "{title}".
Focus on: key findings, methodology, and conclusions relevant to the question.

Paper excerpt:
{content}

Question/Topic: {question}

Summary of this paper's contribution:"""

REDUCE_TEMPLATE = """You are a research synthesis expert. Below are summaries from multiple research papers on the topic: "{question}"

Per-paper summaries:
{summaries}

Synthesize these into a coherent comparative analysis. Your response MUST include these sections:
1. **Common Themes**: Shared findings or approaches across papers
2. **Key Differences**: How the papers differ in methodology or conclusions
3. **Contradictions**: Any conflicting findings or claims (if none, state "No direct contradictions identified")
4. **Overall Synthesis**: A brief integrative summary

For each claim, cite the source paper using [Paper Title].

Comparative Synthesis:"""

REFINE_TEMPLATE = """You are a research synthesis expert. You have an existing synthesis and a new paper summary to incorporate.

Current synthesis:
{current_synthesis}

New paper summary (from "{title}"):
{new_summary}

Question/Topic: {question}

Update the synthesis to incorporate the new paper. Maintain these sections:
1. **Common Themes**
2. **Key Differences**
3. **Contradictions**
4. **Overall Synthesis**

Cite all source papers using [Paper Title].

Updated Synthesis:"""

INITIAL_REFINE_TEMPLATE = """You are a research synthesis expert. Begin a synthesis based on this first paper summary.

Paper summary (from "{title}"):
{summary}

Question/Topic: {question}

Create an initial synthesis with these sections:
1. **Common Themes**: (initial observations)
2. **Key Differences**: (to be filled as more papers are added)
3. **Contradictions**: (to be filled as more papers are added)
4. **Overall Synthesis**: A brief summary so far

Cite the source paper using [Paper Title].

Initial Synthesis:"""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_map_prompt() -> PromptTemplate:
    """Build the prompt template for the map stage."""
    return PromptTemplate(
        template=MAP_TEMPLATE,
        input_variables=["title", "content", "question"],
    )


def build_reduce_prompt() -> PromptTemplate:
    """Build the prompt template for the reduce stage."""
    return PromptTemplate(
        template=REDUCE_TEMPLATE,
        input_variables=["question", "summaries"],
    )


def build_refine_prompt() -> PromptTemplate:
    """Build the prompt template for the refine stage."""
    return PromptTemplate(
        template=REFINE_TEMPLATE,
        input_variables=["current_synthesis", "title", "new_summary", "question"],
    )


def build_initial_refine_prompt() -> PromptTemplate:
    """Build the prompt template for the initial refine stage."""
    return PromptTemplate(
        template=INITIAL_REFINE_TEMPLATE,
        input_variables=["title", "summary", "question"],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_docs_by_title(docs: List[Document]) -> Dict[str, List[Document]]:
    """Group documents by their title metadata."""
    groups: Dict[str, List[Document]] = defaultdict(list)
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        groups[title].append(doc)
    return dict(groups)


def _combine_doc_content(docs: List[Document], max_chars: int = 8000) -> str:
    """Combine page content from multiple docs, with truncation."""
    combined = "\n\n".join(doc.page_content for doc in docs)
    if len(combined) > max_chars:
        combined = combined[:max_chars]
    return combined


def _extract_sources(grouped: Dict[str, List[Document]]) -> List[Dict[str, Any]]:
    """Extract source metadata from grouped documents."""
    sources = []
    for title, docs in grouped.items():
        pages = sorted({doc.metadata.get("page", 0) for doc in docs})
        sources.append({"title": title, "pages": pages})
    return sources


def _llm_call(llm, text: str) -> str:
    """Invoke LLM and return text content."""
    result = llm.invoke(text)
    return result.content if hasattr(result, "content") else str(result)


# ---------------------------------------------------------------------------
# Synthesis strategies
# ---------------------------------------------------------------------------

def synthesize_map_reduce(
    docs: List[Document],
    question: str,
    llm,
) -> Dict[str, Any]:
    """Synthesize across multiple papers using map-reduce strategy."""
    grouped = _group_docs_by_title(docs)
    map_prompt = build_map_prompt()

    # Map stage: summarize each paper
    paper_summaries = []
    for title, group_docs in grouped.items():
        content = _combine_doc_content(group_docs)
        formatted = map_prompt.format(title=title, content=content, question=question)
        summary = _llm_call(llm, formatted)
        paper_summaries.append({"title": title, "summary": summary})

    # Reduce stage: combine summaries
    numbered = "\n\n".join(
        f"[{i}] {ps['title']}:\n{ps['summary']}"
        for i, ps in enumerate(paper_summaries, 1)
    )
    reduce_prompt = build_reduce_prompt()
    formatted = reduce_prompt.format(question=question, summaries=numbered)
    synthesis = _llm_call(llm, formatted)

    return {
        "synthesis": synthesis,
        "paper_summaries": paper_summaries,
        "sources": _extract_sources(grouped),
        "strategy": "map_reduce",
        "papers_analyzed": len(grouped),
    }


def synthesize_refine(
    docs: List[Document],
    question: str,
    llm,
) -> Dict[str, Any]:
    """Synthesize across multiple papers using iterative refine strategy."""
    grouped = _group_docs_by_title(docs)
    map_prompt = build_map_prompt()
    initial_prompt = build_initial_refine_prompt()
    refine_prompt = build_refine_prompt()

    paper_summaries = []
    current_synthesis = ""

    for idx, (title, group_docs) in enumerate(grouped.items()):
        # Map: summarize this paper
        content = _combine_doc_content(group_docs)
        formatted = map_prompt.format(title=title, content=content, question=question)
        summary = _llm_call(llm, formatted)
        paper_summaries.append({"title": title, "summary": summary})

        if idx == 0:
            # Initial refine
            formatted = initial_prompt.format(
                title=title, summary=summary, question=question
            )
            current_synthesis = _llm_call(llm, formatted)
        else:
            # Refine with new paper
            formatted = refine_prompt.format(
                current_synthesis=current_synthesis,
                title=title,
                new_summary=summary,
                question=question,
            )
            current_synthesis = _llm_call(llm, formatted)

    return {
        "synthesis": current_synthesis,
        "paper_summaries": paper_summaries,
        "sources": _extract_sources(grouped),
        "strategy": "refine",
        "papers_analyzed": len(grouped),
    }


def synthesize(
    docs: List[Document],
    question: str,
    llm,
    strategy: str = "map_reduce",
) -> Dict[str, Any]:
    """Synthesize across multiple papers.

    Args:
        docs: Documents from multiple papers.
        question: The synthesis question/topic.
        llm: A LangChain chat model instance.
        strategy: Either "map_reduce" or "refine".

    Returns:
        Dict with synthesis, paper_summaries, sources, strategy, papers_analyzed.
    """
    if not docs:
        raise ValueError("No documents provided for synthesis.")
    if strategy == "map_reduce":
        return synthesize_map_reduce(docs, question, llm)
    elif strategy == "refine":
        return synthesize_refine(docs, question, llm)
    else:
        raise ValueError(f"Unknown strategy: '{strategy}'. Use 'map_reduce' or 'refine'.")
