"""Tool adapters for the research agent."""
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class Tool:
    """A tool the agent can use."""

    name: str
    description: str
    func: Callable[[str], str]


def create_arxiv_tool() -> Tool:
    """Create an arXiv search tool."""
    from langchain_community.tools import ArxivQueryRun

    arxiv = ArxivQueryRun()
    return Tool(
        name="arxiv_search",
        description=(
            "Search arXiv for academic papers. "
            "Input: a search query string. Output: paper titles and summaries."
        ),
        func=lambda query: arxiv.run(query),
    )


def create_web_search_tool() -> Tool:
    """Create a DuckDuckGo web search tool."""
    from langchain_community.tools import DuckDuckGoSearchRun

    ddg = DuckDuckGoSearchRun()
    return Tool(
        name="web_search",
        description=(
            "Search the web using DuckDuckGo. "
            "Input: a search query. Output: search result snippets."
        ),
        func=lambda query: ddg.run(query),
    )


def create_python_repl_tool() -> Tool:
    """Create a Python REPL tool for computations.

    Uses a restricted exec() environment — no file I/O, no dangerous imports.
    """
    BLOCKED_MODULES = {"os", "sys", "subprocess", "shutil", "pathlib"}

    def safe_exec(code: str) -> str:
        for mod in BLOCKED_MODULES:
            if mod in code:
                return f"Error: Use of '{mod}' is not allowed for safety reasons."
        try:
            local_vars: dict = {}
            exec(code, {"__builtins__": {}}, local_vars)
            if local_vars:
                last_key = list(local_vars.keys())[-1]
                return str(local_vars[last_key])
            return "Code executed successfully (no output)."
        except Exception as e:
            return f"Error: {e}"

    return Tool(
        name="python_repl",
        description=(
            "Execute Python code for calculations or data processing. "
            "Input: Python code string. Output: execution result."
        ),
        func=safe_exec,
    )


def create_paper_retriever_tool(retriever) -> Tool:
    """Create a tool that searches the local paper vector store.

    Args:
        retriever: A LangChain retriever from create_retriever().
    """

    def search_papers(query: str) -> str:
        docs = retriever.invoke(query)
        if not docs:
            return "No matching papers found in the local collection."
        results = []
        for doc in docs:
            title = doc.metadata.get("title", "Unknown")
            page = doc.metadata.get("page", "?")
            snippet = doc.page_content[:200]
            results.append(f"[{title}, p.{page}]: {snippet}")
        return "\n\n".join(results)

    return Tool(
        name="paper_search",
        description=(
            "Search your local collection of ingested research papers. "
            "Input: a search query. Output: relevant paper excerpts with citations."
        ),
        func=search_papers,
    )


def get_default_tools(retriever=None) -> List[Tool]:
    """Return the default set of agent tools.

    Args:
        retriever: Optional retriever for paper search. If None, paper_search is excluded.
    """
    tools = [
        create_arxiv_tool(),
        create_web_search_tool(),
        create_python_repl_tool(),
    ]
    if retriever is not None:
        tools.append(create_paper_retriever_tool(retriever))
    return tools
