"""Tests for agent tools and ReAct loop."""
import pytest
from unittest.mock import MagicMock
from langchain_core.language_models import FakeListChatModel
from langchain_core.documents import Document

from src.agent.tools import Tool, create_paper_retriever_tool, create_python_repl_tool
from src.agent.agent import (
    run_agent,
    _parse_llm_output,
    _build_tool_descriptions,
    TraceEntry,
    AgentResult,
)


# --- Helper: create mock tools ---

def _make_mock_tools():
    """Create mock tools that return canned responses."""
    return [
        Tool(
            name="arxiv_search",
            description="Search arXiv for papers.",
            func=lambda q: "Found: 'Diffusion Models Beat GANs' by Dhariwal 2023",
        ),
        Tool(
            name="web_search",
            description="Search the web.",
            func=lambda q: "DuckDuckGo result: Diffusion models are generative...",
        ),
        Tool(
            name="paper_search",
            description="Search local papers.",
            func=lambda q: "[Attention Paper, p.3]: Self-attention mechanism...",
        ),
    ]


class TestParseOutput:
    """Tests for _parse_llm_output."""

    def test_parses_thought_and_action(self):
        text = "Thought: I need to search arXiv\nAction: arxiv_search\nAction Input: diffusion models"
        parsed = _parse_llm_output(text)
        assert parsed["thought"] == "I need to search arXiv"
        assert parsed["action"] == "arxiv_search"
        assert parsed["action_input"] == "diffusion models"
        assert parsed["final_answer"] is None

    def test_parses_final_answer(self):
        text = "Thought: I have enough info.\nFinal Answer: The top 3 papers are..."
        parsed = _parse_llm_output(text)
        assert parsed["thought"] == "I have enough info."
        assert parsed["final_answer"].startswith("The top 3 papers")
        assert parsed["action"] is None

    def test_handles_malformed_output(self):
        text = "I'm not sure what to do here."
        parsed = _parse_llm_output(text)
        assert parsed["final_answer"] is None

    def test_final_answer_takes_priority(self):
        text = "Thought: Done\nAction: web_search\nFinal Answer: Here is the result."
        parsed = _parse_llm_output(text)
        assert parsed["final_answer"] is not None


class TestBuildToolDescriptions:

    def test_formats_tool_list(self):
        tools = _make_mock_tools()
        desc = _build_tool_descriptions(tools)
        assert "arxiv_search" in desc
        assert "web_search" in desc
        assert "paper_search" in desc


class TestRunAgent:
    """Tests for the full agent loop."""

    def test_agent_calls_tool_and_returns_final_answer(self):
        tools = _make_mock_tools()
        fake_llm = FakeListChatModel(responses=[
            "Thought: I should search arXiv for diffusion models.\nAction: arxiv_search\nAction Input: diffusion models 2023",
            "Thought: I found the papers. Let me summarize.\nFinal Answer: The top paper is 'Diffusion Models Beat GANs' by Dhariwal.",
        ])
        result = run_agent(task="Find papers on diffusion models", tools=tools, llm=fake_llm, max_iterations=5)
        assert result.stopped_reason == "final_answer"
        assert "Diffusion Models Beat GANs" in result.answer
        assert result.iterations_used == 2
        assert len(result.trace) == 2

    def test_agent_calls_correct_tool(self):
        call_log = []
        tools = [
            Tool(name="arxiv_search", description="Search arXiv.", func=lambda q: (call_log.append(("arxiv", q)), "result")[1]),
            Tool(name="web_search", description="Search web.", func=lambda q: (call_log.append(("web", q)), "result")[1]),
        ]
        fake_llm = FakeListChatModel(responses=[
            "Thought: Search arXiv.\nAction: arxiv_search\nAction Input: RLHF papers",
            "Thought: Done.\nFinal Answer: Found papers.",
        ])
        run_agent(task="Find RLHF papers", tools=tools, llm=fake_llm, max_iterations=5)
        assert len(call_log) == 1
        assert call_log[0] == ("arxiv", "RLHF papers")

    def test_agent_uses_multiple_tools(self):
        call_log = []
        tools = [
            Tool(name="arxiv_search", description="Search arXiv.", func=lambda q: (call_log.append("arxiv"), "arxiv result")[1]),
            Tool(name="paper_search", description="Search local.", func=lambda q: (call_log.append("paper"), "local result")[1]),
        ]
        fake_llm = FakeListChatModel(responses=[
            "Thought: First search arXiv.\nAction: arxiv_search\nAction Input: RLHF",
            "Thought: Now search local papers.\nAction: paper_search\nAction Input: RLHF",
            "Thought: Got both results.\nFinal Answer: Summary of findings.",
        ])
        result = run_agent(task="Research RLHF", tools=tools, llm=fake_llm, max_iterations=5)
        assert result.iterations_used == 3
        assert "arxiv" in call_log
        assert "paper" in call_log

    def test_iteration_limit_stops_agent(self):
        tools = _make_mock_tools()
        fake_llm = FakeListChatModel(responses=[
            "Thought: Need more info.\nAction: arxiv_search\nAction Input: query"
        ] * 5)
        result = run_agent(task="Endless task", tools=tools, llm=fake_llm, max_iterations=5)
        assert result.stopped_reason == "max_iterations"
        assert result.iterations_used == 5
        assert len(result.trace) == 5
        assert len(result.answer) > 0

    def test_unknown_tool_handled_gracefully(self):
        tools = _make_mock_tools()
        fake_llm = FakeListChatModel(responses=[
            "Thought: Use a nonexistent tool.\nAction: fake_tool\nAction Input: test",
            "Thought: That failed. Let me use arXiv.\nAction: arxiv_search\nAction Input: test",
            "Thought: Done.\nFinal Answer: Result from arXiv.",
        ])
        result = run_agent(task="Test", tools=tools, llm=fake_llm, max_iterations=5)
        assert result.stopped_reason == "final_answer"
        assert "Unknown tool" in result.trace[0].tool_output

    def test_tool_exception_handled(self):
        def bad_func(q):
            raise RuntimeError("boom")

        tools = [
            Tool(name="bad_tool", description="Always fails.", func=bad_func),
        ]
        fake_llm = FakeListChatModel(responses=[
            "Thought: Use bad tool.\nAction: bad_tool\nAction Input: test",
            "Thought: That failed. I'll just answer.\nFinal Answer: Could not complete the task.",
        ])
        result = run_agent(task="Test", tools=tools, llm=fake_llm, max_iterations=5)
        assert result.stopped_reason == "final_answer"
        assert "Tool error" in result.trace[0].tool_output

    def test_on_step_callback_called(self):
        steps_received = []
        tools = _make_mock_tools()
        fake_llm = FakeListChatModel(responses=[
            "Thought: Search.\nAction: arxiv_search\nAction Input: test",
            "Thought: Done.\nFinal Answer: Result.",
        ])
        run_agent(
            task="Test",
            tools=tools,
            llm=fake_llm,
            max_iterations=5,
            on_step=lambda entry: steps_received.append(entry),
        )
        assert len(steps_received) == 2


class TestPaperRetrieverTool:
    """Tests for create_paper_retriever_tool."""

    def test_returns_formatted_results(self):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="Self-attention allows the model to attend to all positions...",
                     metadata={"title": "Attention Paper", "page": 3}),
        ]
        tool = create_paper_retriever_tool(mock_retriever)
        result = tool.func("attention")
        assert "Attention Paper" in result
        assert "p.3" in result

    def test_no_results_message(self):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        tool = create_paper_retriever_tool(mock_retriever)
        result = tool.func("nonexistent")
        assert "No matching papers" in result


class TestPythonReplTool:
    """Tests for the safe Python REPL tool."""

    def test_basic_calculation(self):
        tool = create_python_repl_tool()
        result = tool.func("result = 2 + 3")
        assert "5" in result

    def test_blocked_module(self):
        tool = create_python_repl_tool()
        result = tool.func("import os; os.listdir('.')")
        assert "not allowed" in result.lower() or "error" in result.lower()

    def test_syntax_error_handled(self):
        tool = create_python_repl_tool()
        result = tool.func("def broken(")
        assert "Error" in result
