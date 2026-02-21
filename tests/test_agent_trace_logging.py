"""Tests for agent reasoning trace logging."""
import pytest
from langchain_core.language_models import FakeListChatModel

from src.agent.tools import Tool
from src.agent.agent import run_agent, TraceEntry, AgentResult


def _make_simple_tools():
    return [
        Tool(name="search", description="Search.", func=lambda q: "search result for: " + q),
    ]


class TestTraceEntry:
    """Tests for TraceEntry dataclass."""

    def test_trace_entry_fields(self):
        entry = TraceEntry(
            step_number=1,
            thought="thinking",
            tool_name="search",
            tool_input="query",
            tool_output="result",
        )
        assert entry.step_number == 1
        assert entry.thought == "thinking"
        assert entry.tool_name == "search"
        assert entry.tool_input == "query"
        assert entry.tool_output == "result"
        assert entry.is_final_answer is False
        assert entry.timestamp > 0

    def test_final_answer_entry(self):
        entry = TraceEntry(step_number=2, thought="done", is_final_answer=True)
        assert entry.is_final_answer is True
        assert entry.tool_name is None


class TestTraceLogging:
    """Tests for trace recording during agent runs."""

    def test_trace_has_correct_step_numbers(self):
        tools = _make_simple_tools()
        fake_llm = FakeListChatModel(responses=[
            "Thought: Step one.\nAction: search\nAction Input: q1",
            "Thought: Step two.\nAction: search\nAction Input: q2",
            "Thought: Done.\nFinal Answer: All done.",
        ])
        result = run_agent(task="Test", tools=tools, llm=fake_llm, max_iterations=5)
        assert len(result.trace) == 3
        assert result.trace[0].step_number == 1
        assert result.trace[1].step_number == 2
        assert result.trace[2].step_number == 3

    def test_each_trace_entry_has_required_fields(self):
        tools = _make_simple_tools()
        fake_llm = FakeListChatModel(responses=[
            "Thought: Search something.\nAction: search\nAction Input: test query",
            "Thought: Got it.\nFinal Answer: Done.",
        ])
        result = run_agent(task="Test", tools=tools, llm=fake_llm, max_iterations=5)

        # First entry: tool invocation
        step1 = result.trace[0]
        assert step1.step_number == 1
        assert step1.thought != ""
        assert step1.tool_name == "search"
        assert step1.tool_input == "test query"
        assert step1.tool_output is not None
        assert "search result for:" in step1.tool_output
        assert step1.is_final_answer is False

        # Second entry: final answer
        step2 = result.trace[1]
        assert step2.step_number == 2
        assert step2.is_final_answer is True

    def test_final_entry_is_marked_as_final(self):
        tools = _make_simple_tools()
        fake_llm = FakeListChatModel(responses=[
            "Thought: Answering directly.\nFinal Answer: The answer is 42.",
        ])
        result = run_agent(task="What is 42?", tools=tools, llm=fake_llm, max_iterations=5)
        assert len(result.trace) == 1
        assert result.trace[-1].is_final_answer is True

    def test_trace_at_max_iterations_has_no_final_marker(self):
        tools = _make_simple_tools()
        fake_llm = FakeListChatModel(responses=[
            "Thought: Keep going.\nAction: search\nAction Input: more"
        ] * 3)
        result = run_agent(task="Loop", tools=tools, llm=fake_llm, max_iterations=3)
        assert result.stopped_reason == "max_iterations"
        assert all(not e.is_final_answer for e in result.trace)

    def test_trace_entries_have_timestamps(self):
        tools = _make_simple_tools()
        fake_llm = FakeListChatModel(responses=[
            "Thought: Go.\nAction: search\nAction Input: q",
            "Thought: Done.\nFinal Answer: Result.",
        ])
        result = run_agent(task="Test", tools=tools, llm=fake_llm, max_iterations=5)
        for entry in result.trace:
            assert entry.timestamp > 0
        assert result.trace[1].timestamp >= result.trace[0].timestamp

    def test_agent_result_has_correct_metadata(self):
        tools = _make_simple_tools()
        fake_llm = FakeListChatModel(responses=[
            "Thought: Done.\nFinal Answer: Quick answer.",
        ])
        result = run_agent(task="Quick", tools=tools, llm=fake_llm, max_iterations=10)
        assert isinstance(result, AgentResult)
        assert result.iterations_used == 1
        assert result.stopped_reason == "final_answer"
        assert result.answer == "Quick answer."

    def test_scratchpad_accumulates_across_steps(self):
        """Verify that the agent passes previous observations to subsequent LLM calls."""
        tools = _make_simple_tools()
        prompts_received = []

        class CapturingLLM:
            """Fake LLM that captures prompts."""

            def __init__(self, responses):
                self._responses = iter(responses)

            def invoke(self, text):
                prompts_received.append(text)

                class Msg:
                    content = next(self._responses)
                return Msg()

        llm = CapturingLLM([
            "Thought: First search.\nAction: search\nAction Input: q1",
            "Thought: Done.\nFinal Answer: Result.",
        ])
        run_agent(task="Test", tools=tools, llm=llm, max_iterations=5)

        # Second prompt should contain the observation from step 1
        assert len(prompts_received) == 2
        assert "Observation:" in prompts_received[1]
        assert "search result for: q1" in prompts_received[1]
