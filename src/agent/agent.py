"""ReAct agent factory and executor."""
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agent.tools import Tool


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = """You are a research assistant agent. You can use tools to help answer research questions.

Available tools:
{tool_descriptions}

To use a tool, respond in EXACTLY this format:
Thought: <your reasoning about what to do next>
Action: <tool name>
Action Input: <input to the tool>

When you have enough information to answer, respond in this format:
Thought: <your final reasoning>
Final Answer: <your complete answer>

IMPORTANT RULES:
- Always start with a Thought
- Use exactly one tool per step
- After receiving a tool result, think about what to do next
- If a tool returns an error, try a different approach
- You MUST provide a Final Answer within {max_iterations} steps

Begin!

Task: {task}

{scratchpad}"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TraceEntry:
    """A single step in the agent's reasoning trace."""

    step_number: int
    thought: str
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None
    tool_output: Optional[str] = None
    is_final_answer: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    """The result of an agent run."""

    answer: str
    trace: List[TraceEntry]
    iterations_used: int
    stopped_reason: str  # "final_answer" | "max_iterations" | "error"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tool_descriptions(tools: List[Tool]) -> str:
    """Format tool descriptions for the system prompt."""
    return "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)


def _build_scratchpad(trace: List[TraceEntry]) -> str:
    """Build the scratchpad string from trace history."""
    parts: List[str] = []
    for entry in trace:
        parts.append(f"Thought: {entry.thought}")
        if entry.tool_name and not entry.is_final_answer:
            parts.append(f"Action: {entry.tool_name}")
            parts.append(f"Action Input: {entry.tool_input}")
            parts.append(f"Observation: {entry.tool_output}")
    return "\n".join(parts)


def _parse_llm_output(text: str) -> Dict[str, Any]:
    """Parse LLM output into thought, action, or final answer.

    Returns a dict with keys:
        - thought: str (always present, may be empty)
        - action: Optional[str] (tool name)
        - action_input: Optional[str] (tool input)
        - final_answer: Optional[str]
    """
    result: Dict[str, Any] = {
        "thought": "",
        "action": None,
        "action_input": None,
        "final_answer": None,
    }

    # Extract Thought
    thought_match = re.search(
        r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|\Z)", text, re.DOTALL
    )
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    # Check for Final Answer first (takes priority)
    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if final_match:
        result["final_answer"] = final_match.group(1).strip()
        return result

    # Check for Action
    action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
    if action_match:
        result["action"] = action_match.group(1).strip()

    input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text)
    if input_match:
        result["action_input"] = input_match.group(1).strip()

    return result


def _llm_call(llm, text: str) -> str:
    """Invoke LLM and return text content."""
    result = llm.invoke(text)
    return result.content if hasattr(result, "content") else str(result)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(
    task: str,
    tools: List[Tool],
    llm,
    max_iterations: int = 10,
    on_step: Optional[Any] = None,
) -> AgentResult:
    """Run the ReAct agent loop.

    Args:
        task: The research task to accomplish.
        tools: List of Tool instances the agent can use.
        llm: A LangChain chat model (or FakeListChatModel for testing).
        max_iterations: Maximum number of thinking steps.
        on_step: Optional callback called with (TraceEntry) after each step.

    Returns:
        AgentResult with answer, trace, and metadata.
    """
    tool_map = {tool.name: tool for tool in tools}
    tool_descriptions = _build_tool_descriptions(tools)
    trace: List[TraceEntry] = []

    for step in range(1, max_iterations + 1):
        # Build prompt with current scratchpad
        scratchpad = _build_scratchpad(trace)
        prompt = REACT_SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions,
            max_iterations=max_iterations,
            task=task,
            scratchpad=scratchpad,
        )

        # Call LLM
        raw_output = _llm_call(llm, prompt)
        parsed = _parse_llm_output(raw_output)

        # Build trace entry
        entry = TraceEntry(
            step_number=step,
            thought=parsed["thought"],
        )

        # Check for Final Answer
        if parsed["final_answer"]:
            entry.is_final_answer = True
            trace.append(entry)
            if on_step:
                on_step(entry)
            return AgentResult(
                answer=parsed["final_answer"],
                trace=trace,
                iterations_used=step,
                stopped_reason="final_answer",
            )

        # Execute tool
        action = parsed["action"]
        action_input = parsed["action_input"] or ""

        if action and action in tool_map:
            try:
                tool_output = tool_map[action].func(action_input)
            except Exception as e:
                tool_output = f"Tool error: {e}"
            entry.tool_name = action
            entry.tool_input = action_input
            entry.tool_output = tool_output
        elif action:
            entry.tool_name = action
            entry.tool_input = action_input
            entry.tool_output = (
                f"Error: Unknown tool '{action}'. "
                f"Available tools: {', '.join(tool_map.keys())}"
            )
        else:
            entry.tool_name = None
            entry.tool_output = (
                "Error: Could not parse action. "
                "Please use the format 'Action: <tool_name>'"
            )

        trace.append(entry)
        if on_step:
            on_step(entry)

    # Reached max iterations without a final answer
    last_thought = trace[-1].thought if trace else "No progress made."
    return AgentResult(
        answer=(
            f"I was unable to complete the task within {max_iterations} steps. "
            f"Here is what I found so far: {last_thought}"
        ),
        trace=trace,
        iterations_used=max_iterations,
        stopped_reason="max_iterations",
    )
