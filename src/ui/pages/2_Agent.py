"""Streamlit page: Research Agent Dashboard."""
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.core.llm import get_llm
from src.core.embeddings import get_embeddings
from src.core.vectorstore import get_vectorstore
from src.core.retriever import create_retriever
from src.agent.tools import (
    Tool,
    create_arxiv_tool,
    create_web_search_tool,
    create_python_repl_tool,
    create_paper_retriever_tool,
)
from src.agent.agent import run_agent

st.title("Research Agent")

# --- Sidebar: Settings ---
with st.sidebar:
    st.header("Settings")
    llm_provider = st.selectbox(
        "LLM Provider", ["gemini", "ollama", "fake"], index=0, key="agent_llm"
    )
    llm_model = st.text_input(
        "Model Name",
        value="gemini-2.0-flash" if llm_provider == "gemini" else "llama3",
        key="agent_model",
    )
    temperature = st.slider(
        "Temperature", 0.0, 1.0, 0.1, step=0.05, key="agent_temp"
    )

    st.divider()

    st.header("Tools")
    use_arxiv = st.checkbox("arXiv Search", value=True, key="agent_use_arxiv")
    use_web = st.checkbox("Web Search", value=True, key="agent_use_web")
    use_repl = st.checkbox("Python REPL", value=True, key="agent_use_repl")
    use_papers = st.checkbox("Local Paper Search", value=True, key="agent_use_papers")

    st.divider()

    st.header("Agent Safety")
    max_iterations = st.number_input(
        "Max Iterations", min_value=1, max_value=20, value=10, key="agent_max_iter"
    )

# --- Main Area ---
st.markdown("---")

task = st.text_area(
    "Research Task",
    placeholder="e.g., Find recent papers on RLHF and summarize the top 3",
    height=100,
    key="agent_task",
)

if st.button("Run Agent", type="primary", key="agent_run"):
    if not task.strip():
        st.warning("Please enter a research task.")
    else:
        # Build tools list
        tools = []
        if use_arxiv:
            try:
                tools.append(create_arxiv_tool())
            except Exception as e:
                st.warning(f"Could not load arXiv tool: {e}")
        if use_web:
            try:
                tools.append(create_web_search_tool())
            except Exception as e:
                st.warning(f"Could not load web search tool: {e}")
        if use_repl:
            tools.append(create_python_repl_tool())
        if use_papers:
            try:
                emb = get_embeddings(provider="sentence-transformers")
                store = get_vectorstore(
                    persist_directory="./chroma_db", embedding_function=emb
                )
                retriever = create_retriever(store, k=5)
                tools.append(create_paper_retriever_tool(retriever))
            except Exception as e:
                st.warning(f"Could not load local papers: {e}")

        if not tools:
            st.error("No tools available. Enable at least one tool.")
        else:
            llm = get_llm(
                provider=llm_provider,
                model_name=llm_model,
                temperature=temperature,
            )

            # Trace display
            trace_container = st.container()
            progress_bar = st.progress(0)

            step_counter = [0]

            def on_step(entry):
                step_counter[0] += 1
                progress_bar.progress(min(step_counter[0] / max_iterations, 1.0))
                with trace_container:
                    state = "complete" if (entry.is_final_answer or entry.tool_output) else "running"
                    with st.status(f"Step {entry.step_number}", state=state):
                        st.markdown(f"**Thought:** {entry.thought}")
                        if entry.tool_name and not entry.is_final_answer:
                            st.markdown(f"**Tool:** `{entry.tool_name}`")
                            st.markdown(f"**Input:** {entry.tool_input}")
                            st.markdown(f"**Result:** {entry.tool_output}")
                        if entry.is_final_answer:
                            st.markdown("*Final answer produced*")

            with st.spinner("Agent is working..."):
                try:
                    result = run_agent(
                        task=task,
                        tools=tools,
                        llm=llm,
                        max_iterations=max_iterations,
                        on_step=on_step,
                    )

                    progress_bar.progress(1.0)

                    # Final answer
                    st.markdown("---")
                    st.subheader("Final Answer")
                    st.markdown(result.answer)

                    st.caption(
                        f"Completed in {result.iterations_used} step(s). "
                        f"Stopped: {result.stopped_reason}"
                    )
                except Exception as e:
                    st.error(f"Agent failed: {e}")
