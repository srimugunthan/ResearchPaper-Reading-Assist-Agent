"""Streamlit UI for multi-paper synthesis."""
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.core.llm import get_llm
from src.core.embeddings import get_embeddings
from src.core.vectorstore import get_vectorstore
from src.synthesis.filters import build_title_filter, retrieve_by_filter, get_unique_titles
from src.synthesis.synthesizer import synthesize

st.set_page_config(page_title="Multi-Paper Synthesis", layout="wide")
st.title("Multi-Paper Synthesis")

# --- Sidebar: Settings & Filters ---
with st.sidebar:
    st.header("Settings")
    llm_provider = st.selectbox("LLM Provider", ["gemini", "ollama", "fake"], index=0)
    llm_model = st.text_input(
        "Model Name",
        value="gemini-2.0-flash" if llm_provider == "gemini" else "llama3",
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, step=0.05)

    st.divider()

    st.header("Vector Store")
    persist_dir = st.text_input("Chroma Persist Dir", value="./chroma_db")
    embedding_provider = st.selectbox(
        "Embedding Provider", ["sentence-transformers", "fake"], index=0
    )

    st.divider()

    st.header("Paper Filters")
    try:
        emb = get_embeddings(provider=embedding_provider)
        store = get_vectorstore(persist_directory=persist_dir, embedding_function=emb)
        available_titles = get_unique_titles(store)
    except Exception:
        available_titles = []
        store = None

    if available_titles:
        selected_titles = st.multiselect(
            "Select Papers to Include",
            options=available_titles,
            default=available_titles,
        )
        st.caption(f"Selected: {len(selected_titles)} / {len(available_titles)} papers")
    else:
        selected_titles = []
        st.info("No papers indexed. Ingest papers first via the Q&A page.")

# --- Main Area ---
st.markdown("---")

question = st.text_area(
    "Synthesis Query",
    placeholder="e.g., Compare the approaches to attention mechanisms across all papers",
    height=100,
)

col1, col2 = st.columns(2)
with col1:
    strategy = st.radio(
        "Strategy",
        ["map_reduce", "refine"],
        format_func=lambda x: "MapReduce" if x == "map_reduce" else "Refine",
        horizontal=True,
    )
with col2:
    top_k = st.slider("Max chunks to retrieve", min_value=5, max_value=50, value=20)

if st.button("Synthesize", type="primary"):
    if not question.strip():
        st.warning("Please enter a synthesis query.")
    elif not selected_titles:
        st.warning("Please select at least one paper.")
    elif store is None:
        st.error("Vector store not available. Check your Chroma directory.")
    else:
        with st.spinner("Synthesizing across papers..."):
            try:
                where_filter = build_title_filter(selected_titles)
                docs = retrieve_by_filter(store, query=question, where=where_filter, k=top_k)

                if not docs:
                    st.error("No documents found matching your filters and query.")
                else:
                    llm = get_llm(
                        provider=llm_provider,
                        model_name=llm_model,
                        temperature=temperature,
                    )
                    result = synthesize(
                        docs=docs, question=question, llm=llm, strategy=strategy
                    )

                    # --- Display results ---
                    st.markdown("---")
                    st.subheader(
                        f"Synthesis ({result['papers_analyzed']} papers analyzed)"
                    )
                    st.markdown(result["synthesis"])

                    # Per-paper summaries
                    with st.expander("Per-Paper Summaries", expanded=False):
                        for ps in result["paper_summaries"]:
                            st.markdown(f"**{ps['title']}**")
                            st.markdown(ps["summary"])
                            st.markdown("---")

                    # Sources
                    st.subheader("Sources")
                    for src in result["sources"]:
                        pages_str = ", ".join(str(p) for p in src["pages"])
                        st.markdown(f"- **{src['title']}** (pages: {pages_str})")

            except Exception as e:
                st.error(f"Synthesis failed: {e}")
