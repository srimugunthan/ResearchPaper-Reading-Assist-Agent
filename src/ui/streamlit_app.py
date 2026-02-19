"""Streamlit Q&A chat interface for ResearchPaper-reading-Assist Agent."""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.core.llm import get_llm
from src.core.embeddings import get_embeddings
from src.core.vectorstore import get_vectorstore
from src.core.retriever import create_retriever
from src.qna.qa import ask
from src.ingestion.pipeline import ingest_folder


st.set_page_config(page_title="ResearchPaper-reading-Assist Agent", layout="wide")
st.title("ResearchPaper-reading-Assist Agent")

# --- Sidebar ---
with st.sidebar:
    st.header("Ingestion")
    folder_path = st.text_input("PDF Folder Path", value="./examples/sample_papers")
    persist_dir = st.text_input("Chroma Persist Dir", value="./chroma_db")

    embedding_provider = st.selectbox(
        "Embedding Provider",
        ["sentence-transformers", "fake"],
        index=0,
    )

    if st.button("Ingest Papers"):
        with st.spinner("Ingesting papers..."):
            try:
                emb = get_embeddings(provider=embedding_provider)
                result = ingest_folder(
                    folder_path=folder_path,
                    persist_directory=persist_dir,
                    embedding_function=emb,
                )
                st.success(
                    f"Done! Processed: {result['files_processed']}, "
                    f"Skipped: {result['files_skipped']}, "
                    f"Failed: {result['files_failed']}"
                )
                if result["errors"]:
                    for err in result["errors"]:
                        st.warning(f"{err['file']}: {err['error']}")
            except FileNotFoundError as e:
                st.error(str(e))

    st.divider()

    st.header("Settings")
    llm_provider = st.selectbox("LLM Provider", ["gemini", "ollama", "fake"], index=0)
    llm_model = st.text_input(
        "Model Name",
        value="gemini-2.0-flash" if llm_provider == "gemini" else "llama3",
    )
    top_k = st.slider("Top-K Retrieval", min_value=1, max_value=20, value=5)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
if prompt := st.chat_input("Ask a question about your research papers..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                emb = get_embeddings(provider=embedding_provider)
                store = get_vectorstore(
                    persist_directory=persist_dir,
                    embedding_function=emb,
                )
                retriever = create_retriever(store, k=top_k)
                llm = get_llm(
                    provider=llm_provider,
                    model_name=llm_model,
                    temperature=temperature,
                )

                result = ask(
                    question=prompt,
                    retriever=retriever,
                    llm=llm,
                )

                answer = result["answer"]

                # Append sources
                if result["sources"]:
                    answer += "\n\n**Sources:**\n"
                    for src in result["sources"]:
                        answer += f"- [{src['title']}, {src['authors']}, p.{src['page']}]\n"

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
