"""Streamlit Q&A chat interface for ResearchPaper-reading-Assist Agent."""
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.core.llm import get_llm
from src.core.embeddings import get_embeddings
from src.core.vectorstore import get_vectorstore
from src.core.retriever import create_retriever
from src.qna.qa import ask
from src.ingestion.pipeline import ingest_folder
from src.memory.memory import ConversationMemory
from src.memory.idea_log import IdeaLog
from src.memory.notes_log import NotesLog
from src.memory.intent_detector import detect_intent

DISPLAY_CAP = int(os.getenv("DISPLAY_CAP", "10"))

st.set_page_config(page_title="ResearchPaper-reading-Assist Agent", layout="wide")
st.title("ResearchPaper-reading-Assist Agent")

# --- Initialize persistent stores in session state ---
if "conv_memory" not in st.session_state:
    st.session_state.conv_memory = ConversationMemory(filepath="conversation_memory.json")
if "idea_log" not in st.session_state:
    st.session_state.idea_log = IdeaLog(filepath="idea_log.json")
if "notes_log" not in st.session_state:
    st.session_state.notes_log = NotesLog(filepath="notes_and_insights.json")

conv_memory = st.session_state.conv_memory
idea_log = st.session_state.idea_log
notes_log = st.session_state.notes_log

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

    st.divider()

    # --- Memory section ---
    st.header("Memory")
    if st.button("View Memory"):
        history = conv_memory.get_history()
        if history:
            for ex in history[-DISPLAY_CAP:]:
                st.text(f"User: {ex['input'][:80]}...")
                st.text(f"Asst: {ex['output'][:80]}...")
                st.text("---")
            if len(history) > DISPLAY_CAP:
                st.caption(f"Showing last {DISPLAY_CAP} of {len(history)} exchanges.")
        else:
            st.info("No conversation memory stored.")

    if st.button("Clear Memory"):
        conv_memory.clear()
        st.success("Conversation memory cleared.")

    st.divider()

    # --- Idea Log section ---
    st.header("Idea Log")
    idea_result = idea_log.get_ideas(limit=DISPLAY_CAP)
    if idea_result["ideas"]:
        for idea in idea_result["ideas"]:
            cols = st.columns([4, 1])
            with cols[0]:
                st.text(idea["text"][:80])
                if idea["tags"]:
                    st.caption(", ".join(idea["tags"][:3]))
            with cols[1]:
                if st.button("x", key=f"del_idea_{idea['id']}"):
                    idea_log.delete_idea(idea["id"])
                    st.rerun()
        if idea_result["total_count"] > DISPLAY_CAP:
            st.caption(f"Showing {DISPLAY_CAP} of {idea_result['total_count']} ideas.")
    else:
        st.info("No ideas yet. Add via chat: 'Add to idea log: ...'")

    st.divider()

    # --- Notes & Insights section ---
    st.header("Notes & Insights")
    notes_result = notes_log.get_notes(limit=DISPLAY_CAP)
    if notes_result["notes"]:
        for note in notes_result["notes"]:
            cols = st.columns([4, 1])
            with cols[0]:
                st.text(note["text"][:80])
            with cols[1]:
                if st.button("x", key=f"del_note_{note['id']}"):
                    notes_log.delete_note(note["id"])
                    st.rerun()
        if notes_result["total_count"] > DISPLAY_CAP:
            st.caption(f"Showing {DISPLAY_CAP} of {notes_result['total_count']} notes.")
    else:
        st.info("No notes yet. Add via chat: 'Add this to notes: ...'")


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

    # Detect intent
    intent_result = detect_intent(prompt)
    intent = intent_result["intent"]

    with st.chat_message("assistant"):
        try:
            # --- Handle: Add to idea log ---
            if intent == "add_idea":
                entry = idea_log.add_idea(intent_result["extracted_text"])
                answer = (
                    f"Added to idea log: \"{entry['text']}\"\n\n"
                    f"**Tags:** {', '.join(entry['tags']) if entry['tags'] else 'none'}\n\n"
                    f"**ID:** `{entry['id']}`"
                )
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            # --- Handle: Add to notes ---
            elif intent == "add_note":
                entry = notes_log.add_note(intent_result["extracted_text"])
                answer = (
                    f"Added to notes: \"{entry['text']}\"\n\n"
                    f"**Tags:** {', '.join(entry['tags']) if entry['tags'] else 'none'}\n\n"
                    f"**ID:** `{entry['id']}`"
                )
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            # --- Handle: Show ideas ---
            elif intent == "show_ideas":
                result = idea_log.get_ideas(limit=DISPLAY_CAP)
                if result["ideas"]:
                    lines = [f"**Idea Log** (showing {len(result['ideas'])} of {result['total_count']}):\n"]
                    for i, idea in enumerate(result["ideas"], 1):
                        tags = f" [{', '.join(idea['tags'])}]" if idea["tags"] else ""
                        lines.append(f"{i}. {idea['text']}{tags}")
                    if result["total_count"] > DISPLAY_CAP:
                        lines.append(f"\n*Use 'Search my ideas for ...' to find older entries.*")
                    answer = "\n".join(lines)
                else:
                    answer = "Your idea log is empty. Add ideas with: 'Add to idea log: ...'"
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            # --- Handle: Show notes ---
            elif intent == "show_notes":
                result = notes_log.get_notes(limit=DISPLAY_CAP)
                if result["notes"]:
                    lines = [f"**Notes & Insights** (showing {len(result['notes'])} of {result['total_count']}):\n"]
                    for i, note in enumerate(result["notes"], 1):
                        lines.append(f"{i}. {note['text']}")
                    if result["total_count"] > DISPLAY_CAP:
                        lines.append(f"\n*Use 'Search my notes for ...' to find older entries.*")
                    answer = "\n".join(lines)
                else:
                    answer = "Your notes are empty. Add notes with: 'Add this to notes: ...'"
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            # --- Handle: Search ideas ---
            elif intent == "search_ideas":
                query = intent_result["params"].get("query", "")
                results = idea_log.search_ideas(query=query)
                if results:
                    lines = [f"**Ideas matching '{query}':**\n"]
                    for i, idea in enumerate(results, 1):
                        tags = f" [{', '.join(idea['tags'])}]" if idea["tags"] else ""
                        lines.append(f"{i}. {idea['text']}{tags}")
                    answer = "\n".join(lines)
                else:
                    answer = f"No ideas found matching '{query}'."
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            # --- Handle: Search notes ---
            elif intent == "search_notes":
                query = intent_result["params"].get("query", "")
                results = notes_log.search_notes(query=query)
                if results:
                    lines = [f"**Notes matching '{query}':**\n"]
                    for i, note in enumerate(results, 1):
                        lines.append(f"{i}. {note['text']}")
                    answer = "\n".join(lines)
                else:
                    answer = f"No notes found matching '{query}'."
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            # --- Handle: Regular question ---
            else:
                with st.spinner("Thinking..."):
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

                    # Save to conversation memory
                    conv_memory.save_exchange(prompt, answer)

        except Exception as e:
            error_msg = f"Error: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
