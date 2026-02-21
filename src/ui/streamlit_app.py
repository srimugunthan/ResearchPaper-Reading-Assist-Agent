"""Streamlit Q&A chat interface for ResearchPaper-reading-Assist Agent."""
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.core.llm import get_llm
from src.core.embeddings import get_embeddings
from src.core.vectorstore import get_vectorstore
from src.core.retriever import create_filtered_retriever
from src.qna.qa import ask
from src.synthesis.filters import get_unique_titles
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
    # --- Idea Log & Notes buttons ---
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Idea Log", use_container_width=True):
            st.session_state.pending_sidebar_query = "Show my ideas"
    with btn_col2:
        if st.button("Notes & Insights", use_container_width=True):
            st.session_state.pending_sidebar_query = "Show my notes"

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

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
# Check for pending sidebar button query
prompt = st.chat_input("Ask a question about your research papers...")
if not prompt and st.session_state.get("pending_sidebar_query"):
    prompt = st.session_state.pop("pending_sidebar_query")

if prompt:
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

            # --- Handle: List papers ---
            elif intent == "list_papers":
                emb = get_embeddings(provider="sentence-transformers", model_name="nomic-ai/nomic-embed-text-v1.5")
                store = get_vectorstore(
                    persist_directory="./chroma_db",
                    embedding_function=emb,
                )
                titles = get_unique_titles(store)
                if titles:
                    lines = [f"**Papers in the collection** ({len(titles)} total):\n"]
                    for i, t in enumerate(titles, 1):
                        lines.append(f"{i}. {t}")
                    answer = "\n".join(lines)
                else:
                    answer = "No papers found. Please ingest papers first."
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            # --- Handle: Regular question ---
            else:
                with st.spinner("Thinking..."):
                    emb = get_embeddings(provider="sentence-transformers", model_name="nomic-ai/nomic-embed-text-v1.5")
                    store = get_vectorstore(
                        persist_directory="./chroma_db",
                        embedding_function=emb,
                    )
                    retriever = create_filtered_retriever(store, query=prompt, k=top_k)
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

                    # Append sources with section info
                    if result["sources"]:
                        answer += "\n\n**Sources:**\n"
                        for src in result["sources"]:
                            section_str = f", {src.get('section', '')}" if src.get("section") else ""
                            answer += f"- [{src['title']}, {src['authors']}, p.{src['page']}{section_str}]\n"

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    # Save to conversation memory
                    conv_memory.save_exchange(prompt, answer)

        except Exception as e:
            error_msg = f"Error: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
