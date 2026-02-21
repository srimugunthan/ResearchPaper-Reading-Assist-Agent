# ResearchPaper-reading-Assist Agent

An intelligent LangChain-based system that helps researchers and students manage, understand, and extract insights from academic papers. Ingest PDFs, ask natural language questions with citations, and manage research ideas and notes.

## Features

- **PDF Ingestion** — Scan folders, extract text/metadata, chunk and embed into a Chroma vector store with deduplication
- **Q&A with Citations** — Ask questions about your papers and get answers with source citations (title, authors, page)
- **Idea Log** — Save, search, edit, and delete research ideas via chat or REST API
- **Notes & Insights** — Maintain a separate log of personal observations and learnings
- **Persistent Memory** — Conversation history preserved across sessions
- **Chat Commands** — Natural language commands to add ideas/notes, search, and display logs
- **REST APIs** — FastAPI endpoints for memory, ideas, and notes

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Gemini (cloud, default), Ollama (local/offline) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | Chroma (local persistence) |
| Framework | LangChain |
| Frontend | Streamlit |
| API | FastAPI + Uvicorn |
| Package Manager | uv |
| Language | Python 3.10+ |

## Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- **Gemini (cloud):** A Google API key (`GOOGLE_API_KEY`) from [AI Studio](https://aistudio.google.com/apikey)
- **Ollama (local):** [Ollama](https://ollama.com) installed with a model pulled (e.g., `ollama pull llama3`)

## Installation

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd PersonalResearchAssistant
   ```

2. **Create a virtual environment with uv**

   ```bash
   uv venv .venv
   source .venv/bin/activate  # macOS/Linux
   ```

3. **Install the project in editable mode**

   ```bash
   uv pip install -e .
   ```

4. **Install dependencies**

   ```bash
   uv pip install --python .venv/bin/python langchain langchain-core langchain-community langchain-chroma langchain-google-genai langchain-ollama langchain-text-splitters
   uv pip install --python .venv/bin/python chromadb sentence-transformers pypdf
   uv pip install --python .venv/bin/python streamlit fastapi uvicorn httpx python-dotenv
   uv pip install --python .venv/bin/python pytest pytest-mock
   ```

5. **Configure environment variables**

   Copy `.env` and fill in your API key:

   ```bash
   cp .env .env.local  # optional backup
   ```

   Edit `.env`:

   ```env
   GOOGLE_API_KEY=your-api-key-here
   LLM_PROVIDER=gemini           # or "ollama" or "fake"
   LLM_MODEL=gemini-2.0-flash    # or "llama3" for Ollama
   EMBEDDING_PROVIDER=sentence-transformers
   DISPLAY_CAP=10
   ```

## Running the App

### 1. Ingest Papers (CLI)

Scan a folder of PDFs to build the vector index:

```bash
python cli.py --folder ./examples/sample_papers/
```

Running again on the same folder skips already-ingested papers (deduplication by file hash).

### 2. Q&A Chat Interface (Streamlit)

Launch the interactive chat UI:

```bash
streamlit run src/ui/streamlit_app.py
```

**Sidebar controls:**
- Set the PDF folder path and click "Ingest Papers" to index PDFs
- Choose LLM provider (Gemini / Ollama / Fake), model name, Top-K, and temperature
- View/clear conversation memory
- Browse and delete ideas and notes

**Chat commands:**
| Command | Example |
|---------|---------|
| Ask a question | `What are the key findings of the Attention paper?` |
| Add to idea log | `Add to idea log: use LQ loss for image segmentation` |
| Add to notes | `Add this to notes: 'Only for symmetric loss, MAE is useful'` |
| Show idea log | `Show my idea log` |
| Show notes | `Show my notes` |
| Search ideas | `Search my ideas for contrastive learning` |
| Search notes | `Search my notes for MAE` |

### 3. REST APIs (FastAPI)

Start any of the API servers:

```bash
# Memory API
uvicorn src.api.memory_api:app --port 8001 --reload

# Idea Log API
uvicorn src.api.idea_log_api:app --port 8002 --reload

# Notes API
uvicorn src.api.notes_api:app --port 8003 --reload
```

**Memory API** (`http://localhost:8001`):
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/memory` | View conversation history |
| DELETE | `/memory` | Clear conversation history |

**Idea Log API** (`http://localhost:8002`):
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ideas` | Add a new idea (`{"text": "...", "tags": [...]}`) |
| GET | `/ideas` | List all ideas (optional: `?q=`, `?tag=`, `?limit=`) |
| PUT | `/ideas/{id}` | Edit an idea (`{"text": "..."}`) |
| DELETE | `/ideas/{id}` | Delete an idea |

**Notes API** (`http://localhost:8003`):
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/notes` | Add a new note (`{"text": "..."}`) |
| GET | `/notes` | List all notes (optional: `?q=`, `?limit=`) |
| PUT | `/notes/{id}` | Edit a note (`{"text": "..."}`) |
| DELETE | `/notes/{id}` | Delete a note |

**Example API calls:**

```bash
# Add an idea
curl -X POST http://localhost:8002/ideas \
  -H "Content-Type: application/json" \
  -d '{"text": "Use contrastive loss for audio tasks", "tags": ["audio", "contrastive"]}'

# List ideas
curl http://localhost:8002/ideas

# Search ideas
curl "http://localhost:8002/ideas?q=contrastive"

# Add a note
curl -X POST http://localhost:8003/notes \
  -H "Content-Type: application/json" \
  -d '{"text": "Only for symmetric loss, MAE is useful"}'

# View conversation memory
curl http://localhost:8001/memory
```

## Running Tests

All tests use mocked LLM and vector DB calls — no API keys required.

```bash
# Run all tests
pytest tests/ -v

# Run tests for a specific phase
pytest tests/test_scanner.py tests/test_ingestion_loader.py tests/test_splitter.py tests/test_vectorstore_integration.py tests/test_pipeline.py -v  # Phase 1
pytest tests/test_llm_factory.py tests/test_retriever.py tests/test_retrievalqa_citations.py tests/test_streaming_callback.py -v  # Phase 2
pytest tests/test_intent_detector.py tests/test_notes_log.py tests/test_idea_log.py tests/test_memory_persistence.py tests/test_memory_clear.py tests/test_notes_api.py tests/test_idea_log_api.py tests/test_memory_api.py -v  # Phase 3

# Run a single test file
pytest tests/test_intent_detector.py -v

# Run with short summary
pytest tests/ -q
```

**Current test count:** 135 tests across 17 test files (Phases 1-3).

## Configuration

**Environment variables** (read from `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | — | Google API key for Gemini (read by LangChain) |
| `DISPLAY_CAP` | `10` | Max entries shown for "show my notes/ideas" commands |

**Hardcoded defaults** (configurable via UI sidebar or CLI args):

| Setting | Default | Where to change |
|---------|---------|-----------------|
| LLM provider | `gemini` | Streamlit sidebar selectbox |
| LLM model | `gemini-2.0-flash` | Streamlit sidebar text input |
| LLM temperature | `0.1` | Streamlit sidebar slider |
| Embedding provider | `sentence-transformers` | Streamlit sidebar / `cli.py --embedding-provider` |
| Chroma persist dir | `./chroma_db` | Streamlit sidebar text input / `cli.py --persist-dir` |
| Top-K results | `5` | Streamlit sidebar slider |
| Chunk size | `1000` | `cli.py --chunk-size` |
| Chunk overlap | `200` | `cli.py --chunk-overlap` |

## License

This project is for personal/research use.
