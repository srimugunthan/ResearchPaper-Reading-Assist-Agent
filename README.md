# ResearchPaper-reading-Assist Agent

An intelligent LangChain-based system that helps researchers and students manage, understand, and extract insights from academic papers. Ingest PDFs, ask natural language questions with citations, synthesize across papers, and run autonomous research tasks.

## Features

- **PDF Ingestion** — Scan folders, extract text/metadata, chunk and embed into a vector store
- **Q&A with Citations** — Ask questions about your papers and get answers with source citations
- **Persistent Memory** — Conversation history preserved across sessions
- **Multi-Paper Synthesis** — Comparative analysis across papers with metadata filtering
- **Research Agent** — Autonomous ReAct agent with arXiv search, web search, and Python REPL tools

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Gemini (cloud, default), Ollama (local/offline) |
| Embeddings | Gemini embeddings / sentence-transformers |
| Vector Store | Chroma (local persistence) |
| Framework | LangChain 0.1+ |
| Frontend | Streamlit |
| API | FastAPI + Uvicorn |
| Language | Python 3.10+ |

## Prerequisites

- Python 3.10 or higher
- **Gemini (cloud):** A Google Gemini API key (`GEMINI_API_KEY`)
- **Ollama (local):** [Ollama](https://ollama.com) installed with a model pulled (e.g., `ollama pull llama3`)

## Installation

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd PersonalResearchAssistant
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Copy the example env file and fill in your API keys:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your values:

   ```
   GEMINI_API_KEY=...
   ANTHROPIC_API_KEY=sk-ant-...
   ```

## Running the App

### Ingest Papers (CLI)

Scan a folder of PDFs to build the vector index:

```bash
python cli.py --folder /path/to/papers/
```

### Q&A Chat Interface (Streamlit)

Launch the interactive Q&A UI:

```bash
streamlit run src/ui/streamlit_app.py
```

### Multi-Paper Synthesis View

```bash
streamlit run src/ui/synthesis_view.py
```

### Research Agent Dashboard

```bash
streamlit run src/ui/agent_dashboard.py
```

### Memory API (FastAPI)

```bash
uvicorn src.api.memory_api:app --reload
```

Endpoints:
- `GET /memory` — View conversation memory
- `DELETE /memory` — Clear conversation memory

## Running Tests

```bash
pytest tests/ -v
```

All tests use mocked LLM and vector DB calls — no API keys required for testing.

## Configuration

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `CHROMA_PERSIST_PATH` | `./chroma_db` | Path for Chroma vector store persistence |
| `LLM_MODEL` | `gemini-2.0-flash` | Primary LLM model |
| `EMBEDDING_MODEL` | `models/embedding-001` | Embedding model |
| `CHUNK_SIZE` | `1000` | Text chunk size for splitting |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `5` | Number of retrieved chunks per query |
| `DEV_MODE` | `false` | Enable mocked LLM/vector DB for local dev |

## Project Structure

```
PersonalResearchAssistant/
├── cli.py                    # CLI entry point for PDF ingestion
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── src/
│   ├── core/                 # Embeddings, vector store, retriever utils
│   ├── ingestion/            # PDF scanning, loading, splitting, pipeline
│   ├── qna/                  # RetrievalQA chain and citation prompts
│   ├── memory/               # Conversation memory and persistence
│   ├── synthesis/            # Multi-paper synthesis chains and filters
│   ├── agent/                # ReAct agent and tool adapters
│   ├── ui/                   # Streamlit pages
│   └── api/                  # FastAPI endpoints
├── tests/                    # Unit and integration tests
├── docs/                     # Phase-specific documentation
└── examples/                 # Quickstart guide and sample papers
```

## License

This project is for personal/research use.
