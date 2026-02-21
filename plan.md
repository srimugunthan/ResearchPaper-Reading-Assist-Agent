# ResearchPaper-reading-Assist Agent — Phased MVP Implementation Plan

## Title
ResearchPaper-reading-Assist Agent — Phased MVP Plan

## TL;DR
Implement a LangChain-based ResearchPaper-reading-Assist Agent in five phases: ingestion/indexing, Q&A with citations, persistent memory, multi-paper synthesis, and a ReAct research agent. Each phase delivers a small, testable surface (loaders, retrievers, chains, memory, agent) with fast unit tests and mocked LLM/vector DB calls to keep development deterministic and reviewable.

## Steps

### Phase 1: Foundation — Local PDF Ingestion & Vector Index
**Objective:** Read all PDF research papers from a user-specified local folder, parse and chunk them, generate embeddings, and store everything in a Chroma vector index.

**Concrete Tasks:**
- Create a folder scanner that discovers all `.pdf` files in a given directory (recursively)
- Create PDF loader module using PyPDFLoader to extract text and metadata (title, authors, page count) from each PDF
- Implement recursive text splitter with configured chunk sizes (chunk_size=1000, overlap=200)
- Add embeddings wrapper (sentence-transformers for local/offline use; Gemini as optional)
- Persist chunks, embeddings, and per-paper metadata to Chroma vector store
- Build a CLI entry point that accepts a folder path, scans for PDFs, and runs the full ingestion pipeline with progress logging

**Acceptance Criteria:**
- User points to a local folder containing research paper PDFs
- All PDFs in the folder are discovered, parsed, chunked, and embedded
- Chunks stored in Chroma with embeddings and metadata (source file, title, page number)
- Ingestion logs progress per file and completes <30s per paper (local test)
- Running the CLI a second time on the same folder skips already-ingested papers (deduplication by file path + hash)

**Files to Create:**
- `src/ingestion/__init__.py` — ingestion package entry
- `src/ingestion/scanner.py` — recursively discover PDF files in a given folder
- `src/ingestion/loader.py` — PyPDFLoader wrapper with metadata extraction (title, authors, page count)
- `src/ingestion/splitter.py` — RecursiveCharacterTextSplitter config
- `src/ingestion/pipeline.py` — orchestrates scan → load → split → embed → store for a folder
- `src/core/embeddings.py` — embeddings factory (sentence-transformers default, Gemini optional)
- `src/core/vectorstore.py` — Chroma integration and helper functions
- `cli.py` — CLI entry point: accepts `--folder` path, runs ingestion pipeline
- `tests/test_scanner.py` — Unit tests for PDF discovery in nested folders, skipping non-PDF files
- `tests/test_ingestion_loader.py` — Mocks PyPDFLoader; asserts metadata extraction and returned text chunks
- `tests/test_splitter.py` — Unit tests for chunk sizes and overlap behavior using synthetic text
- `tests/test_vectorstore_integration.py` — Mocks embeddings and Chroma; asserts embeddings and metadata are stored
- `tests/test_pipeline.py` — End-to-end test of folder ingestion with mocked loader and vector store

---

### Phase 2: Core Q&A with Citations
**Objective:** Build retrieval pipeline, RetrievalQA chain, citation-aware prompt templates, and a simple Streamlit UI for querying.

**Concrete Tasks:**
- Create LLM factory (`src/core/llm.py`) supporting Gemini and Ollama via a `--llm-provider` config, with a `fake` mode for testing
- Implement Retriever wrapper (MultiQuery + Compression)
- Create RetrievalQA chain with citation-preserving prompts
- Wire chat models with streaming callback support
- Add conversation buffer memory for short-term context
- Add Streamlit query UI and response streaming with model selector (Gemini / Ollama)

**Acceptance Criteria:**
- LLM provider is configurable: Gemini (cloud, default), Ollama (local/offline), or fake (testing)
- Queries return synthesized answers with paper title+authors+location citations
- Follow-up questions use conversation context
- Unknown queries return explicit "I don't know" responses

**Files to Create:**
- `src/core/llm.py` — LLM factory: returns `ChatGoogleGenerativeAI` (Gemini), `ChatOllama` (Ollama), or `FakeListChatModel` (testing) based on provider config
- `src/qna/qa.py` — retrieval and RetrievalQA chain setup
- `src/qna/prompts.py` — citation-aware prompt templates
- `src/ui/streamlit_app.py` — Streamlit UI for questions and streaming, with LLM provider selector in sidebar
- `src/core/retriever.py` — MultiQueryRetriever & ContextualCompressionRetriever wiring
- `tests/test_llm_factory.py` — Unit tests for LLM factory: asserts correct model type returned per provider, fake mode works without API keys
- `tests/test_retriever.py` — Mocks vector search to assert retriever returns expected doc ids and metadata
- `tests/test_retrievalqa_citations.py` — Mocks LLM; asserts generated answer includes title/author/location citation formatting
- `tests/test_streaming_callback.py` — Unit test that streaming callback is invoked and yields partial chunks

---

### Phase 3: Memory Layer & Persistence
**Objective:** Persist conversation history, research interests, idea log, and notes & insights using VectorStoreRetrieverMemory, entity summaries, and dedicated JSON-backed stores.

**Concrete Tasks:**
- Implement VectorStoreRetrieverMemory integration with Chroma
- Add ConversationSummaryMemory for long sessions
- Implement EntityMemory for authors/papers/topics
- Add Idea Log: a persistent, searchable log where users can save research ideas via natural language (e.g., "I have this idea about using LQ loss for image tasks. Add it in idea log"). Each entry stores the idea text, timestamp, optional tags (auto-extracted or user-specified), and links to related papers if mentioned. Users can browse, search, edit, and delete ideas from the log.
- Add Notes & Insights Log: a separate persistent log for the user's personal notes, observations, and insights (e.g., "Add this to notes: 'Only for symmetric loss, MAE is useful'"). Unlike the idea log (which captures forward-looking research ideas), notes capture factual observations, learnings, and takeaways. Each entry stores the note text, timestamp, and optional tags. Persisted in a dedicated JSON file (`notes_and_insights.json`). Users can add, view, search, edit, and delete notes.
- Add chat intent detection to distinguish between: idea log commands ("Add to idea log: ..."), notes commands ("Add this to notes: ..."), display commands ("Show my notes", "Show my idea log"), and regular questions.
- Add display cap: when the user asks to "show my notes" or "show my idea log" in chat, display only the most recent N entries (default N=10, configurable via `DISPLAY_CAP` in .env). Show a count of total entries and a hint to use search for older items.
- Add endpoints to view and clear memory
- Add endpoints to list, search, add, edit, and delete idea log entries
- Add endpoints to list, search, add, edit, and delete notes entries
- Add privacy controls and tests for clearing memory

**Acceptance Criteria:**
- Conversation history persists across sessions
- System can recall previously discussed papers and topics
- User can clear memory and subsequent queries forget cleared context
- User can add ideas to the log via natural language commands in the chat (e.g., "Add to idea log: use LQ loss for image segmentation tasks")
- Idea log entries are persisted across sessions with timestamp and auto-extracted tags
- User can browse all ideas, search by keyword/tag, edit, and delete entries
- Ideas referencing ingested papers are automatically linked to those papers
- User can add notes via natural language commands in chat (e.g., "Add this to notes: 'Only for symmetric loss, MAE is useful'")
- Notes & insights entries are persisted across sessions with timestamp and optional tags
- User can browse, search, edit, and delete notes
- Chat commands "Show my notes" and "Show my idea log" display the most recent entries (capped at `DISPLAY_CAP`, default 10) with total count
- Intent detection correctly routes "add to notes", "add to idea log", "show my notes", "show my idea log" commands

**Files to Create:**
- `src/memory/memory.py` — memory implementations and persistence adapters
- `src/memory/idea_log.py` — Idea log storage, retrieval, search, edit, and delete operations. Each idea entry: `{id, text, tags[], timestamp, related_papers[], created_at, updated_at}`. Persisted in Chroma with a dedicated collection. Supports intent detection to recognize "add to idea log" commands from chat input.
- `src/memory/notes_log.py` — Notes & Insights log storage, retrieval, search, edit, and delete operations. Each note entry: `{id, text, tags[], created_at, updated_at}`. Persisted in a JSON file (`notes_and_insights.json`). Supports intent detection for "add this to notes" commands.
- `src/memory/intent_detector.py` — Detects user intent from chat input: `add_idea`, `add_note`, `show_ideas`, `show_notes`, `search_ideas`, `search_notes`, or `question` (default). Returns `{intent, extracted_text, params}`.
- `src/api/memory_api.py` — endpoints for viewing/clearing memory
- `src/api/idea_log_api.py` — REST endpoints: `GET /ideas` (list/search with `?q=` and `?tag=`), `POST /ideas` (add), `PUT /ideas/{id}` (edit), `DELETE /ideas/{id}` (delete)
- `src/api/notes_api.py` — REST endpoints: `GET /notes` (list/search with `?q=` and `?tag=`, supports `?limit=N`), `POST /notes` (add), `PUT /notes/{id}` (edit), `DELETE /notes/{id}` (delete)
- `docs/memory.md` — memory behavior, privacy controls, idea log usage, and notes & insights usage
- `tests/test_memory_persistence.py` — Mocks Chroma-backed memory; asserts conversation persists and is queryable
- `tests/test_memory_clear.py` — Asserts memory clear endpoint wipes stored memory (using in-memory stub)
- `tests/test_idea_log.py` — Tests for adding ideas via natural language, auto-tag extraction, search by keyword/tag, edit, delete, and paper linking
- `tests/test_idea_log_api.py` — Tests for idea log REST endpoints (CRUD operations and search)
- `tests/test_notes_log.py` — Tests for adding notes via natural language, search by keyword/tag, edit, delete, and display cap behavior
- `tests/test_notes_api.py` — Tests for notes REST endpoints (CRUD operations and search)
- `tests/test_intent_detector.py` — Tests for intent detection: correctly classifies "add to idea log", "add this to notes", "show my notes", "show my idea log", and regular questions

---

### Phase 4: Multi-Paper Synthesis
**Objective:** Implement MapReduce/Refine chains and self-query retriever to synthesize comparative summaries across the collection.

**Concrete Tasks:**
- Add MapReduceDocumentsChain and RefineDocumentsChain wrappers
- Implement SelfQueryRetriever with metadata filters
- Create few-shot prompts for comparative summaries
- Add UI flow to request cross-paper synthesis
- Benchmark synthesis latency and refine chunking if needed

**Acceptance Criteria:**
- Synthesis covers multiple papers and highlights themes/contradictions
- Support metadata filters (year, author, topic)
- Summaries are coherent and traceable to source papers

**Files to Create:**
- `src/synthesis/synthesizer.py` — MapReduce and Refine chain implementations
- `src/synthesis/filters.py` — metadata filter utilities and SelfQueryRetriever config
- `src/ui/synthesis_view.py` — UI for multi-paper synthesis
- `docs/synthesis.md` — prompt design and examples
- `tests/test_mapreduce_chain.py` — Mocks LLM responses for map/reduce stages; asserts final synthesized structure and source trace
- `tests/test_selfquery_filters.py` — Unit tests for metadata filters selecting expected documents

---

### Phase 5: Smart Research Agent
**Objective:** Add a ReAct agent with tool integrations (arXiv, web search, Python REPL) and agent tracing for multi-step tasks.

**Concrete Tasks:**
- Implement tool wrappers: ArxivQueryRun, DuckDuckGoSearchRun, PythonREPL
- Implement create_react_agent and AgentExecutor wiring
- Add AgentCallbackHandler to log reasoning and tool use
- Add UI to start agent tasks and stream reasoning traces
- Add safety and iteration limits for agent runs

**Acceptance Criteria:**
- Agent plans and executes multi-step research tasks using tools
- Agent logs (tool, reason, result) for each step
- Agent asks clarifying questions when instructions are ambiguous

**Files to Create:**
- `src/agent/agent.py` — ReAct agent factory and executor
- `src/agent/tools.py` — tool adapters for arXiv, web search, PythonREPL, paper retriever
- `src/ui/agent_dashboard.py` — UI for launching agent tasks and viewing traces
- `docs/agent.md` — agent persona, safety, and iteration limits
- `tests/test_agent_tooling.py` — Mocks tool outputs and asserts agent invokes correct tool sequence for a simple task
- `tests/test_agent_trace_logging.py` — Asserts reasoning trace entries are recorded with tool, input, and output fields

---

## RAG Performance Improvement Plan

**Problem Statement:** Current RAG system (Phase 2) has poor retrieval accuracy. Example: querying "What is the main idea in the paper 'Gradient-based learning applied to document recognition'" returns irrelevant papers about noisy labels and cross-modal learning. Root causes: weak vector search, generic embedding model, suboptimal chunking, and weak LLM prompting.

**Solution:** Three-phase improvement plan targeting retrieval quality, ranking, and embeddings.

---

### RAG Phase 1: Immediate — Hybrid Retrieval & Prompt Enforcement

**Objective:** Quick wins to improve title matching and LLM citation awareness. Target time: 30 minutes.

**Concrete Tasks:**
- Add BM25 (sparse) retrieval to hybrid search with vector (dense) retrieval using EnsembleRetriever
- Weight ensemble: 60% vector + 40% BM25 (tune empirically)
- Create title-enforcing QA prompt that asks LLM to prioritize paper title/author matches
- Increase retriever `k` parameter from 3 to 5 for first-pass candidate pool
- Add fallback logic: if answer confidence is low, return "Paper found but insufficient excerpts" instead of hallucinating

**Acceptance Criteria:**
- BM25 retriever is integrated and weighted in ensemble retriever
- Title-based queries (e.g., "main idea in X paper") retrieve the correct paper in top-3 results
- QA prompt enforces: "If the question mentions a paper title, prioritize results matching that title"
- No hallucinated answers for unknown papers; explicit fallback message is returned

**Files to Create/Modify:**
- `src/core/retriever.py` — add BM25Retriever and EnsembleRetriever with 60/40 weighting
- `src/qna/prompts.py` — add `TITLE_ENFORCING_QA_PROMPT` that prioritizes title matches and enforces "paper found but insufficient" fallback
- `src/qna/qa.py` — integrate new prompt and increase retriever `k=5`
- `tests/test_retriever_hybrid.py` — assert BM25 + vector ensemble returns expected docs in correct order
- `tests/test_qa_prompt_enforcement.py` — mock LLM and assert prompt causes title-prioritization behavior

---

### RAG Phase 2: Short-Term — Query Expansion & Reranking

**Objective:** Improve retrieval recall and ranking. Target time: 1-2 hours.

**Concrete Tasks:**
- Add MultiQueryRetriever to reformulate user queries into 3-5 variations (e.g., "gradient-based learning" → "gradient descent methods", "learning by backpropagation", "automatic differentiation applied to", etc.)
- Implement contextual compression with a CrossEncoderReranker to re-score and reorder retrieved documents by relevance
- Use `cross-encoder/ms-marco-MiniLM-L-12-v2` model (lightweight, fast)
- Increase retriever `k` from 5 to 10 in first-pass retrieval, then rerank to top 5
- Add confidence scoring: if top reranked score is below threshold (e.g., 0.5), warn user that retrieval confidence is low

**Acceptance Criteria:**
- MultiQueryRetriever generates 3-5 query variations for diverse retrieval
- Reranker re-scores and reorders retrieved docs, top-1 becomes more relevant than original top-1
- For the original failing query about "Gradient-based learning", reranked results should include the correct paper or similar papers
- Confidence scores are computed and displayed to user
- Latency increase is <2 seconds for reranking (benchmark on 10-doc set)

**Files to Create/Modify:**
- `src/core/retriever.py` — add MultiQueryRetriever, ContextualCompressionRetriever with CrossEncoderReranker
- `src/qna/qa.py` — wire multi-query + reranking retriever, add confidence scoring to QA chain output
- `src/ui/streamlit_app.py` — display retrieval confidence score next to answer
- `tests/test_multiquery_retriever.py` — assert MultiQueryRetriever generates varied queries
- `tests/test_reranker.py` — assert CrossEncoderReranker reorders docs and improves top-1 relevance
- `tests/test_qa_confidence_scoring.py` — assert confidence score is computed and returned

---

### RAG Phase 3: Medium-Term — Domain-Specific Embeddings & Structured Chunking

**Objective:** Improve semantic understanding of academic papers. Target time: half day.

**Concrete Tasks:**
- Switch from generic sentence-transformers to domain-specific embedding model: `allenai/specter` (trained on 100k arXiv papers and citations) or `nomic-ai/nomic-embed-text-v1.5`
- Refactor chunking strategy: detect paper structure (Abstract, Introduction, Methods, Results, Conclusion) and extract each section as a separate high-priority chunk with metadata tag `chunk_type=abstract|introduction|conclusion`
- Add metadata filtering using SelfQueryRetriever to allow filtering by chunk type, so queries like "what is the main idea" prioritize abstract+introduction chunks
- Implement deduplication: after reranking, filter out duplicate-content chunks to avoid repeating the same information
- Add structured metadata to each chunk: `{paper_title, authors, year, section, chunk_type, page_range}`

**Acceptance Criteria:**
- Embeddings switched to specter/nomic, vector store re-embedded with new model
- Papers are parsed to extract Abstract, Introduction, Methods, Results, Conclusion as separate indexed sections
- SelfQueryRetriever can filter by `chunk_type` and `section` metadata
- Queries about "main idea / contribution / hypothesis" prioritize abstract+introduction chunks (metadata filter: `chunk_type in [abstract, introduction]`)
- For the original failing query, retrieval now returns abstract of the correct paper
- Duplicate-detection filter removes redundant results

**Files to Create/Modify:**
- `src/core/embeddings.py` — switch to `allenai/specter` and re-embed all ingested papers (add migration script)
- `src/ingestion/splitter.py` — add structural parsing to detect sections (Abstract, Intro, Methods, etc.) and create tagged chunks
- `src/core/retriever.py` — add SelfQueryRetriever with metadata filters for chunk_type and section
- `src/core/vectorstore.py` — add deduplication utility to filter redundant chunks post-retrieval
- `cli.py` — add `--re-embed` flag to trigger re-embedding with new model for existing papers
- `tests/test_paper_structure_extraction.py` — assert Abstract, Intro, Methods are correctly extracted
- `tests/test_selfquery_metadata_filtering.py` — assert SelfQueryRetriever filters by chunk_type and section
- `tests/test_deduplication.py` — assert duplicate chunks are filtered out

---

## Test Plans

### Phase 1 Test Plan: Foundation — Local PDF Ingestion & Vector Index

**Prerequisites:**
- Place 3–5 sample PDFs (varying page counts) in `examples/sample_papers/`
- Install dependencies: `pip install -r requirements.txt`
- Ensure no existing Chroma DB at the default persist path

**Step 1 — Unit: Scanner discovery** (`tests/test_scanner.py`)
```bash
pytest tests/test_scanner.py -v
```
- Create a temp directory with nested subfolders containing `.pdf`, `.txt`, and `.docx` files
- Call `scan_folder(tmp_dir)` and assert only `.pdf` paths are returned
- Assert results include PDFs from nested subdirectories
- Assert an empty folder returns an empty list
- Assert a non-existent folder raises `FileNotFoundError`

**Step 2 — Unit: PDF loader & metadata extraction** (`tests/test_ingestion_loader.py`)
```bash
pytest tests/test_ingestion_loader.py -v
```
- Mock `PyPDFLoader.load()` to return synthetic `Document` objects with page content and metadata
- Call `load_pdf(path)` and assert returned documents contain expected text
- Assert metadata dict includes keys: `source`, `title`, `authors`, `page`, `total_pages`
- Assert a corrupted/unreadable PDF path raises a handled exception (not a crash)

**Step 3 — Unit: Text splitter** (`tests/test_splitter.py`)
```bash
pytest tests/test_splitter.py -v
```
- Feed a 3000-character synthetic string into `split_text(text, chunk_size=1000, overlap=200)`
- Assert output chunks are each ≤1000 characters
- Assert consecutive chunks overlap by ~200 characters (check last 200 chars of chunk N == first 200 chars of chunk N+1)
- Assert a string shorter than `chunk_size` returns a single chunk unchanged

**Step 4 — Integration: Vector store storage** (`tests/test_vectorstore_integration.py`)
```bash
pytest tests/test_vectorstore_integration.py -v
```
- Use an in-memory Chroma instance (or mock)
- Call `store_chunks(chunks, embeddings, metadata)` with 5 synthetic chunks
- Query the store and assert all 5 chunks are retrievable
- Assert each stored document has metadata fields: `source`, `title`, `page`
- Assert duplicate insertion (same content + metadata) does not create duplicates

**Step 5 — Integration: Full pipeline** (`tests/test_pipeline.py`)
```bash
pytest tests/test_pipeline.py -v
```
- Mock the PDF loader and embeddings model
- Point pipeline at a temp folder with 2 mock PDFs
- Run `ingest_folder(folder_path)` and assert both PDFs are processed
- Assert Chroma store contains chunks from both files
- Run pipeline a second time on the same folder and assert zero new chunks are added (deduplication)

**Step 6 — Manual: CLI end-to-end**
```bash
python cli.py --folder examples/sample_papers/
```
- Verify progress logs print per file (file name, chunk count, time)
- Verify Chroma persist directory is created with data
- Run the same command again and verify logs say "skipping already-ingested" for each file
- Run `python cli.py --folder /nonexistent/path` and verify a clear error message

---

### Phase 2 Test Plan: Core Q&A with Citations

**Prerequisites:**
- Phase 1 complete with sample papers ingested into Chroma
- LLM API key set in `.env` (or mock mode enabled)

**Step 1 — Unit: Retriever returns relevant docs** (`tests/test_retriever.py`)
```bash
pytest tests/test_retriever.py -v
```
- Mock the vector store's `similarity_search` to return 3 predefined documents with metadata
- Call `retrieve(query="attention mechanism", k=3)`
- Assert exactly 3 documents are returned
- Assert each document has `source`, `title`, `page` in metadata
- Assert retriever handles an empty result set gracefully (returns empty list, no crash)

**Step 2 — Unit: Citation formatting in QA chain** (`tests/test_retrievalqa_citations.py`)
```bash
pytest tests/test_retrievalqa_citations.py -v
```
- Mock the LLM to return a canned answer string
- Mock the retriever to return documents with known metadata (title="Attention Is All You Need", authors="Vaswani et al.", page=5)
- Call `ask(question="What is self-attention?")`
- Assert the response string contains citation markers matching the format `[Attention Is All You Need, Vaswani et al., p.5]`
- Assert a query with no relevant documents returns a response containing "I don't know" or equivalent

**Step 3 — Unit: Streaming callback** (`tests/test_streaming_callback.py`)
```bash
pytest tests/test_streaming_callback.py -v
```
- Create a mock streaming callback handler
- Simulate LLM token generation by calling `on_llm_new_token("Hello")`, `on_llm_new_token(" world")`
- Assert the callback accumulates tokens in order
- Assert `on_llm_end` is called once at the end

**Step 4 — Manual: Streamlit UI**
```bash
streamlit run src/ui/streamlit_app.py
```
- Type a question about an ingested paper (e.g., "What are the key findings of [paper title]?")
- Verify the answer streams token-by-token in the UI
- Verify citations appear with paper title, authors, and page/section
- Ask a follow-up question referencing the previous answer (e.g., "Can you elaborate on point 2?")
- Verify the follow-up uses conversation context (doesn't ask you to re-specify the paper)
- Ask a question about a topic not in any ingested paper
- Verify the response explicitly states it doesn't have enough information

---

### Phase 3 Test Plan: Memory Layer & Persistence

**Prerequisites:**
- Phases 1–2 complete and functional
- Memory API server can start (FastAPI/uvicorn)

**Step 1 — Unit: Conversation memory persistence** (`tests/test_memory_persistence.py`)
```bash
pytest tests/test_memory_persistence.py -v
```
- Create an in-memory Chroma-backed `VectorStoreRetrieverMemory`
- Save 3 conversation exchanges: `memory.save_context({"input": "Q1"}, {"output": "A1"})` etc.
- Query memory with `memory.load_memory_variables({"input": "topic from Q2"})`
- Assert the relevant exchange (Q2/A2) appears in the returned variables
- Simulate a "new session" by re-instantiating memory with the same Chroma persist path
- Assert previously saved conversations are still retrievable

**Step 2 — Unit: Memory clear** (`tests/test_memory_clear.py`)
```bash
pytest tests/test_memory_clear.py -v
```
- Save 3 conversation exchanges to memory
- Call `clear_memory()` or the clear API endpoint
- Query memory for any previously saved topic
- Assert the result is empty (no conversations returned)
- Save a new exchange after clearing and assert it is retrievable (memory still functional)

**Step 3 — Unit: Idea log operations** (`tests/test_idea_log.py`)
```bash
pytest tests/test_idea_log.py -v
```
- Call `add_idea("Use LQ loss for image segmentation tasks")` and assert an entry is created with auto-generated tags (e.g., `["LQ loss", "image segmentation"]`), a timestamp, and a unique ID
- Add an idea mentioning an ingested paper title and assert `related_papers` is populated with the matching paper ID
- Call `search_ideas(query="LQ loss")` and assert the matching idea is returned
- Call `search_ideas(tag="image segmentation")` and assert tag-based filtering works
- Call `edit_idea(id, new_text="...")` and assert the text and `updated_at` are changed
- Call `delete_idea(id)` and assert the idea is no longer retrievable
- Test intent detection: pass `"I have this idea about using contrastive loss for audio. Add it in idea log"` through the intent detector and assert it extracts the idea text and triggers an add operation

**Step 4 — Unit: Idea log API** (`tests/test_idea_log_api.py`)
```bash
pytest tests/test_idea_log_api.py -v
```
- `POST /ideas` with `{"text": "Use LQ loss for image tasks"}` — assert 201 and returned entry has `id`, `tags`, `timestamp`
- `GET /ideas` — assert the list contains the added idea
- `GET /ideas?q=LQ+loss` — assert search returns the matching idea
- `GET /ideas?tag=image` — assert tag filter works
- `PUT /ideas/{id}` with updated text — assert 200 and text is changed
- `DELETE /ideas/{id}` — assert 200 and subsequent GET returns empty

**Step 5 — Unit: Notes & Insights log operations** (`tests/test_notes_log.py`)
```bash
pytest tests/test_notes_log.py -v
```
- Call `add_note("Only for symmetric loss, MAE is useful")` and assert an entry is created with a unique ID, timestamp, and auto-extracted tags (e.g., `["symmetric loss", "MAE"]`)
- Call `get_notes(limit=10)` and assert the note appears in the list
- Add 15 notes, call `get_notes(limit=10)` and assert only the 10 most recent are returned, with `total_count=15`
- Call `search_notes(query="MAE")` and assert the matching note is returned
- Call `edit_note(id, new_text="...")` and assert the text and `updated_at` are changed
- Call `delete_note(id)` and assert the note is no longer retrievable

**Step 6 — Unit: Notes API** (`tests/test_notes_api.py`)
```bash
pytest tests/test_notes_api.py -v
```
- `POST /notes` with `{"text": "Only for symmetric loss, MAE is useful"}` — assert 201 and returned entry has `id`, `tags`, `created_at`
- `GET /notes` — assert the list contains the added note
- `GET /notes?q=MAE` — assert search returns the matching note
- `GET /notes?limit=5` — assert at most 5 entries returned with `total_count` field
- `PUT /notes/{id}` with updated text — assert 200 and text is changed
- `DELETE /notes/{id}` — assert 200 and subsequent GET returns empty

**Step 7 — Unit: Intent detector** (`tests/test_intent_detector.py`)
```bash
pytest tests/test_intent_detector.py -v
```
- Input: `"Add to idea log: use contrastive loss for audio"` → assert intent=`add_idea`, extracted_text=`"use contrastive loss for audio"`
- Input: `"Add this to notes: 'Only for symmetric loss, MAE is useful'"` → assert intent=`add_note`, extracted_text=`"Only for symmetric loss, MAE is useful"`
- Input: `"Show my notes"` → assert intent=`show_notes`
- Input: `"Show my idea log"` → assert intent=`show_ideas`
- Input: `"What is self-attention?"` → assert intent=`question`
- Input: `"Search my notes for MAE"` → assert intent=`search_notes`, params contain query=`"MAE"`

**Step 8 — Manual: Memory API endpoints**
```bash
uvicorn src.api.memory_api:app --reload
```
- `GET /memory` — verify it returns stored conversation history as JSON
- `DELETE /memory` — call it and verify response is 200/success
- `GET /memory` again — verify the list is now empty
- Ask a question via the QA chain, then `GET /memory` and verify the new exchange appears

**Step 9 — Manual: Idea log & Notes via chat**
- In Streamlit, type: "I have this idea about using LQ loss for image tasks. Add it in idea log"
- Verify the system confirms the idea was saved and shows the extracted tags
- Type: "Add this to notes: 'Only for symmetric loss, MAE is useful'"
- Verify the system confirms the note was saved
- Type: "Show my idea log" — verify the most recent ideas appear (capped at 10) with timestamp and tags
- Type: "Show my notes" — verify the most recent notes appear (capped at 10) with timestamp
- Type: "Search my ideas for LQ loss" — verify the matching idea is returned
- Type: "Search my notes for MAE" — verify the matching note is returned
- Use the sidebar panels to edit and delete ideas and notes

**Step 10 — Manual: Display cap behavior**
- Add 15+ notes via chat or API
- Type: "Show my notes" — verify only 10 most recent are shown with a message like "Showing 10 of 15 notes. Use search to find older entries."

**Step 11 — Manual: Cross-session persistence**
- Start Streamlit, ask: "Tell me about transformer architectures"
- Note the answer, then stop the Streamlit server
- Restart Streamlit, ask: "What did we discuss earlier?"
- Verify the system recalls the transformer architecture discussion
- Clear memory via the API, restart Streamlit, ask: "What did we discuss earlier?"
- Verify the system does NOT recall previous conversations
- Verify idea log and notes persist across restarts independently of memory clear

---

### Phase 4 Test Plan: Multi-Paper Synthesis

**Prerequisites:**
- Phases 1–3 complete
- At least 3–5 papers ingested covering overlapping topics

**Step 1 — Unit: MapReduce chain** (`tests/test_mapreduce_chain.py`)
```bash
pytest tests/test_mapreduce_chain.py -v
```
- Create 4 synthetic documents (simulating chunks from 4 different papers)
- Mock the LLM to return canned summaries for the map stage and a combined summary for the reduce stage
- Call `synthesize(docs, mode="map_reduce")`
- Assert the map stage is called once per document (4 calls)
- Assert the reduce stage is called once with all map outputs
- Assert the final output references all 4 source papers
- Assert each source paper's title appears in the output's source trace

**Step 2 — Unit: SelfQuery metadata filters** (`tests/test_selfquery_filters.py`)
```bash
pytest tests/test_selfquery_filters.py -v
```
- Populate a mock vector store with documents having metadata: `{year: 2020, author: "Smith", topic: "NLP"}`, `{year: 2023, author: "Jones", topic: "CV"}`
- Call `filter_query(query="NLP papers by Smith")` which should produce filter `{author: "Smith", topic: "NLP"}`
- Assert only the matching document is returned
- Call `filter_query(query="papers after 2022")` and assert only the 2023 paper is returned
- Call `filter_query(query="all papers")` with no filters and assert all documents are returned

**Step 3 — Manual: Synthesis UI**
```bash
streamlit run src/ui/synthesis_view.py
```
- Enter: "Compare the approaches to attention mechanisms across all ingested papers"
- Verify the output covers multiple papers, not just one
- Verify themes and/or contradictions are highlighted
- Verify each claim in the synthesis is traceable to a specific paper (citation present)
- Apply a filter (e.g., year ≥ 2022) and re-run synthesis
- Verify only papers matching the filter are included in the output
- Enter a synthesis query on a topic with no matching papers
- Verify a clear message indicating insufficient sources

---

### Phase 5 Test Plan: Smart Research Agent

**Prerequisites:**
- Phases 1–4 complete
- API keys for arXiv and DuckDuckGo search available (or mock mode)

**Step 1 — Unit: Agent tool invocation sequence** (`tests/test_agent_tooling.py`)
```bash
pytest tests/test_agent_tooling.py -v
```
- Mock all tools: `ArxivQueryRun` returns a canned paper summary, `DuckDuckGoSearchRun` returns a canned search snippet, `PythonREPL` returns a canned computation result, paper retriever returns stored docs
- Give the agent a task: "Find recent papers on RLHF and summarize the top 3"
- Assert the agent calls `ArxivQueryRun` at least once
- Assert the agent calls the paper retriever tool
- Assert the agent produces a final answer (not stuck in a loop)
- Assert the agent stops within the configured iteration limit

**Step 2 — Unit: Reasoning trace logging** (`tests/test_agent_trace_logging.py`)
```bash
pytest tests/test_agent_trace_logging.py -v
```
- Run the agent on a simple task with mocked tools
- Capture the trace log output
- Assert each trace entry has fields: `step_number`, `tool_name`, `tool_input`, `tool_output`, `reasoning`
- Assert the trace has at least 2 entries (thought + tool use)
- Assert the final entry is marked as `final_answer`

**Step 3 — Unit: Safety and iteration limits**
```bash
pytest tests/test_agent_tooling.py -v -k "test_iteration_limit"
```
- Mock a tool to always return "need more info" (forcing the agent to loop)
- Set `max_iterations=5`
- Assert the agent stops after exactly 5 iterations
- Assert the agent returns a graceful timeout message (not an exception)

**Step 4 — Manual: Agent dashboard**
```bash
streamlit run src/ui/agent_dashboard.py
```
- Enter task: "Find the 3 most cited papers on diffusion models from 2023 and compare their methods"
- Verify the reasoning trace panel shows each step (Thought → Tool → Observation)
- Verify the agent uses arXiv search and paper retriever tools
- Verify the final answer synthesizes information from multiple sources
- Enter an ambiguous task: "Tell me about that thing"
- Verify the agent asks a clarifying question instead of guessing
- Monitor that the agent does not exceed the iteration limit displayed in the UI

**Step 5 — Manual: Full pipeline integration**
```bash
# Ingest papers first
python cli.py --folder examples/sample_papers/
# Launch the full app
streamlit run src/ui/agent_dashboard.py
```
- Ask the agent to "Summarize all ingested papers and find 2 related papers on arXiv"
- Verify it reads from the local vector store AND queries arXiv
- Verify the final output distinguishes between local papers and newly found papers
- Verify citations are present for all referenced papers

---

## Further Considerations

- **Configuration & Secrets:** Add configuration and secrets management (dotenv) and local dev mode to stub LLM/vector calls. This enables isolated testing without API keys.
- **CI/Testing:** Use CI to run pytest with mocked external calls; add benchmark jobs for ingestion and synthesis latencies to track performance over time.
- **LLM Providers:** Support Gemini (cloud, default) and Ollama (local/offline) via a single `src/core/llm.py` factory. All phases (2–5) use this factory — switching providers is a config change, not a code change. Ollama requires local install + model pull; Gemini requires an API key.
- **Initial Tech Stack:** Start with Gemini embeddings + Chroma local server; keep agent tool usage gated behind safety/iteration limits to prevent runaway costs and tokens.

---

## Streamlit UI Design

### Page 1: Q&A — `src/ui/streamlit_app.py` (Phase 2)

```
┌──────────────────────────────────────────────────────────────────┐
│  📚 ResearchPaper-reading-Assist Agent                        [⚙ Settings] │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌── Sidebar ──────────────────┐  ┌── Main Area ──────────────┐ │
│  │                             │  │                            │ │
│  │  📁 Ingestion               │  │  Chat History              │ │
│  │  ┌────────────────────────┐ │  │                            │ │
│  │  │ Folder path...     [📂]│ │  │  🧑 What are the key       │ │
│  │  └────────────────────────┘ │  │     findings of the        │ │
│  │  [Ingest Papers]            │  │     Attention paper?       │ │
│  │                             │  │                            │ │
│  │  Status: 12 papers indexed  │  │  🤖 The key findings are:  │ │
│  │                             │  │     1. Self-attention...   │ │
│  │  ─────────────────────────  │  │     2. Multi-head...       │ │
│  │                             │  │                            │ │
│  │  📄 Ingested Papers         │  │     **Sources:**           │ │
│  │  ┌────────────────────────┐ │  │     [Attention Is All You  │ │
│  │  │ ▸ Attention Is All...  │ │  │      Need, Vaswani et al., │ │
│  │  │ ▸ BERT: Pre-training...│ │  │      p.5]                 │ │
│  │  │ ▸ GPT-4 Technical...   │ │  │     [BERT, Devlin et al., │ │
│  │  │ ▸ ...                  │ │  │      p.12]                │ │
│  │  └────────────────────────┘ │  │                            │ │
│  │                             │  │  🧑 Can you elaborate on   │ │
│  │  🔧 Settings                │  │     point 2?               │ │
│  │  Model: [Gemini ▾]         │  │                            │ │
│  │  Top-K: [5      ]          │  │  🤖 ▌ (streaming response)  │ │
│  │  Temperature: [0.1]        │  │                            │ │
│  │                             │  │                            │ │
│  │  🧠 Memory                  │  ├────────────────────────────┤ │
│  │  [View Memory]              │  │ ┌────────────────────┐     │ │
│  │  [Clear Memory]             │  │ │ Ask a question...  │ [➤] │ │
│  │                             │  │ └────────────────────┘     │ │
│  │  ─────────────────────────  │  │                            │ │
│  │                             │  │                            │ │
│  │  💡 Idea Log                │  │                            │ │
│  │  ┌────────────────────────┐ │  │                            │ │
│  │  │ ▸ LQ loss for image   │ │  │                            │ │
│  │  │   tasks  [edit][×]    │ │  │                            │ │
│  │  │ ▸ Contrastive pre-    │ │  │                            │ │
│  │  │   training for...     │ │  │                            │ │
│  │  └────────────────────────┘ │  │                            │ │
│  │  [Search ideas...       🔍] │  │                            │ │
│  │                             │  │                            │ │
│  │  ─────────────────────────  │  │                            │ │
│  │                             │  │                            │ │
│  │  📝 Notes & Insights        │  │                            │ │
│  │  ┌────────────────────────┐ │  │                            │ │
│  │  │ ▸ Only for symmetric  │ │  │                            │ │
│  │  │   loss, MAE is useful │ │  │                            │ │
│  │  │   [edit][×]           │ │  │                            │ │
│  │  │ ▸ Cross-attention     │ │  │                            │ │
│  │  │   works better for... │ │  │                            │ │
│  │  └────────────────────────┘ │  │                            │ │
│  │  Showing 2 of 2 notes      │  │                            │ │
│  │  [Search notes...       🔍] │  │                            │ │
│  │                             │  │                            │ │
│  └─────────────────────────────┘  └────────────────────────────┘ │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Layout Details:**
- **Sidebar — Ingestion Controls:** Folder path input with a browse button; "Ingest Papers" button triggers the Phase 1 pipeline with a progress bar shown inline. Below it, a count of total indexed papers and an expandable list of ingested paper titles.
- **Sidebar — Settings:** Dropdowns/sliders for model selection, retriever top-K, and temperature. These apply to all subsequent queries.
- **Sidebar — Memory:** Buttons to view stored conversation memory (opens a modal/expander) and to clear memory with a confirmation dialog.
- **Sidebar — Idea Log:** An expandable list of saved ideas with inline edit/delete buttons. A search bar to filter ideas by keyword or tag. Ideas can also be added via chat commands (e.g., "Add to idea log: ...") — the system detects the intent and saves the idea automatically.
- **Sidebar — Notes & Insights:** A separate expandable list of personal notes and insights with inline edit/delete buttons. A search bar to filter notes by keyword. Notes can be added via chat commands (e.g., "Add this to notes: 'Only for symmetric loss, MAE is useful'"). Shows a count of displayed vs total entries. Both the sidebar list and chat display are capped at `DISPLAY_CAP` (default 10) most recent entries.
- **Main Area — Chat Interface:** A scrollable chat history using `st.chat_message`. User messages and assistant responses are displayed in alternating bubbles. Responses stream token-by-token using `st.write_stream`. Each answer ends with a **Sources** block listing cited papers with title, authors, and page number. The chat input is pinned to the bottom using `st.chat_input`. Chat commands like "Show my notes" or "Show my idea log" display the most recent entries inline in the chat (capped at `DISPLAY_CAP`).

**Key Streamlit Components:**
- `st.sidebar` for all controls
- `st.chat_input` / `st.chat_message` for the conversational interface
- `st.write_stream` for streaming LLM responses
- `st.progress` for ingestion progress
- `st.expander` for the ingested papers list and memory viewer
- `st.selectbox` / `st.slider` for settings

---

### Page 2: Multi-Paper Synthesis — `src/ui/synthesis_view.py` (Phase 4)

```
┌──────────────────────────────────────────────────────────────────┐
│  📚 ResearchPaper-reading-Assist Agent  >  Synthesis          [← Back]  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌── Sidebar ──────────────────┐  ┌── Main Area ──────────────┐ │
│  │                             │  │                            │ │
│  │  🔍 Filters                 │  │  Synthesis Query           │ │
│  │                             │  │  ┌──────────────────────┐  │ │
│  │  Year Range:                │  │  │ Compare approaches to│  │ │
│  │  [2020] ——●—— [2024]       │  │  │ attention mechanisms  │  │ │
│  │                             │  │  │ across all papers     │  │ │
│  │  Authors:                   │  │  └──────────────────────┘  │ │
│  │  ┌────────────────────────┐ │  │  Strategy: (●) MapReduce  │ │
│  │  │ ☑ Vaswani et al.      │ │  │            ( ) Refine     │ │
│  │  │ ☑ Devlin et al.       │ │  │                            │ │
│  │  │ ☐ Brown et al.        │ │  │  [🔬 Synthesize]           │ │
│  │  │ ☐ ...                 │ │  │                            │ │
│  │  └────────────────────────┘ │  │  ━━━━━━━━━━━━━━━━━━━━━━━  │ │
│  │                             │  │                            │ │
│  │  Topics:                    │  │  📊 Synthesis Result       │ │
│  │  ┌────────────────────────┐ │  │                            │ │
│  │  │ [NLP ×] [Vision ×]    │ │  │  **Common Themes:**        │ │
│  │  │ [+ Add topic]         │ │  │  • Self-attention is used  │ │
│  │  └────────────────────────┘ │  │    across all papers as... │ │
│  │                             │  │  • Pre-training on large   │ │
│  │  Papers Included: 4 / 12   │  │    corpora is a shared...  │ │
│  │                             │  │                            │ │
│  │                             │  │  **Key Differences:**      │ │
│  │                             │  │  • Paper A uses encoder-   │ │
│  │                             │  │    only while Paper B...   │ │
│  │                             │  │                            │ │
│  │                             │  │  **Contradictions:**       │ │
│  │                             │  │  • Vaswani et al. claim... │ │
│  │                             │  │    but Devlin et al. show..│ │
│  │                             │  │                            │ │
│  │                             │  │  **Sources:**              │ │
│  │                             │  │  [1] Attention Is All...   │ │
│  │                             │  │  [2] BERT: Pre-training... │ │
│  │                             │  │  [3] GPT-4 Technical...    │ │
│  │                             │  │  [4] LLaMA: Open and...    │ │
│  └─────────────────────────────┘  └────────────────────────────┘ │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Layout Details:**
- **Sidebar — Filters:** A year-range slider (`st.slider`), a multi-select checklist for authors (`st.multiselect`), and a tag-style topic selector. A live counter shows how many papers match the current filters vs. total ingested.
- **Main Area — Query Input:** A `st.text_area` for the synthesis prompt. Radio buttons to choose between MapReduce or Refine chain strategy. A "Synthesize" button triggers the chain.
- **Main Area — Results:** The synthesis output is rendered in structured markdown sections: Common Themes, Key Differences, Contradictions, and a numbered Sources list. Each claim links back to its source paper. A `st.spinner` shows during processing.

**Key Streamlit Components:**
- `st.slider` for year range filtering
- `st.multiselect` for author/topic filtering
- `st.text_area` for the synthesis query
- `st.radio` for chain strategy selection
- `st.spinner` during synthesis processing
- `st.markdown` for structured result rendering

---

### Page 3: Research Agent Dashboard — `src/ui/agent_dashboard.py` (Phase 5)

```
┌──────────────────────────────────────────────────────────────────┐
│  📚 ResearchPaper-reading-Assist Agent  >  Agent              [← Back]  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌── Main Area (full width) ─────────────────────────────────┐  │
│  │                                                            │  │
│  │  🤖 Research Agent                                         │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │ Find the 3 most cited papers on diffusion models     │  │  │
│  │  │ from 2023 and compare their methods                  │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │  Max iterations: [10    ]    [🚀 Run Agent]   [⏹ Stop]    │  │
│  │                                                            │  │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │  │
│  │                                                            │  │
│  │  ┌─ Reasoning Trace ────────────────────────────────────┐  │  │
│  │  │                                                      │  │  │
│  │  │  Step 1                                    ✅ Done   │  │  │
│  │  │  ├─ 💭 Thought: I need to search arXiv for          │  │  │
│  │  │  │   diffusion model papers from 2023...             │  │  │
│  │  │  ├─ 🔧 Tool: ArxivQueryRun                          │  │  │
│  │  │  │   Input: "diffusion models 2023"                  │  │  │
│  │  │  └─ 📋 Result: Found 15 papers. Top results:        │  │  │
│  │  │      1. "Denoising Diffusion..." (cited 1240x)       │  │  │
│  │  │      2. "Stable Diffusion..." (cited 890x)           │  │  │
│  │  │      3. ...                                          │  │  │
│  │  │                                                      │  │  │
│  │  │  Step 2                                    ✅ Done   │  │  │
│  │  │  ├─ 💭 Thought: Now I'll retrieve details from      │  │  │
│  │  │  │   our local store for any matching papers...      │  │  │
│  │  │  ├─ 🔧 Tool: PaperRetriever                         │  │  │
│  │  │  │   Input: "diffusion models denoising"             │  │  │
│  │  │  └─ 📋 Result: Found 2 matching local papers...     │  │  │
│  │  │                                                      │  │  │
│  │  │  Step 3                                    🔄 Running│  │  │
│  │  │  ├─ 💭 Thought: Comparing the methods across        │  │  │
│  │  │  │   the top 3 papers...                             │  │  │
│  │  │  └─ ▌ (generating...)                                │  │  │
│  │  │                                                      │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │                                                            │  │
│  │  ┌─ Final Answer ───────────────────────────────────────┐  │  │
│  │  │                                                      │  │  │
│  │  │  (appears here when agent completes)                 │  │  │
│  │  │                                                      │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │                                                            │  │
│  │  Progress: ████████░░  Step 3 / 10                        │  │
│  │                                                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Layout Details:**
- **Task Input:** A `st.text_area` for the research task description. A number input for max iterations. "Run Agent" button starts the agent; "Stop" button halts it early.
- **Reasoning Trace Panel:** A live-updating container (using `st.container` + `st.empty`) that renders each agent step as it completes. Each step shows: the agent's thought/reasoning, the tool it chose, the tool input, and the tool output. Steps are color-coded by status: done (green check), running (blue spinner), pending (grey).
- **Final Answer Section:** Appears below the trace once the agent finishes. Contains the synthesized answer with full citations.
- **Progress Bar:** A `st.progress` bar at the bottom tracking current step vs. max iterations.

**Key Streamlit Components:**
- `st.text_area` for task input
- `st.number_input` for max iterations
- `st.container` / `st.empty` for live-updating trace
- `st.status` for collapsible step-by-step trace items
- `st.progress` for iteration tracking
- `st.button` for Run/Stop controls
- `st.markdown` for the final answer

---

### Navigation Between Pages

Use Streamlit's built-in multi-page app structure (`st.navigation` or a `pages/` directory):

```
src/ui/
├── streamlit_app.py          # Main entry: Q&A chat (default page)
├── pages/
│   ├── 1_Synthesis.py        # Multi-paper synthesis page
│   └── 2_Agent.py            # Research agent dashboard
```

The sidebar automatically shows page links. Each page shares the same Chroma vector store and memory backend via `st.session_state` or a shared config module.

---

## Folder Structure

```
cli.py
src/
├── core/
│   ├── embeddings.py
│   ├── llm.py
│   ├── vectorstore.py
│   └── retriever.py
├── ingestion/
│   ├── __init__.py
│   ├── scanner.py
│   ├── loader.py
│   ├── splitter.py
│   └── pipeline.py
├── qna/
│   ├── qa.py
│   └── prompts.py
├── memory/
│   ├── memory.py
│   ├── idea_log.py
│   ├── notes_log.py
│   └── intent_detector.py
├── synthesis/
│   ├── synthesizer.py
│   └── filters.py
├── agent/
│   ├── agent.py
│   └── tools.py
├── ui/
│   ├── streamlit_app.py
│   ├── synthesis_view.py
│   └── agent_dashboard.py
└── api/
    ├── memory_api.py
    ├── idea_log_api.py
    └── notes_api.py

tests/
├── test_scanner.py
├── test_ingestion_loader.py
├── test_splitter.py
├── test_vectorstore_integration.py
├── test_pipeline.py
├── test_llm_factory.py
├── test_retriever.py
├── test_retrievalqa_citations.py
├── test_streaming_callback.py
├── test_memory_persistence.py
├── test_memory_clear.py
├── test_idea_log.py
├── test_idea_log_api.py
├── test_notes_log.py
├── test_notes_api.py
├── test_intent_detector.py
├── test_mapreduce_chain.py
├── test_selfquery_filters.py
├── test_agent_tooling.py
├── test_agent_trace_logging.py
├── test_retriever_hybrid.py
├── test_qa_prompt_enforcement.py
├── test_multiquery_retriever.py
├── test_reranker.py
├── test_qa_confidence_scoring.py
├── test_paper_structure_extraction.py
├── test_selfquery_metadata_filtering.py
└── test_deduplication.py

docs/
├── ingestion.md
├── qna.md
├── memory.md
├── synthesis.md
└── agent.md

examples/
├── quickstart_streamlit.md
└── sample_papers/

requirements.txt
README.md
```

---

## Suggested Dependencies (requirements.txt)

```
langchain>=0.1.0
langchain-google-genai>=1.0.0
langchain-ollama>=0.1.0
langchain-community>=0.1.0
chromadb>=0.3.0
google-generativeai>=0.3.0
sentence-transformers>=2.2.0
transformers>=4.30.0
rank-bm25>=0.2.2
cross-encoder>=2.0.0
streamlit>=1.20.0
pytest>=7.1.0
pytest-mock>=3.10.0
pytest-asyncio>=0.22.0
fastapi>=0.95.0
uvicorn>=0.22.0
pydantic>=1.10.0
python-dotenv>=1.0.0
tqdm>=4.65.0
pdfplumber>=0.7.0
arxiv>=1.4.0
faiss-cpu>=1.7.2
typing-extensions>=4.5.0
```
