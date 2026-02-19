# Product Requirements Document (PRD)

## ResearchPaper-reading-Assist Agent for Academic Papers

**Version:** 1.0  
**Date:** February 16, 2026  
**Product Owner:** Srimugunthan  
**Status:** Draft - MVP Scope

---

## 1. Executive Summary

### 1.1 Product Vision
Build an intelligent ResearchPaper-reading-Assist Agent that helps researchers and students manage, understand, and extract insights from academic papers using LangChain. The assistant ingests papers, answers questions with citations, remembers research context, and synthesizes information across multiple papers.

### 1.2 Success Metrics
- **Query Success Rate**: 90% of questions answered satisfactorily
- **Response Time**: <5 seconds for typical queries
- **Paper Processing**: Handle 20+ papers per user
- **User Satisfaction**: 85%+ positive feedback

---

## 2. Problem Statement

### 2.1 User Pain Points
1. **Information Overload**: Too many papers to read thoroughly
2. **Context Loss**: Difficult to remember details across papers read weeks apart
3. **Manual Synthesis**: Connecting insights requires extensive note-taking
4. **Citation Tracking**: Hard to remember which paper contained specific information
5. **Time Constraints**: Reading dense academic papers is slow

### 2.2 Target Users

**Primary Persona: Graduate Student**
- Conducting literature reviews for thesis
- Needs to understand 50-100 papers in their domain
- Limited time, needs efficient research tools

**Secondary Persona: Research Scientist**
- Staying current with developments in ML/AI
- Needs quick answers about methodologies
- Values accuracy and proper citations

---

## 3. Core Features (MVP Only)

### Feature 1: Paper Ingestion & Indexing

**Description:** Upload PDF papers or provide arXiv URLs for automatic processing and indexing.

**User Stories:**
- As a researcher, I want to upload multiple PDFs so I can build my research library quickly
- As a student, I want to add papers from arXiv by URL without manual downloads
- As a user, I want to see processing status so I know when papers are ready

**Acceptance Criteria:**
- Support PDF upload (drag-and-drop)
- Support arXiv URL input (auto-fetch)
- Extract text and metadata (title, authors, year, abstract)
- Index full text in vector database
- Process single paper in <30 seconds
- Show processing progress

**LangChain Components:**
- **DocumentLoaders**: PyPDFLoader, ArxivLoader
- **Text Splitters**: RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
- **Embeddings**: OpenAI embeddings or sentence-transformers
- **Vector Store**: Chroma with metadata
- **Indexers**: VectorStoreIndexCreator
- **Callbacks**: Custom callback for progress tracking

---

### Feature 2: Intelligent Q&A with Citations

**Description:** Ask natural language questions and receive accurate answers with source citations.

**User Stories:**
- As a researcher, I want to ask "What datasets were used?" and get answers from relevant papers
- As a student, I want citations included so I can verify and reference claims
- As a user, I want to ask follow-up questions that build on previous context

**Acceptance Criteria:**
- Natural language question input
- Answers synthesized from relevant paper sections
- Citations include paper title, authors, and page/section reference
- Maintain conversational context for follow-up questions
- Return "I don't know" when information not available in papers
- Response time <5 seconds
- Stream responses for longer answers

**LangChain Components:**
- **Retrievers**: MultiQueryRetriever (generates query variations), ContextualCompressionRetriever
- **Chains**: RetrievalQA chain with source documents
- **Chat Models**: ChatOpenAI or ChatAnthropic (Claude)
- **Prompt Templates**: Custom templates with citation instructions
- **Memory**: ConversationBufferMemory for context
- **Callbacks**: StreamingStdOutCallbackHandler

---

### Feature 3: Research Session Memory

**Description:** Remember past conversations, research interests, and frequently accessed papers across sessions.

**User Stories:**
- As a researcher, I want the system to remember my research focus area
- As a user, I want to reference previous discussions like "that attention paper we discussed"
- As a student, I want the system to understand my evolving research interests

**Acceptance Criteria:**
- Persist conversation history across sessions
- Track research interests (topics, methods, authors mentioned)
- Remember frequently queried papers
- Support queries about past conversations
- Allow users to clear memory (privacy control)

**LangChain Components:**
- **Memory**: VectorStoreRetrieverMemory for long-term context, ConversationSummaryMemory for long sessions, EntityMemory for tracking papers/authors/concepts
- **Callbacks**: Custom callback to log and extract research interests
- **Prompt Templates**: System prompts that leverage memory context

---

### Feature 4: Multi-Paper Synthesis

**Description:** Synthesize insights across multiple papers to answer complex research questions.

**User Stories:**
- As a researcher, I want to compare approaches across papers to understand the field
- As a student, I want a summary of "transformer architectures" from all uploaded papers
- As a user, I want to identify common themes and differences

**Acceptance Criteria:**
- Query across entire paper collection
- Identify common themes and differences
- Generate comparative summaries
- Highlight contradictions when present
- Support queries like "Compare X across all papers"

**LangChain Components:**
- **Chains**: MapReduceDocumentsChain for multi-document summarization, RefineDocumentsChain for iterative synthesis
- **Retrievers**: SelfQueryRetriever with metadata filters (year, author, topic)
- **Prompt Templates**: Few-shot prompts for comparison tasks
- **Chat Models**: Claude/GPT-4 for synthesis reasoning

---

### Feature 5: Smart Research Agent

**Description:** Autonomous agent that plans and executes multi-step research tasks.

**User Stories:**
- As a researcher, I want to ask "What are recent approaches to few-shot learning?" and have the agent search arXiv and summarize findings
- As a user, I want to see the agent's reasoning process
- As a student, I want the agent to explain complex concepts from papers

**Acceptance Criteria:**
- Plan multi-step research tasks autonomously
- Access tools: arXiv search, web search, Python REPL (for math), paper retrieval
- Show reasoning traces (what tool, why, what found)
- Handle failures gracefully
- Ask clarifying questions when needed

**LangChain Components:**
- **Agents**: ReAct agent (create_react_agent)
- **Tools**: ArxivQueryRun, DuckDuckGoSearchRun, PythonREPL, custom PaperRetrieverTool
- **Chains**: AgentExecutor with max_iterations limit
- **Callbacks**: AgentCallbackHandler to log reasoning steps
- **Prompt Templates**: System message defining agent persona and capabilities

---

## 4. Technical Architecture

### 4.1 System Architecture
```
User Interface (Streamlit/Gradio)
         ↓
    Agent Layer (ReAct Agent + Tools)
         ↓
    Chain Orchestration (Retrieval, MapReduce, Refine)
         ↓
    Memory Layer (Conversation, Vector, Entity)
         ↓
    Vector Store (Chroma/FAISS) + Retrievers
         ↓
    Document Processing (Loaders, Splitters)
         ↓
    LLM Layer (Claude/GPT-4 + Embeddings)
```

### 4.2 Key Workflows

**Paper Ingestion:**
```
PDF/URL → DocumentLoader → TextSplitter → 
Embeddings → VectorStore → Metadata Storage
```

**Query Processing:**
```
User Query → Agent Planning → MultiQueryRetriever → 
Vector Search → Compression → LLM Synthesis → 
Response + Citations → Memory Update
```

### 4.3 Technology Stack

**Core:**
- LangChain 0.1.x
- Python 3.10+

**LLMs (configurable via factory):**
- Gemini (cloud, default) — `ChatGoogleGenerativeAI`
- Ollama (local/offline) — `ChatOllama` (e.g., llama3, mistral)
- Embeddings: sentence-transformers (local) or Gemini embeddings

**Storage:**
- Chroma (vector database)
- PostgreSQL (metadata, sessions)
- Local filesystem (PDFs)

**Frontend:**
- Streamlit (MVP UI)

**Monitoring:**
- LangSmith for tracing
- Custom callbacks for metrics

---
## Technical architecture
## 4.1 System Components
```
┌─────────────────────────────────────────────────────┐
│                  User Interface                      │
│            (Streamlit/Gradio/CLI)                   │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│              LangChain Agent Layer                   │
│  ┌──────────────┐  ┌──────────────┐                │
│  │ ReAct Agent  │  │  Tool Router │                │
│  └──────────────┘  └──────────────┘                │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│                Chain Orchestration                   │
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────┐ │
│  │ Retrieval   │ │ MapReduce    │ │   Refine    │ │
│  │   Chain     │ │   Chain      │ │   Chain     │ │
│  └─────────────┘ └──────────────┘ └─────────────┘ │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│              Memory & Context                        │
│  ┌──────────────────┐  ┌────────────────────────┐  │
│  │ Conversation     │  │ VectorStore Retriever  │  │
│  │ Memory           │  │ Memory                 │  │
│  └──────────────────┘  └────────────────────────┘  │
│  ┌──────────────────┐  ┌────────────────────────┐  │
│  │ Entity Memory    │  │ Session Persistence    │  │
│  └──────────────────┘  └────────────────────────┘  │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│            Retrieval & Indexing Layer                │
│  ┌──────────────────────────────────────────────┐  │
│  │         Vector Database (Chroma/FAISS)       │  │
│  │  - Paper embeddings                          │  │
│  │  - Metadata (title, author, year, domain)    │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │  Retrievers:                                 │  │
│  │  - MultiQueryRetriever                       │  │
│  │  - ContextualCompressionRetriever            │  │
│  │  - SelfQueryRetriever                        │  │
│  │  - ParentDocumentRetriever                   │  │
│  └──────────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│              Data Ingestion Layer                    │
│  ┌──────────────────┐  ┌────────────────────────┐  │
│  │ Document Loaders │  │  Text Splitters        │  │
│  │ - PyPDFLoader    │  │  - Recursive           │  │
│  │ - ArxivLoader    │  │  - CharacterBased      │  │
│  │ - WebBaseLoader  │  │                        │  │
│  └──────────────────┘  └────────────────────────┘  │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│                 LLM Layer                            │
│  - Gemini (cloud, default)                           │
│  - Ollama (local/offline alternative)                │
│  - Embeddings: sentence-transformers / Gemini        │
└─────────────────────────────────────────────────────┘
```

### Layer Descriptions

**1. User Interface Layer**
- Streamlit for rapid prototyping
- Gradio as alternative UI framework
- CLI for programmatic access

**2. LangChain Agent Layer**
- ReAct Agent for autonomous reasoning and task planning
- Tool Router for intelligent tool selection

**3. Chain Orchestration Layer**
- Retrieval Chain for Q&A with citations
- MapReduce Chain for multi-document summarization
- Refine Chain for iterative synthesis

**4. Memory & Context Layer**
- Conversation Memory for maintaining dialogue context
- VectorStore Retriever Memory for long-term research context
- Entity Memory for tracking papers, authors, concepts
- Session Persistence for cross-session continuity

**5. Retrieval & Indexing Layer**
- Vector Database (Chroma/FAISS) storing paper embeddings and metadata
- Multiple retriever strategies for optimal information retrieval
- Metadata filtering by title, author, year, domain

**6. Data Ingestion Layer**
- Document Loaders for PDF and arXiv sources
- Text Splitters for chunking documents efficiently
- Support for web-based sources

**7. LLM Layer**
- Gemini as default cloud reasoning engine
- Ollama as local/offline alternative (e.g., llama3, mistral)
- sentence-transformers / Gemini for embeddings
- Configurable via `src/core/llm.py` factory

## 5. Implementation Roadmap

### 1: Foundation
- Set up vector database (Chroma)
- Implement paper ingestion (Feature 1)
- Basic document loading and indexing
- Simple retrieval testing

**Deliverable:** Can upload and index papers

### 2: Core Q&A
- Implement Q&A with citations (Feature 2)
- Add conversation memory (Feature 3)
- Build basic Streamlit UI
- Test with 20+ papers

**Deliverable:** Working Q&A system with memory

### 3: Advanced Capabilities
- Multi-paper synthesis (Feature 4)
- Agent implementation (Feature 5)
- Improve retrieval quality
- Add streaming responses

**Deliverable:** Full MVP with all 5 features

### 4: Polish & Testing
- Bug fixes and edge cases
- Performance optimization
- User testing with 5-10 researchers
- Documentation

**Deliverable:** Production-ready MVP

---

## 6. LangChain Components Mapping

| Feature | Agents | Chains | Callbacks | Memory | Retrievers | ChatModels | PromptTemplates |
|---------|--------|--------|-----------|--------|------------|------------|-----------------|
| **F1: Ingestion** | - | Document chains | Progress tracking | - | - | - | - |
| **F2: Q&A** | - | RetrievalQA | Streaming | ConversationBuffer | MultiQuery, Compression | ✓ | Citation prompts |
| **F3: Memory** | - | - | Interest logging | VectorMemory, Entity, Summary | VectorRetriever | - | Context-aware |
| **F4: Synthesis** | - | MapReduce, Refine | - | Summary | SelfQuery | ✓ | Few-shot |
| **F5: Agent** | ReAct | AgentExecutor | Reasoning traces | All types | Custom tools | ✓ | System prompts |

---

## 7. Success Criteria

### MVP Launch Checklist
- [ ] Successfully process 20+ papers
- [ ] Answer questions with 90%+ accuracy
- [ ] Provide correct citations for all answers
- [ ] Maintain conversation context across sessions
- [ ] Agent completes multi-step tasks successfully
- [ ] Response time <5 seconds (p95)
- [ ] 5 beta users test successfully

### Key Metrics (First Month)
- Papers indexed per user: 20+
- Queries per session: 8+
- User satisfaction: 85%+
- Citation accuracy: 95%+
- System uptime: 99%+

---

## 8. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM hallucinations | High | Always ground in sources, verify citations |
| Slow vector search | Medium | Optimize chunk size, use compression retriever |
| Memory context overflow | Medium | Implement summarization, limit context window |
| API costs | Medium | Cache responses, optimize prompts |

---

## 9. Sample User Queries

**Simple Queries:**
- "What is the main contribution of this paper?"
- "List all datasets mentioned in my papers"
- "Who are the authors of the attention paper?"

**Complex Queries:**
- "Compare attention mechanisms in vision vs. language transformers"
- "What are common approaches to noise-robust learning in my collection?"
- "Summarize recent developments in few-shot learning"

**Agent-Driven:**
- "Find recent arXiv papers on graph neural networks and summarize them"
- "Explain the AGCE loss function from that paper we discussed"
- "What mathematical techniques are used across my RL papers?"

---

## 10. Out of Scope (Post-MVP)

The following features are explicitly excluded from MVP:
- Literature review generation
- Collaborative features / team sharing
- Citation graph visualization
- Export to LaTeX/BibTeX
- Mobile apps
- Integration with reference managers (Zotero, Mendeley)
- Automatic paper recommendations
- Email alerts for new papers

---

## Appendix: Quick Reference

### Essential LangChain Imports
```python
# Document Processing
from langchain.document_loaders import PyPDFLoader, ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Retrievers
from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever

# Chains
from langchain.chains import RetrievalQA, MapReduceDocumentsChain, RefineDocumentsChain
from langchain.chains import LLMChain

# Memory
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.memory import VectorStoreRetrieverMemory, EntityMemory

# Agents
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import ArxivQueryRun, DuckDuckGoSearchRun, PythonREPL

# Chat Models
from langchain.chat_models import ChatOpenAI, ChatAnthropic

# Prompts
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate

# Callbacks
from langchain.callbacks import StreamingStdOutCallbackHandler
```

---

**End of Document**

*This PRD focuses exclusively on MVP features demonstrating all core LangChain capabilities: Agents, Chains, Callbacks, Memory, Retrievers, Chat Models, and Prompt Templates.*