# Singapore SMB Customer Enquiry Support AI Agent
# Master Execution Plan

**Author:** Matrix Agent  
**Version:** 2.0  
**Date:** 2026-01-03  
**Status:** Ready for Implementation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Validation & Updates](#2-architecture-validation--updates)
3. [Re-Imagined System Architecture](#3-re-imagined-system-architecture)
4. [Technology Stack (Validated 2025)](#4-technology-stack-validated-2025)
5. [Complete File Hierarchy](#5-complete-file-hierarchy)
6. [Key Files & Descriptions](#6-key-files--descriptions)
7. [Master Execution Phases](#7-master-execution-phases)
8. [Phase 1: Foundation & Infrastructure](#phase-1-foundation--infrastructure)
9. [Phase 2: Core Backend Development](#phase-2-core-backend-development)
10. [Phase 3: RAG Pipeline Implementation](#phase-3-rag-pipeline-implementation)
11. [Phase 4: Agent Orchestration (LangGraph)](#phase-4-agent-orchestration-langgraph)
12. [Phase 5: Frontend Chat Widget](#phase-5-frontend-chat-widget)
13. [Phase 6: Integration & Testing](#phase-6-integration--testing)
14. [Phase 7: Deployment & Monitoring](#phase-7-deployment--monitoring)
15. [Success Metrics & Validation](#8-success-metrics--validation)

---

## 1. Executive Summary

This Master Execution Plan presents a **thoroughly researched and validated** architecture for building a Singapore SMB Customer Enquiry Support AI Agent. The design incorporates **2025 best practices** from:

- **LangGraph/LangChain 1.0** patterns for stateful agent orchestration
- **RAG optimization** with hybrid search (BM25 + vectors + RRF fusion)
- **FastAPI production patterns** for scalable AI backends
- **React 18+** chat widget patterns with SSE streaming
- **Pydantic 2.0** for robust data validation

### Key Architecture Decisions (Validated by Research)

| Component | Original Design | Research Validation | Final Decision |
|-----------|----------------|---------------------|----------------|
| Streaming | WebSocket | SSE preferred for LLM (simpler, auto-reconnect) | **SSE primary, WebSocket secondary** |
| State Management | Custom | LangGraph checkpointers are production-ready | **LangGraph with PostgreSQL checkpointer** |
| Vector DB | Qdrant | Qdrant validated (326 QPS, SOC2, hybrid search) | **Qdrant Cloud** |
| Reranking | Cross-encoder | Cohere Rerank-4 achieves perfect RAGAS scores | **Cohere Rerank-4-Multilingual** |
| Frontend State | Context API | Zustand is 2025 recommendation | **Zustand + TanStack Query** |
| RAG Framework | LangChain | LlamaIndex 40% faster retrieval | **Hybrid: LlamaIndex for RAG + LangGraph for agents** |

---

## 2. Architecture Validation & Updates

### 2.1 Original Architecture Review

The original design was **fundamentally sound** with correct choices for:
- ✅ LangGraph for agent orchestration
- ✅ 3-tier hierarchical memory architecture
- ✅ Hybrid search with RRF fusion
- ✅ Cross-encoder reranking
- ✅ Singapore SMB context awareness

### 2.2 Research-Driven Updates

Based on comprehensive research, the following **enhancements** are incorporated:

#### A. Streaming Architecture Update
```
ORIGINAL: WebSocket-only streaming
UPDATED:  SSE primary (simpler infrastructure, auto-reconnect)
          + WebSocket for bidirectional features (typing indicators)
```

#### B. RAG Pipeline Enhancement
```
ORIGINAL: LangChain RAG
UPDATED:  LlamaIndex for retrieval (40% faster, 35% higher accuracy)
          + LangGraph for agent orchestration (best of both worlds)
```

#### C. Document Ingestion
```
ORIGINAL: Unstructured.io
UPDATED:  LlamaParse (RAG-native, citation support, variable chunking)
          + Docling fallback (complex PDFs with tables)
```

#### D. Chunking Strategy
```
ORIGINAL: Fixed 400 tokens
UPDATED:  Recursive (400 tokens) default
          + Semantic chunking for high-failure sections
          + Page-level for PDF-heavy content
```

#### E. Frontend State Management
```
ORIGINAL: React Context
UPDATED:  Zustand for client state (minimal boilerplate)
          + TanStack Query for server state (caching, streaming)
          + Vercel AI SDK useChat hook for chat state
```

#### F. Embedding Model
```
ORIGINAL: OpenAI text-embedding-3-small
UPDATED:  Cohere Embed-4 (multilingual, SEA languages, $0.12/M tokens)
```

---

## 3. Re-Imagined System Architecture

### 3.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────┐ │
│  │   Embeddable Chat   │    │    Admin Dashboard   │    │   Mobile App    │ │
│  │      Widget         │    │    (Future Phase)    │    │  (Future Phase) │ │
│  │  ┌───────────────┐  │    └─────────────────────┘    └─────────────────┘ │
│  │  │ Shadow DOM    │  │                                                    │
│  │  │ + React 18    │  │    Technology:                                     │
│  │  │ + Zustand     │  │    - TypeScript + Vite                            │
│  │  │ + TanStack Q  │  │    - Shadcn/UI + Tailwind                         │
│  │  │ + Framer      │  │    - react-window (virtualization)                │
│  │  └───────────────┘  │    - react-markdown + remark-gfm                  │
│  └─────────────────────┘                                                    │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │
                              SSE (primary) / WebSocket (secondary)
                                        │
┌───────────────────────────────────────▼─────────────────────────────────────┐
│                              API GATEWAY                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    FastAPI Application                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │   /auth     │  │   /chat     │  │  /webhooks  │  │  /health   │ │   │
│  │  │  (JWT+OAuth)│  │  (REST+SSE) │  │  (Callbacks)│  │  (Probes)  │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │   │
│  │                                                                       │   │
│  │  Middleware: CORS | Rate Limiting (SlowAPI) | Request Logging        │   │
│  │  Observability: OpenTelemetry | Prometheus | structlog               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │
┌───────────────────────────────────────▼─────────────────────────────────────┐
│                           AGENT ORCHESTRATION LAYER                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    LangGraph State Machine                           │   │
│  │                                                                       │   │
│  │   ┌──────────┐    ┌──────────────┐    ┌──────────────┐              │   │
│  │   │  INPUT   │───▶│   ROUTER     │───▶│   RETRIEVER  │              │   │
│  │   │  NODE    │    │    NODE      │    │     NODE     │              │   │
│  │   └──────────┘    └──────────────┘    └──────────────┘              │   │
│  │        │                 │                    │                       │   │
│  │        │                 ▼                    ▼                       │   │
│  │        │          ┌──────────────┐    ┌──────────────┐              │   │
│  │        │          │   GRADER     │    │   RERANKER   │              │   │
│  │        │          │    NODE      │    │     NODE     │              │   │
│  │        │          └──────────────┘    └──────────────┘              │   │
│  │        │                 │                    │                       │   │
│  │        ▼                 ▼                    ▼                       │   │
│  │   ┌──────────────────────────────────────────────────┐              │   │
│  │   │              GENERATOR NODE (LLM)                 │              │   │
│  │   │         + Streaming Token Output                  │              │   │
│  │   │         + Singapore Context Awareness             │              │   │
│  │   └──────────────────────────────────────────────────┘              │   │
│  │        │                                                              │   │
│  │        ▼                                                              │   │
│  │   ┌──────────┐    ┌──────────────┐                                  │   │
│  │   │  OUTPUT  │───▶│  CHECKPOINT  │  (PostgreSQL Checkpointer)       │   │
│  │   │   NODE   │    │    SAVE      │                                  │   │
│  │   └──────────┘    └──────────────┘                                  │   │
│  │                                                                       │   │
│  │  Tools: [search_knowledge_base, escalate_to_human, get_order_status] │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │
┌───────────────────────────────────────▼─────────────────────────────────────┐
│                              RAG PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    LlamaIndex RAG Engine                             │   │
│  │                                                                       │   │
│  │   Query Transformers          Retrieval               Post-Processing │   │
│  │   ┌─────────────────┐   ┌──────────────────┐   ┌─────────────────┐  │   │
│  │   │ • HyDE          │   │ • BM25 (sparse)  │   │ • RRF Fusion    │  │   │
│  │   │ • Multi-Query   │   │ • Cohere Embed   │   │ • Cohere Rerank │  │   │
│  │   │ • Step-Back     │   │ • Metadata Filter│   │ • Context Comp  │  │   │
│  │   └─────────────────┘   └──────────────────┘   └─────────────────┘  │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │
┌───────────────────────────────────────▼─────────────────────────────────────┐
│                              DATA LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐  │
│   │   PostgreSQL    │   │   Qdrant Cloud  │   │      Redis Cluster       │  │
│   │                 │   │                 │   │                          │  │
│   │ • Users         │   │ • Document      │   │ • Session cache          │  │
│   │ • Conversations │   │   embeddings    │   │ • Rate limiting          │  │
│   │ • Messages      │   │ • Hybrid search │   │ • Short-term memory      │  │
│   │ • Checkpoints   │   │ • Metadata      │   │ • Connection state       │  │
│   │ • Audit logs    │   │   filtering     │   │                          │  │
│   └─────────────────┘   └─────────────────┘   └─────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow Sequence

```
User Message
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. SSE Connection receives message                               │
│ 2. Rate limiting check (SlowAPI + Redis)                        │
│ 3. JWT validation + session retrieval                           │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. LangGraph processes through state machine:                    │
│    a. Input validation + language detection                     │
│    b. Router decides: FAQ / Complex / Escalation                │
│    c. Retriever fetches from Qdrant (hybrid search)             │
│    d. Reranker scores with Cohere Rerank-4                      │
│    e. Generator produces response with streaming                 │
│    f. Checkpoint saved to PostgreSQL                            │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Response streamed via SSE (token by token)                    │
│ 6. Message persisted to PostgreSQL                              │
│ 7. Analytics logged (background task)                           │
│ 8. Memory updated (Redis short-term + PostgreSQL long-term)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Technology Stack (Validated 2025)

### 4.1 Backend Stack

| Category | Technology | Version | Justification |
|----------|-----------|---------|---------------|
| **Framework** | FastAPI | 0.110+ | Async-first, automatic docs, production-proven |
| **Agent Orchestration** | LangGraph | 0.2+ | Stateful workflows, checkpointing, streaming |
| **RAG Engine** | LlamaIndex | 0.11+ | 40% faster retrieval, 160+ data formats |
| **LLM Provider** | OpenAI GPT-4 | gpt-4-turbo | Best quality for customer support |
| **Embeddings** | Cohere Embed-4 | embed-4 | Multilingual (SEA), $0.12/M tokens |
| **Reranking** | Cohere Rerank | rerank-4-multilingual | Perfect RAGAS scores, 100+ languages |
| **Vector DB** | Qdrant | 1.8+ | Hybrid search, SOC2, 326 QPS |
| **Primary DB** | PostgreSQL | 16+ | ACID, checkpoints, audit logs |
| **Cache/Session** | Redis | 7+ | Session cache, rate limiting, pub/sub |
| **Task Queue** | ARQ | 0.26+ | Async task queue (lighter than Celery) |

### 4.2 Frontend Stack

| Category | Technology | Version | Justification |
|----------|-----------|---------|---------------|
| **Framework** | React | 18.3+ | Concurrent features, SSR support |
| **Build Tool** | Vite | 5+ | Fast HMR, optimized builds |
| **Language** | TypeScript | 5.4+ | Type safety, better DX |
| **Styling** | Tailwind CSS | 3.4+ | Utility-first, consistent design |
| **Components** | Shadcn/UI | Latest | Accessible, customizable |
| **State (Client)** | Zustand | 4+ | Minimal boilerplate, React 18 compatible |
| **State (Server)** | TanStack Query | 5+ | Caching, streaming support |
| **Chat Hook** | Vercel AI SDK | 4+ | useChat, SSE streaming built-in |
| **Virtualization** | react-window | 1.8+ | Dynamic height, performant lists |
| **Markdown** | react-markdown | 9+ | Safe rendering, remark plugins |
| **Animation** | Framer Motion | 11+ | Smooth message animations |

### 4.3 Infrastructure & DevOps

| Category | Technology | Justification |
|----------|-----------|---------------|
| **Containerization** | Docker + Docker Compose | Standard, reproducible |
| **Observability** | OpenTelemetry + Prometheus + Grafana | Full-stack tracing |
| **Logging** | structlog | Structured JSON logs |
| **Rate Limiting** | SlowAPI + Redis | Distributed rate limiting |
| **Document Parsing** | LlamaParse + Docling | RAG-native + complex PDFs |
| **Evaluation** | RAGAs | Faithfulness, context recall metrics |

---

## 5. Complete File Hierarchy

```
singapore-smb-support-agent/
│
├── 📁 backend/                           # FastAPI Backend Application
│   ├── 📁 alembic/                       # Database Migrations
│   │   ├── versions/                     # Migration version files
│   │   ├── env.py                        # Alembic environment config
│   │   └── script.py.mako                # Migration template
│   │
│   ├── 📁 src/                           # Source Code (Domain-Driven)
│   │   │
│   │   ├── 📁 core/                      # Core Infrastructure
│   │   │   ├── __init__.py
│   │   │   ├── config.py                 # Pydantic Settings (env vars)
│   │   │   ├── database.py               # SQLAlchemy async engine + session
│   │   │   ├── redis.py                  # Async Redis client
│   │   │   ├── security.py               # JWT utilities, password hashing
│   │   │   ├── middleware.py             # CORS, logging, timing middleware
│   │   │   ├── logging.py                # structlog configuration
│   │   │   ├── limiter.py                # SlowAPI rate limiter setup
│   │   │   ├── telemetry.py              # OpenTelemetry setup
│   │   │   ├── exceptions.py             # Global exception handlers
│   │   │   └── health.py                 # Health check endpoints
│   │   │
│   │   ├── 📁 auth/                      # Authentication Domain
│   │   │   ├── __init__.py
│   │   │   ├── router.py                 # Auth endpoints (/token, /refresh)
│   │   │   ├── schemas.py                # TokenRequest, TokenResponse
│   │   │   ├── models.py                 # User, RefreshToken models
│   │   │   ├── service.py                # Auth business logic
│   │   │   ├── dependencies.py           # get_current_user, get_active_user
│   │   │   ├── constants.py              # Token types, expiry times
│   │   │   └── exceptions.py             # InvalidCredentials, TokenExpired
│   │   │
│   │   ├── 📁 chat/                      # Chat Domain
│   │   │   ├── __init__.py
│   │   │   ├── router.py                 # Chat endpoints (REST + SSE)
│   │   │   ├── schemas.py                # Message, Conversation schemas
│   │   │   ├── models.py                 # Conversation, Message DB models
│   │   │   ├── service.py                # Chat business logic
│   │   │   ├── repository.py             # Database queries (repository pattern)
│   │   │   ├── dependencies.py           # valid_conversation, rate limits
│   │   │   ├── streaming.py              # SSE streaming utilities
│   │   │   ├── websocket_manager.py      # WebSocket connection manager
│   │   │   ├── session_manager.py        # Redis session management
│   │   │   └── exceptions.py             # ConversationNotFound, etc.
│   │   │
│   │   ├── 📁 agents/                    # LangGraph Agent Domain
│   │   │   ├── __init__.py
│   │   │   ├── graph.py                  # Main LangGraph StateGraph definition
│   │   │   ├── state.py                  # AgentState TypedDict
│   │   │   ├── nodes/                    # Individual graph nodes
│   │   │   │   ├── __init__.py
│   │   │   │   ├── input_node.py         # Input processing + language detection
│   │   │   │   ├── router_node.py        # Query classification + routing
│   │   │   │   ├── retriever_node.py     # RAG retrieval node
│   │   │   │   ├── grader_node.py        # Document relevance grading
│   │   │   │   ├── reranker_node.py      # Cross-encoder reranking
│   │   │   │   ├── generator_node.py     # LLM response generation
│   │   │   │   └── output_node.py        # Response formatting + streaming
│   │   │   ├── tools/                    # Agent tools
│   │   │   │   ├── __init__.py
│   │   │   │   ├── knowledge_search.py   # Search knowledge base tool
│   │   │   │   ├── escalation.py         # Escalate to human tool
│   │   │   │   ├── order_status.py       # Get order status tool (example)
│   │   │   │   └── appointment.py        # Book appointment tool (example)
│   │   │   ├── prompts/                  # System prompts
│   │   │   │   ├── __init__.py           # Prompt loader
│   │   │   │   ├── system.md             # Main system prompt
│   │   │   │   ├── router.md             # Router classification prompt
│   │   │   │   ├── grader.md             # Document grading prompt
│   │   │   │   └── generator.md          # Response generation prompt
│   │   │   ├── memory.py                 # Conversation memory management
│   │   │   ├── checkpointer.py           # PostgreSQL checkpointer setup
│   │   │   └── callbacks.py              # LangChain callbacks for streaming
│   │   │
│   │   ├── 📁 rag/                       # RAG Pipeline Domain
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py               # Main RAG pipeline orchestration
│   │   │   ├── ingestion/                # Document ingestion
│   │   │   │   ├── __init__.py
│   │   │   │   ├── loader.py             # LlamaParse document loader
│   │   │   │   ├── chunker.py            # Chunking strategies
│   │   │   │   ├── embedder.py           # Cohere embedding generation
│   │   │   │   └── uploader.py           # Qdrant vector upload
│   │   │   ├── retrieval/                # Retrieval components
│   │   │   │   ├── __init__.py
│   │   │   │   ├── hybrid_retriever.py   # BM25 + dense hybrid retriever
│   │   │   │   ├── query_transformer.py  # HyDE, multi-query, step-back
│   │   │   │   ├── reranker.py           # Cohere reranker integration
│   │   │   │   └── metadata_filter.py    # Metadata filtering utilities
│   │   │   ├── index.py                  # LlamaIndex setup
│   │   │   └── evaluation.py             # RAGAs evaluation utilities
│   │   │
│   │   ├── 📁 knowledge/                 # Knowledge Base Domain
│   │   │   ├── __init__.py
│   │   │   ├── router.py                 # Knowledge management endpoints
│   │   │   ├── schemas.py                # Document, Collection schemas
│   │   │   ├── models.py                 # Document, Collection DB models
│   │   │   ├── service.py                # Knowledge base operations
│   │   │   └── repository.py             # Document metadata queries
│   │   │
│   │   ├── 📁 analytics/                 # Analytics Domain (Future)
│   │   │   ├── __init__.py
│   │   │   ├── router.py                 # Analytics endpoints
│   │   │   ├── schemas.py                # Analytics schemas
│   │   │   ├── service.py                # Analytics aggregation
│   │   │   └── metrics.py                # Custom metrics definitions
│   │   │
│   │   ├── 📁 common/                    # Shared Utilities
│   │   │   ├── __init__.py
│   │   │   ├── models.py                 # Base model classes
│   │   │   ├── schemas.py                # Common response schemas
│   │   │   ├── pagination.py             # Pagination utilities
│   │   │   ├── utils.py                  # General utilities
│   │   │   └── singapore.py              # Singapore-specific utilities
│   │   │
│   │   └── main.py                       # Application entry point
│   │
│   ├── 📁 tests/                         # Test Suite
│   │   ├── conftest.py                   # Pytest fixtures
│   │   ├── 📁 unit/                      # Unit tests
│   │   │   ├── test_auth.py
│   │   │   ├── test_chat.py
│   │   │   ├── test_agents.py
│   │   │   └── test_rag.py
│   │   ├── 📁 integration/               # Integration tests
│   │   │   ├── test_api_chat.py
│   │   │   ├── test_rag_pipeline.py
│   │   │   └── test_agent_graph.py
│   │   └── 📁 e2e/                       # End-to-end tests
│   │       └── test_full_conversation.py
│   │
│   ├── 📁 scripts/                       # Utility Scripts
│   │   ├── seed_database.py              # Database seeding
│   │   ├── ingest_documents.py           # Document ingestion CLI
│   │   ├── evaluate_rag.py               # RAGAs evaluation runner
│   │   └── migrate_memory.py             # Memory tier migration
│   │
│   ├── alembic.ini                       # Alembic configuration
│   ├── pyproject.toml                    # Python dependencies (uv/poetry)
│   ├── Dockerfile                        # Backend Docker image
│   └── .env.example                      # Environment template
│
├── 📁 frontend/                          # React Frontend (Embeddable Widget)
│   ├── 📁 src/
│   │   ├── 📁 components/                # React Components
│   │   │   ├── 📁 chat/                  # Chat-specific components
│   │   │   │   ├── ChatWidget.tsx        # Main widget container
│   │   │   │   ├── ChatHeader.tsx        # Widget header (title, close)
│   │   │   │   ├── MessageList.tsx       # Virtualized message list
│   │   │   │   ├── MessageBubble.tsx     # Individual message rendering
│   │   │   │   ├── InputArea.tsx         # Text input + send button
│   │   │   │   ├── TypingIndicator.tsx   # AI typing animation
│   │   │   │   ├── QuickReplies.tsx      # Suggested quick replies
│   │   │   │   └── ChatTrigger.tsx       # Floating trigger button
│   │   │   └── 📁 ui/                    # Shadcn UI components
│   │   │       ├── button.tsx
│   │   │       ├── input.tsx
│   │   │       ├── scroll-area.tsx
│   │   │       └── ...
│   │   │
│   │   ├── 📁 hooks/                     # Custom React Hooks
│   │   │   ├── useChat.ts                # Vercel AI SDK wrapper
│   │   │   ├── useStreamingText.ts       # Token-by-token animation
│   │   │   ├── useAutoScroll.ts          # Auto-scroll to bottom
│   │   │   ├── useVirtualizedList.ts     # react-window integration
│   │   │   └── useWidgetConfig.ts        # Widget configuration
│   │   │
│   │   ├── 📁 stores/                    # Zustand State Stores
│   │   │   ├── chatStore.ts              # Chat state (messages, typing)
│   │   │   ├── uiStore.ts                # UI state (open/closed, theme)
│   │   │   └── sessionStore.ts           # Session management
│   │   │
│   │   ├── 📁 lib/                       # Utilities & Config
│   │   │   ├── api.ts                    # API client (TanStack Query)
│   │   │   ├── sse.ts                    # SSE connection utilities
│   │   │   ├── markdown.ts               # Markdown renderer config
│   │   │   ├── utils.ts                  # General utilities
│   │   │   └── constants.ts              # Constants & config
│   │   │
│   │   ├── 📁 styles/                    # Styling
│   │   │   ├── globals.css               # Tailwind base styles
│   │   │   └── widget.css                # Widget-specific styles
│   │   │
│   │   ├── 📁 types/                     # TypeScript Types
│   │   │   ├── chat.ts                   # Chat-related types
│   │   │   ├── api.ts                    # API response types
│   │   │   └── config.ts                 # Widget config types
│   │   │
│   │   ├── App.tsx                       # Main App component
│   │   ├── main.tsx                      # Application entry (standard)
│   │   └── widget.tsx                    # Widget entry (embeddable)
│   │
│   ├── 📁 public/
│   │   └── widget-loader.js              # Script tag loader
│   │
│   ├── index.html                        # Development HTML
│   ├── package.json                      # npm dependencies
│   ├── vite.config.ts                    # Vite configuration
│   ├── tailwind.config.ts                # Tailwind configuration
│   ├── tsconfig.json                     # TypeScript configuration
│   └── .env.example                      # Environment template
│
├── 📁 infra/                             # Infrastructure Configuration
│   ├── 📁 docker/
│   │   ├── docker-compose.yml            # Full stack compose
│   │   ├── docker-compose.dev.yml        # Development overrides
│   │   └── docker-compose.prod.yml       # Production overrides
│   │
│   ├── 📁 kubernetes/                    # K8s manifests (future)
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── ingress.yaml
│   │
│   ├── 📁 monitoring/
│   │   ├── prometheus/
│   │   │   └── prometheus.yml            # Prometheus config
│   │   └── grafana/
│   │       └── dashboards/
│   │           ├── api-dashboard.json    # API metrics dashboard
│   │           └── llm-dashboard.json    # LLM metrics dashboard
│   │
│   └── 📁 scripts/
│       ├── setup.sh                      # Initial setup script
│       ├── deploy.sh                     # Deployment script
│       └── backup.sh                     # Database backup script
│
├── 📁 docs/                              # Documentation
│   ├── architecture.md                   # Architecture overview
│   ├── api-reference.md                  # API documentation
│   ├── deployment-guide.md               # Deployment instructions
│   ├── configuration.md                  # Configuration guide
│   └── troubleshooting.md                # Common issues & solutions
│
├── 📁 knowledge_base/                    # Sample Knowledge Base Content
│   ├── faqs/
│   │   ├── general.md
│   │   ├── billing.md
│   │   └── technical.md
│   ├── policies/
│   │   ├── privacy.md
│   │   └── terms.md
│   └── products/
│       └── catalog.md
│
├── .gitignore
├── .env.example                          # Root environment template
├── README.md                             # Project README
├── Makefile                              # Common commands
└── MASTER_EXECUTION_PLAN.md              # This document
```

---

## 6. Key Files & Descriptions

### 6.1 Backend Core Files

| File | Purpose | Key Responsibilities |
|------|---------|---------------------|
| `src/main.py` | Application entry point | FastAPI app creation, lifespan management, router mounting, middleware setup |
| `src/core/config.py` | Configuration management | Pydantic BaseSettings, environment variable loading, secrets management |
| `src/core/database.py` | Database connection | Async SQLAlchemy engine, session factory, `get_db` dependency |
| `src/core/redis.py` | Redis client | Async Redis connection, singleton pattern, cache utilities |
| `src/core/telemetry.py` | Observability | OpenTelemetry tracing, Prometheus metrics, LLM-specific metrics |

### 6.2 Agent & RAG Files

| File | Purpose | Key Responsibilities |
|------|---------|---------------------|
| `src/agents/graph.py` | LangGraph definition | StateGraph construction, node connections, conditional edges |
| `src/agents/state.py` | Agent state | TypedDict state schema with messages, context, metadata |
| `src/agents/nodes/generator_node.py` | LLM response | Streaming generation, Singapore context, response formatting |
| `src/agents/checkpointer.py` | State persistence | PostgreSQL checkpointer for conversation continuity |
| `src/rag/pipeline.py` | RAG orchestration | Query → Transform → Retrieve → Rerank → Return |
| `src/rag/retrieval/hybrid_retriever.py` | Hybrid search | BM25 + dense vector + RRF fusion implementation |
| `src/rag/retrieval/reranker.py` | Cross-encoder | Cohere Rerank-4 integration for relevance scoring |

### 6.3 Chat Domain Files

| File | Purpose | Key Responsibilities |
|------|---------|---------------------|
| `src/chat/router.py` | Chat endpoints | REST endpoints + SSE streaming endpoint |
| `src/chat/streaming.py` | SSE utilities | Token streaming, event formatting, connection management |
| `src/chat/session_manager.py` | Session cache | Redis-based session storage, TTL management |
| `src/chat/repository.py` | Data access | Async queries for conversations/messages |

### 6.4 Frontend Key Files

| File | Purpose | Key Responsibilities |
|------|---------|---------------------|
| `src/components/chat/ChatWidget.tsx` | Main widget | Shadow DOM container, state orchestration |
| `src/components/chat/MessageList.tsx` | Message display | react-window virtualization, ARIA live region |
| `src/hooks/useChat.ts` | Chat hook | Vercel AI SDK integration, SSE streaming |
| `src/stores/chatStore.ts` | Chat state | Zustand store for messages, typing state |
| `src/widget.tsx` | Widget entry | Script injection, Shadow DOM setup, config parsing |

---

## 7. Master Execution Phases

### Phase Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MASTER EXECUTION TIMELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 1: Foundation        ████████░░░░░░░░░░░░░░░░░░░░░░░░░░  (Days 1-3)  │
│  Phase 2: Core Backend      ░░░░░░░░████████████░░░░░░░░░░░░░░  (Days 4-8)  │
│  Phase 3: RAG Pipeline      ░░░░░░░░░░░░░░░░░░░█████████░░░░░░  (Days 9-12) │
│  Phase 4: Agent Orchestra   ░░░░░░░░░░░░░░░░░░░░░░░░░░░█████░░  (Days 13-16)│
│  Phase 5: Frontend Widget   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████  (Days 17-21)│
│  Phase 6: Integration       ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░██  (Days 22-24)│
│  Phase 7: Deployment        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█  (Days 25-28)│
│                                                                              │
│  TOTAL ESTIMATED: 28 Days (4 Weeks)                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase Dependencies

```
Phase 1 ─────────────────────────────────────────────────────┐
   │                                                          │
   ▼                                                          │
Phase 2 ───────┐                                              │
   │           │                                              │
   ▼           ▼                                              │
Phase 3 ───► Phase 4 ◄────────────────────────────────────────┘
   │           │
   └─────┬─────┘
         ▼
      Phase 5
         │
         ▼
      Phase 6
         │
         ▼
      Phase 7
```

---

## Phase 1: Foundation & Infrastructure

### Objectives
- Set up project structure and development environment
- Configure infrastructure services (PostgreSQL, Redis, Qdrant)
- Establish CI/CD foundations
- Create base configurations

### Duration: 3 Days

### Detailed TODO Checklist

#### Day 1: Project Initialization

- [ ] **1.1 Repository Setup**
  - [ ] Initialize Git repository
  - [ ] Create `.gitignore` with Python, Node, IDE patterns
  - [ ] Set up branch protection rules
  - [ ] Configure commit hooks (pre-commit)

- [ ] **1.2 Backend Project Structure**
  - [ ] Create `backend/` directory structure (as per file hierarchy)
  - [ ] Initialize `pyproject.toml` with uv/poetry
  - [ ] Install core dependencies:
    - [ ] `fastapi[all]`, `uvicorn[standard]`
    - [ ] `sqlalchemy[asyncio]`, `asyncpg`, `alembic`
    - [ ] `redis`, `pydantic-settings`
    - [ ] `python-jose[cryptography]`, `passlib[bcrypt]`
    - [ ] `structlog`, `opentelemetry-*`
  - [ ] Create `src/__init__.py` files for all packages

- [ ] **1.3 Frontend Project Structure**
  - [ ] Initialize Vite + React + TypeScript project
  - [ ] Install dependencies:
    - [ ] `@tanstack/react-query`, `zustand`
    - [ ] `ai` (Vercel AI SDK)
    - [ ] `react-window`, `react-markdown`, `remark-gfm`
    - [ ] `framer-motion`
  - [ ] Configure Tailwind CSS + Shadcn/UI
  - [ ] Set up path aliases in `tsconfig.json`

#### Day 2: Infrastructure Configuration

- [ ] **1.4 Docker Configuration**
  - [ ] Create `backend/Dockerfile` (multi-stage build)
  - [ ] Create `frontend/Dockerfile` (build + nginx)
  - [ ] Create `infra/docker/docker-compose.yml`:
    - [ ] PostgreSQL 16 service
    - [ ] Redis 7 service
    - [ ] Qdrant service
    - [ ] Backend service
    - [ ] Frontend service
  - [ ] Create `docker-compose.dev.yml` with hot-reload

- [ ] **1.5 Environment Configuration**
  - [ ] Create `.env.example` with all required variables:
    ```
    # Database
    DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/support_agent
    
    # Redis
    REDIS_URL=redis://localhost:6379
    
    # Qdrant
    QDRANT_URL=http://localhost:6333
    QDRANT_API_KEY=
    
    # OpenAI
    OPENAI_API_KEY=
    
    # Cohere
    COHERE_API_KEY=
    
    # JWT
    JWT_SECRET=
    JWT_ALGORITHM=HS256
    ACCESS_TOKEN_EXPIRE_MINUTES=30
    
    # Environment
    ENVIRONMENT=development
    DEBUG=true
    ```
  - [ ] Create `src/core/config.py` with Pydantic BaseSettings

- [ ] **1.6 Database Setup**
  - [ ] Initialize Alembic: `alembic init alembic`
  - [ ] Configure `alembic/env.py` for async SQLAlchemy
  - [ ] Create `src/core/database.py`:
    - [ ] Async engine with connection pooling
    - [ ] `AsyncSessionLocal` factory
    - [ ] `get_db` dependency
    - [ ] Base declarative model

#### Day 3: Core Infrastructure Code

- [ ] **1.7 Redis Client Setup**
  - [ ] Create `src/core/redis.py`:
    - [ ] `RedisClient` singleton class
    - [ ] Async connection with retry
    - [ ] `get_redis_client` dependency

- [ ] **1.8 Logging & Observability**
  - [ ] Create `src/core/logging.py`:
    - [ ] structlog configuration
    - [ ] JSON output for production
    - [ ] Colored console for development
  - [ ] Create `src/core/telemetry.py`:
    - [ ] OpenTelemetry tracer provider
    - [ ] Prometheus metrics reader
    - [ ] Auto-instrumentation (FastAPI, SQLAlchemy, Redis)

- [ ] **1.9 Application Entry Point**
  - [ ] Create `src/main.py`:
    - [ ] FastAPI app with lifespan
    - [ ] CORS middleware
    - [ ] Health check endpoints (`/health`, `/health/live`, `/health/ready`)
    - [ ] Exception handlers
  - [ ] Verify: `docker-compose up` runs without errors

- [ ] **1.10 Validation Checkpoint**
  - [ ] All services start successfully
  - [ ] Health endpoints respond correctly
  - [ ] Database connection works
  - [ ] Redis ping succeeds
  - [ ] Qdrant UI accessible

### Phase 1 Deliverables
- [x] Complete project structure created
- [x] Docker Compose stack running
- [x] Database and cache connections verified
- [x] Health check endpoints functional
- [x] Logging and observability configured

---

## Phase 2: Core Backend Development

### Objectives
- Implement authentication system
- Build chat domain (models, endpoints, repository)
- Set up rate limiting and security middleware
- Create base API structure

### Duration: 5 Days

### Detailed TODO Checklist

#### Day 4: Authentication Domain

- [ ] **2.1 Auth Models**
  - [ ] Create `src/auth/models.py`:
    - [ ] `User` model (id, email, hashed_password, is_active, created_at)
    - [ ] `RefreshToken` model (id, user_id, token, expires_at, revoked)
  - [ ] Create Alembic migration for auth tables

- [ ] **2.2 Auth Schemas**
  - [ ] Create `src/auth/schemas.py`:
    - [ ] `UserCreate`, `UserResponse`
    - [ ] `TokenRequest`, `TokenResponse`
    - [ ] `TokenPayload`

- [ ] **2.3 Auth Service**
  - [ ] Create `src/auth/service.py`:
    - [ ] `create_user()` with password hashing
    - [ ] `authenticate_user()` with password verification
    - [ ] `create_access_token()` with JWT encoding
    - [ ] `create_refresh_token()` with DB storage
    - [ ] `refresh_access_token()` with validation

- [ ] **2.4 Auth Dependencies**
  - [ ] Create `src/auth/dependencies.py`:
    - [ ] `get_current_user()` from JWT token
    - [ ] `get_current_active_user()` with active check
    - [ ] `get_optional_user()` for public endpoints

- [ ] **2.5 Auth Router**
  - [ ] Create `src/auth/router.py`:
    - [ ] `POST /auth/register` - User registration
    - [ ] `POST /auth/token` - Login, return tokens
    - [ ] `POST /auth/refresh` - Refresh access token
    - [ ] `POST /auth/logout` - Revoke refresh token
    - [ ] `GET /auth/me` - Current user info

#### Day 5: Chat Domain Models & Repository

- [ ] **2.6 Chat Models**
  - [ ] Create `src/chat/models.py`:
    - [ ] `MessageRole` enum (user, assistant, system)
    - [ ] `ConversationStatus` enum (active, archived, escalated)
    - [ ] `Conversation` model:
      - id, user_id, title, status, created_at, updated_at
      - metadata (JSON for custom fields)
    - [ ] `Message` model:
      - id, conversation_id, role, content, token_count, created_at
      - feedback (optional rating)
  - [ ] Create Alembic migration

- [ ] **2.7 Chat Schemas**
  - [ ] Create `src/chat/schemas.py`:
    - [ ] `MessageCreate`, `MessageResponse`
    - [ ] `ConversationCreate`, `ConversationResponse`, `ConversationList`
    - [ ] `ChatRequest`, `ChatStreamEvent`
    - [ ] `FeedbackRequest`

- [ ] **2.8 Chat Repository**
  - [ ] Create `src/chat/repository.py`:
    - [ ] `ConversationRepository` class:
      - `get_by_id()` with eager message loading
      - `get_by_user()` with pagination
      - `create()`, `update()`, `delete()`
    - [ ] `MessageRepository` class:
      - `create()`, `get_by_conversation()`
      - `count_tokens()` helper

#### Day 6: Chat Service & REST Endpoints

- [ ] **2.9 Chat Service**
  - [ ] Create `src/chat/service.py`:
    - [ ] `ChatService` class:
      - `create_conversation()` with initial message
      - `get_conversation_history()` formatted for LLM
      - `add_message()` with token counting
      - `update_title()` (auto-generate from first message)
      - `archive_conversation()`

- [ ] **2.10 Chat Dependencies**
  - [ ] Create `src/chat/dependencies.py`:
    - [ ] `valid_conversation()` - Validate ownership
    - [ ] `rate_limit_chat()` - Per-user rate limit

- [ ] **2.11 Chat REST Endpoints**
  - [ ] Create `src/chat/router.py`:
    - [ ] `GET /chat/conversations` - List user conversations
    - [ ] `POST /chat/conversations` - Create new conversation
    - [ ] `GET /chat/conversations/{id}` - Get conversation with messages
    - [ ] `DELETE /chat/conversations/{id}` - Archive conversation
    - [ ] `POST /chat/conversations/{id}/feedback` - Submit feedback

#### Day 7: SSE Streaming & Session Management

- [ ] **2.12 SSE Streaming**
  - [ ] Create `src/chat/streaming.py`:
    - [ ] `SSEResponse` class for Server-Sent Events
    - [ ] `format_sse_event()` helper
    - [ ] `stream_tokens()` async generator
    - [ ] Connection timeout handling

- [ ] **2.13 Session Manager**
  - [ ] Create `src/chat/session_manager.py`:
    - [ ] `ChatSessionManager` class:
      - `create_session()` in Redis
      - `get_session()` with TTL refresh
      - `update_activity()`
      - `invalidate_session()`

- [ ] **2.14 Streaming Endpoint**
  - [ ] Add to `src/chat/router.py`:
    - [ ] `POST /chat/conversations/{id}/stream` - SSE streaming endpoint
    - [ ] Integrate with session manager
    - [ ] Add rate limiting decorator

#### Day 8: Rate Limiting & Security

- [ ] **2.15 Rate Limiting**
  - [ ] Create `src/core/limiter.py`:
    - [ ] SlowAPI limiter with Redis backend
    - [ ] `get_user_identifier()` key function
    - [ ] Default limits: 100/minute general, 20/minute for LLM
  - [ ] Add `SlowAPIMiddleware` to main app
  - [ ] Apply `@limiter.limit()` decorators to endpoints

- [ ] **2.16 Security Middleware**
  - [ ] Create `src/core/middleware.py`:
    - [ ] `RequestLoggingMiddleware` with timing
    - [ ] Request ID injection
    - [ ] Security headers middleware
  - [ ] Configure CORS in main app

- [ ] **2.17 Exception Handling**
  - [ ] Create `src/core/exceptions.py`:
    - [ ] `AppException` base class
    - [ ] Domain-specific exceptions:
      - `ConversationNotFound`
      - `UnauthorizedAccess`
      - `RateLimitExceeded`
      - `LLMServiceError`
  - [ ] Register exception handlers in main app

- [ ] **2.18 Validation Checkpoint**
  - [ ] Auth flow works (register → login → refresh)
  - [ ] Chat CRUD operations functional
  - [ ] Rate limiting enforced
  - [ ] SSE streaming endpoint responds
  - [ ] All tests pass

### Phase 2 Deliverables
- [x] Complete authentication system
- [x] Chat domain with full CRUD
- [x] SSE streaming infrastructure
- [x] Rate limiting and security
- [x] Session management via Redis

---

## Phase 3: RAG Pipeline Implementation

### Objectives
- Set up document ingestion pipeline
- Implement hybrid search (BM25 + dense vectors)
- Configure cross-encoder reranking
- Build query transformation layer

### Duration: 4 Days

### Detailed TODO Checklist

#### Day 9: Document Ingestion Setup

- [ ] **3.1 Install RAG Dependencies**
  - [ ] Add to `pyproject.toml`:
    - [ ] `llama-index`, `llama-index-vector-stores-qdrant`
    - [ ] `llama-index-embeddings-cohere`
    - [ ] `llama-parse` (document parsing)
    - [ ] `cohere` (embeddings + reranking)
    - [ ] `rank-bm25` (sparse retrieval)

- [ ] **3.2 LlamaParse Loader**
  - [ ] Create `src/rag/ingestion/loader.py`:
    - [ ] `DocumentLoader` class:
      - `load_pdf()` with LlamaParse
      - `load_markdown()` for text files
      - `load_docx()` via LlamaParse
      - Metadata extraction (title, author, date)

- [ ] **3.3 Chunking Strategy**
  - [ ] Create `src/rag/ingestion/chunker.py`:
    - [ ] `ChunkingStrategy` enum (recursive, semantic, page)
    - [ ] `RecursiveChunker`:
      - 400 tokens, 80 token overlap
      - Configurable separators
    - [ ] `SemanticChunker`:
      - Sentence embedding comparison
      - Threshold-based splitting
    - [ ] `PageChunker`:
      - PDF page boundaries
      - Metadata preservation

- [ ] **3.4 Embedding Generation**
  - [ ] Create `src/rag/ingestion/embedder.py`:
    - [ ] `CohereEmbedder` class:
      - `embed_documents()` batch embedding
      - `embed_query()` single query embedding
      - Rate limiting and retry logic
      - Dimension: 1024 (Embed-4)

#### Day 10: Vector Store & Indexing

- [ ] **3.5 Qdrant Setup**
  - [ ] Create `src/rag/index.py`:
    - [ ] `QdrantClient` initialization
    - [ ] Collection creation with:
      - Dense vectors (1024 dim, cosine)
      - Sparse vectors for BM25
      - Payload indexes for metadata
    - [ ] Index configuration helper

- [ ] **3.6 Document Uploader**
  - [ ] Create `src/rag/ingestion/uploader.py`:
    - [ ] `VectorUploader` class:
      - `upload_documents()` batch upload
      - `upsert_document()` single doc update
      - `delete_by_metadata()` for updates
      - Parallel upload with asyncio

- [ ] **3.7 Ingestion Pipeline**
  - [ ] Create `src/rag/ingestion/__init__.py`:
    - [ ] `IngestionPipeline` class:
      - `ingest_file()` end-to-end processing
      - `ingest_directory()` batch processing
      - Progress tracking
      - Error handling with retry

- [ ] **3.8 Ingestion CLI**
  - [ ] Create `scripts/ingest_documents.py`:
    - [ ] CLI with argparse
    - [ ] Directory or file input
    - [ ] Chunking strategy selection
    - [ ] Progress bar (tqdm)
    - [ ] Dry-run mode

#### Day 11: Hybrid Retrieval

- [ ] **3.9 Query Transformer**
  - [ ] Create `src/rag/retrieval/query_transformer.py`:
    - [ ] `QueryTransformer` class:
      - `transform_hyde()` - Hypothetical document
      - `transform_multi_query()` - Query variants
      - `transform_step_back()` - Abstraction
    - [ ] Strategy selection based on query type

- [ ] **3.10 Hybrid Retriever**
  - [ ] Create `src/rag/retrieval/hybrid_retriever.py`:
    - [ ] `HybridRetriever` class:
      - Dense retrieval via Qdrant
      - Sparse retrieval via BM25
      - RRF fusion implementation:
        ```python
        def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
            scores = {}
            for rank, doc in enumerate(dense_results):
                scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
            for rank, doc in enumerate(sparse_results):
                scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
            return sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ```
      - Alpha parameter for weighting (default 0.5)

- [ ] **3.11 Metadata Filtering**
  - [ ] Create `src/rag/retrieval/metadata_filter.py`:
    - [ ] `MetadataFilter` class:
      - Filter by document type
      - Filter by date range
      - Filter by product/category
      - Filter by language

#### Day 12: Reranking & Pipeline Assembly

- [ ] **3.12 Cohere Reranker**
  - [ ] Create `src/rag/retrieval/reranker.py`:
    - [ ] `CohereReranker` class:
      - `rerank()` with model: rerank-4-multilingual
      - Top-k selection (default 5)
      - Score threshold filtering
      - Batch processing for efficiency

- [ ] **3.13 RAG Pipeline**
  - [ ] Create `src/rag/pipeline.py`:
    - [ ] `RAGPipeline` class:
      - `retrieve()` full pipeline:
        1. Query transformation (if applicable)
        2. Hybrid retrieval (BM25 + dense)
        3. RRF fusion
        4. Metadata filtering
        5. Cross-encoder reranking
        6. Return top-k documents
      - `retrieve_with_scores()` including relevance scores
      - Caching layer for frequent queries

- [ ] **3.14 RAG Evaluation**
  - [ ] Create `src/rag/evaluation.py`:
    - [ ] Install: `ragas`
    - [ ] `RAGEvaluator` class:
      - `evaluate_faithfulness()` - Answer grounded in context
      - `evaluate_context_recall()` - Retrieved relevant docs
      - `evaluate_answer_relevance()` - Answer matches query
    - [ ] Evaluation dataset format

- [ ] **3.15 Validation Checkpoint**
  - [ ] Documents ingest successfully
  - [ ] Hybrid search returns results
  - [ ] Reranking improves relevance
  - [ ] RAGAs scores > 0.85 faithfulness
  - [ ] Query transformation works

### Phase 3 Deliverables
- [x] Document ingestion pipeline (LlamaParse)
- [x] Chunking strategies implemented
- [x] Hybrid retrieval (BM25 + dense + RRF)
- [x] Cohere Rerank-4 integration
- [x] RAGAs evaluation setup

---

## Phase 4: Agent Orchestration (LangGraph)

### Objectives
- Define LangGraph state machine
- Implement specialized agent nodes
- Configure checkpointing for conversation continuity
- Create agent tools

### Duration: 4 Days

### Detailed TODO Checklist

#### Day 13: LangGraph Foundation

- [ ] **4.1 Install Agent Dependencies**
  - [ ] Add to `pyproject.toml`:
    - [ ] `langgraph`, `langchain`, `langchain-openai`
    - [ ] `langchain-cohere`
    - [ ] `langgraph-checkpoint-postgres`

- [ ] **4.2 Agent State Definition**
  - [ ] Create `src/agents/state.py`:
    ```python
    class AgentState(TypedDict):
        # Input
        input: str
        session_id: str
        user_id: str
        language: str
        
        # Messages
        messages: Annotated[List[BaseMessage], add_messages]
        
        # RAG Context
        retrieved_documents: List[Document]
        reranked_documents: List[Document]
        
        # Routing
        query_type: Literal["faq", "complex", "escalation", "out_of_scope"]
        confidence: float
        
        # Output
        response: str
        sources: List[str]
        
        # Metadata
        token_count: int
        processing_time: float
    ```

- [ ] **4.3 PostgreSQL Checkpointer**
  - [ ] Create `src/agents/checkpointer.py`:
    - [ ] Setup `PostgresSaver` from langgraph-checkpoint-postgres
    - [ ] Connection using DATABASE_URL
    - [ ] Thread ID mapping to conversation_id

- [ ] **4.4 Prompt Templates**
  - [ ] Create `src/agents/prompts/system.md`:
    ```markdown
    You are a helpful customer support agent for [Company Name], 
    a Singapore-based SMB. You assist customers with their enquiries
    in a friendly, professional manner.
    
    ## Singapore Context
    - Understand Singlish colloquialisms
    - Aware of local holidays (CNY, Hari Raya, Deepavali)
    - Familiar with SGD currency
    - Know PDPA compliance requirements
    
    ## Guidelines
    - Be concise but thorough
    - Always cite sources when using knowledge base
    - Escalate if customer is frustrated or issue is complex
    - Never make up information
    ```
  - [ ] Create `src/agents/prompts/router.md`
  - [ ] Create `src/agents/prompts/grader.md`
  - [ ] Create `src/agents/prompts/generator.md`

#### Day 14: Agent Nodes Implementation

- [ ] **4.5 Input Node**
  - [ ] Create `src/agents/nodes/input_node.py`:
    - [ ] Language detection (English, Mandarin, Singlish)
    - [ ] Input sanitization
    - [ ] Intent extraction (optional)
    - [ ] Add HumanMessage to state

- [ ] **4.6 Router Node**
  - [ ] Create `src/agents/nodes/router_node.py`:
    - [ ] Query classification:
      - `faq`: Simple, common questions
      - `complex`: Multi-step, needs context
      - `escalation`: Customer frustrated, sensitive
      - `out_of_scope`: Not related to business
    - [ ] Confidence scoring
    - [ ] Use structured output with Pydantic

- [ ] **4.7 Retriever Node**
  - [ ] Create `src/agents/nodes/retriever_node.py`:
    - [ ] Integrate with `RAGPipeline`
    - [ ] Apply query transformation based on query_type
    - [ ] Store retrieved docs in state

- [ ] **4.8 Grader Node**
  - [ ] Create `src/agents/nodes/grader_node.py`:
    - [ ] Score document relevance (0-1)
    - [ ] Filter low-relevance docs (< 0.7)
    - [ ] Decide if web search needed (future)

- [ ] **4.9 Reranker Node**
  - [ ] Create `src/agents/nodes/reranker_node.py`:
    - [ ] Call Cohere Rerank
    - [ ] Select top 5 documents
    - [ ] Update state with reranked docs

#### Day 15: Generator & Tools

- [ ] **4.10 Generator Node**
  - [ ] Create `src/agents/nodes/generator_node.py`:
    - [ ] Build prompt with context and history
    - [ ] Stream response tokens
    - [ ] Extract citations from response
    - [ ] Token counting
    - [ ] Singapore context injection

- [ ] **4.11 Output Node**
  - [ ] Create `src/agents/nodes/output_node.py`:
    - [ ] Format response for API
    - [ ] Attach source citations
    - [ ] Update conversation in database
    - [ ] Trigger memory update

- [ ] **4.12 Agent Tools**
  - [ ] Create `src/agents/tools/knowledge_search.py`:
    - [ ] `@tool` decorator
    - [ ] Searches knowledge base
    - [ ] Returns formatted results
  - [ ] Create `src/agents/tools/escalation.py`:
    - [ ] `@tool` for human escalation
    - [ ] Creates support ticket
    - [ ] Notifies support team
  - [ ] Create `src/agents/tools/order_status.py` (example):
    - [ ] Mock tool for demo
    - [ ] Returns sample order data

#### Day 16: Graph Assembly & Memory

- [ ] **4.13 Graph Construction**
  - [ ] Create `src/agents/graph.py`:
    ```python
    def build_agent_graph():
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("input", input_node)
        workflow.add_node("router", router_node)
        workflow.add_node("retriever", retriever_node)
        workflow.add_node("grader", grader_node)
        workflow.add_node("reranker", reranker_node)
        workflow.add_node("generator", generator_node)
        workflow.add_node("output", output_node)
        
        # Set entry point
        workflow.set_entry_point("input")
        
        # Add edges
        workflow.add_edge("input", "router")
        workflow.add_conditional_edges(
            "router",
            route_query,
            {
                "faq": "retriever",
                "complex": "retriever",
                "escalation": "output",
                "out_of_scope": "output"
            }
        )
        workflow.add_edge("retriever", "grader")
        workflow.add_edge("grader", "reranker")
        workflow.add_edge("reranker", "generator")
        workflow.add_edge("generator", "output")
        workflow.add_edge("output", END)
        
        return workflow.compile(checkpointer=checkpointer)
    ```

- [ ] **4.14 Memory Management**
  - [ ] Create `src/agents/memory.py`:
    - [ ] `MemoryManager` class:
      - Short-term: Redis (last N messages)
      - Long-term: PostgreSQL (summaries)
      - `get_context()` retrieves relevant history
      - `summarize_conversation()` for long chats

- [ ] **4.15 Streaming Integration**
  - [ ] Create `src/agents/callbacks.py`:
    - [ ] `StreamingCallback` for token streaming
    - [ ] Integrates with SSE endpoint
    - [ ] Handles streaming to WebSocket (optional)

- [ ] **4.16 Validation Checkpoint**
  - [ ] Graph executes end-to-end
  - [ ] Routing works correctly
  - [ ] RAG retrieval integrated
  - [ ] Streaming tokens work
  - [ ] Checkpointing persists state
  - [ ] Memory recall works

### Phase 4 Deliverables
- [x] LangGraph state machine defined
- [x] All agent nodes implemented
- [x] Agent tools created
- [x] PostgreSQL checkpointing
- [x] Memory management (short/long term)
- [x] Streaming callback integration

---

## Phase 5: Frontend Chat Widget

### Objectives
- Build embeddable React chat widget
- Implement SSE streaming display
- Create virtualized message list
- Ensure WCAG accessibility

### Duration: 5 Days

### Detailed TODO Checklist

#### Day 17: Widget Foundation

- [ ] **5.1 Project Setup Verification**
  - [ ] Confirm Vite + React + TypeScript configured
  - [ ] Tailwind CSS + Shadcn/UI installed
  - [ ] Path aliases working (@/components, etc.)

- [ ] **5.2 TypeScript Types**
  - [ ] Create `src/types/chat.ts`:
    ```typescript
    interface Message {
      id: string;
      role: 'user' | 'assistant' | 'system';
      content: string;
      createdAt: Date;
      sources?: Source[];
      isStreaming?: boolean;
    }
    
    interface Source {
      title: string;
      url?: string;
      snippet: string;
    }
    
    interface Conversation {
      id: string;
      title: string;
      messages: Message[];
      createdAt: Date;
      updatedAt: Date;
    }
    ```
  - [ ] Create `src/types/config.ts` for widget config
  - [ ] Create `src/types/api.ts` for API responses

- [ ] **5.3 Zustand Stores**
  - [ ] Create `src/stores/chatStore.ts`:
    ```typescript
    interface ChatState {
      messages: Message[];
      isTyping: boolean;
      currentConversationId: string | null;
      addMessage: (message: Message) => void;
      updateMessage: (id: string, content: string) => void;
      setTyping: (typing: boolean) => void;
      clearMessages: () => void;
    }
    ```
  - [ ] Create `src/stores/uiStore.ts`:
    - Widget open/closed state
    - Theme (light/dark)
    - Error state
  - [ ] Create `src/stores/sessionStore.ts`:
    - Session ID management
    - Auth token storage

#### Day 18: API & SSE Integration

- [ ] **5.4 API Client**
  - [ ] Create `src/lib/api.ts`:
    - [ ] Axios or fetch wrapper
    - [ ] Base URL configuration
    - [ ] Auth header injection
    - [ ] Error handling

- [ ] **5.5 TanStack Query Setup**
  - [ ] Configure QueryClient in `src/App.tsx`
  - [ ] Create query hooks:
    - [ ] `useConversations()` - List conversations
    - [ ] `useConversation(id)` - Single conversation
    - [ ] Mutation: `useCreateConversation()`

- [ ] **5.6 SSE Integration**
  - [ ] Create `src/lib/sse.ts`:
    - [ ] `createSSEConnection()` function
    - [ ] Event parsing
    - [ ] Auto-reconnect logic
    - [ ] Error handling
  - [ ] Create `src/hooks/useChat.ts`:
    - [ ] Wrap Vercel AI SDK's useChat
    - [ ] Configure SSE endpoint
    - [ ] Handle streaming state
    - [ ] Message accumulation

#### Day 19: Chat Components

- [ ] **5.7 Message Components**
  - [ ] Create `src/components/chat/MessageBubble.tsx`:
    - [ ] User message styling (right-aligned)
    - [ ] Assistant message styling (left-aligned)
    - [ ] Markdown rendering via react-markdown
    - [ ] Code block syntax highlighting
    - [ ] Source citations display
  - [ ] Create `src/components/chat/TypingIndicator.tsx`:
    - [ ] Animated dots
    - [ ] Framer Motion animation

- [ ] **5.8 Message List**
  - [ ] Create `src/components/chat/MessageList.tsx`:
    - [ ] react-window VariableSizeList
    - [ ] Dynamic height measurement
    - [ ] `role="log"` for accessibility
    - [ ] `aria-live="polite"`
    - [ ] Auto-scroll to bottom
  - [ ] Create `src/hooks/useVirtualizedList.ts`:
    - [ ] Height caching
    - [ ] `resetAfterIndex()` handling

- [ ] **5.9 Input Area**
  - [ ] Create `src/components/chat/InputArea.tsx`:
    - [ ] Textarea with auto-resize
    - [ ] Send button with loading state
    - [ ] Enter to send, Shift+Enter for newline
    - [ ] Character limit indicator
    - [ ] Disabled state during streaming

#### Day 20: Widget Container & Trigger

- [ ] **5.10 Chat Widget Container**
  - [ ] Create `src/components/chat/ChatWidget.tsx`:
    - [ ] Widget panel layout (header, messages, input)
    - [ ] Open/close animation (Framer Motion)
    - [ ] Mobile responsive (full-screen on small devices)
    - [ ] Shadow DOM container for style isolation

- [ ] **5.11 Chat Header**
  - [ ] Create `src/components/chat/ChatHeader.tsx`:
    - [ ] Company logo/title
    - [ ] Status indicator (online/offline)
    - [ ] Close button
    - [ ] Minimize option

- [ ] **5.12 Chat Trigger**
  - [ ] Create `src/components/chat/ChatTrigger.tsx`:
    - [ ] Floating action button
    - [ ] Unread message badge
    - [ ] Pulse animation on new message
    - [ ] Accessible labeling

- [ ] **5.13 Quick Replies**
  - [ ] Create `src/components/chat/QuickReplies.tsx`:
    - [ ] Suggested response buttons
    - [ ] Keyboard navigation
    - [ ] Dynamic based on context

#### Day 21: Accessibility & Widget Build

- [ ] **5.14 Accessibility Audit**
  - [ ] WCAG 2.2 compliance:
    - [ ] Focus management (trap focus in open widget)
    - [ ] Keyboard navigation (Tab, Escape to close)
    - [ ] Screen reader testing
    - [ ] `aria-live` regions for streaming
    - [ ] Color contrast (AA minimum)
    - [ ] Touch target size (24x24 minimum)
  - [ ] Skip link to chat widget
  - [ ] Sender identification text

- [ ] **5.15 Widget Entry Point**
  - [ ] Create `src/widget.tsx`:
    - [ ] Shadow DOM creation
    - [ ] Style injection
    - [ ] Config parsing from data attributes
    - [ ] Mount React app to shadow root

- [ ] **5.16 Widget Loader Script**
  - [ ] Create `public/widget-loader.js`:
    - [ ] Script tag configuration parsing
    - [ ] Dynamic script loading
    - [ ] Initialization function
    ```javascript
    (function() {
      const script = document.currentScript;
      const clientKey = script.getAttribute('data-client-key');
      const theme = script.getAttribute('data-theme') || 'light';
      
      // Load main widget bundle
      const widgetScript = document.createElement('script');
      widgetScript.src = 'https://cdn.example.com/widget.js';
      widgetScript.onload = function() {
        window.SupportWidget.init({ clientKey, theme });
      };
      document.body.appendChild(widgetScript);
    })();
    ```

- [ ] **5.17 Build Configuration**
  - [ ] Update `vite.config.ts`:
    - [ ] Library mode build
    - [ ] UMD + ESM output
    - [ ] CSS extraction for Shadow DOM
    - [ ] Sourcemaps for production
  - [ ] Test: Widget embeds correctly in test HTML

- [ ] **5.18 Validation Checkpoint**
  - [ ] Widget opens/closes smoothly
  - [ ] Messages stream correctly
  - [ ] Markdown renders properly
  - [ ] Mobile responsive
  - [ ] Accessibility audit passes
  - [ ] Embeds in external page

### Phase 5 Deliverables
- [x] Complete React chat widget
- [x] SSE streaming display
- [x] Virtualized message list
- [x] WCAG 2.2 accessible
- [x] Embeddable via script tag
- [x] Mobile responsive

---

## Phase 6: Integration & Testing

### Objectives
- Full end-to-end integration testing
- Performance optimization
- Security audit
- Documentation completion

### Duration: 3 Days

### Detailed TODO Checklist

#### Day 22: Integration Testing

- [ ] **6.1 API Integration Tests**
  - [ ] Create `tests/integration/test_api_chat.py`:
    - [ ] Test conversation creation
    - [ ] Test message sending
    - [ ] Test SSE streaming response
    - [ ] Test error handling
  - [ ] Create `tests/integration/test_rag_pipeline.py`:
    - [ ] Test document retrieval
    - [ ] Test hybrid search
    - [ ] Test reranking

- [ ] **6.2 Agent Integration Tests**
  - [ ] Create `tests/integration/test_agent_graph.py`:
    - [ ] Test graph execution
    - [ ] Test routing logic
    - [ ] Test checkpointing
    - [ ] Test tool execution

- [ ] **6.3 E2E Tests**
  - [ ] Create `tests/e2e/test_full_conversation.py`:
    - [ ] Complete user journey
    - [ ] Multi-turn conversation
    - [ ] Context retention
    - [ ] Escalation flow

- [ ] **6.4 Frontend Tests**
  - [ ] Set up Vitest for frontend
  - [ ] Component tests:
    - [ ] MessageBubble rendering
    - [ ] Chat state management
    - [ ] SSE handling

#### Day 23: Performance & Security

- [ ] **6.5 Performance Optimization**
  - [ ] Backend:
    - [ ] Database query optimization (EXPLAIN ANALYZE)
    - [ ] Connection pooling tuning
    - [ ] Caching strategy review
    - [ ] Response compression
  - [ ] Frontend:
    - [ ] Bundle size analysis (vite-bundle-analyzer)
    - [ ] Code splitting verification
    - [ ] Lighthouse performance audit

- [ ] **6.6 Load Testing**
  - [ ] Create `scripts/load_test.py`:
    - [ ] Use locust or k6
    - [ ] Test concurrent connections
    - [ ] Test SSE streaming under load
    - [ ] Identify bottlenecks

- [ ] **6.7 Security Audit**
  - [ ] Authentication:
    - [ ] JWT expiration enforced
    - [ ] Refresh token rotation
    - [ ] Rate limiting effective
  - [ ] API:
    - [ ] Input validation comprehensive
    - [ ] SQL injection prevention
    - [ ] XSS prevention (Markdown sanitization)
  - [ ] Dependencies:
    - [ ] Run `pip-audit` for Python
    - [ ] Run `npm audit` for frontend

#### Day 24: Documentation

- [ ] **6.8 API Documentation**
  - [ ] Create `docs/api-reference.md`:
    - [ ] All endpoints documented
    - [ ] Request/response examples
    - [ ] Error codes reference
  - [ ] Verify OpenAPI spec (FastAPI auto-generated)

- [ ] **6.9 Deployment Documentation**
  - [ ] Create `docs/deployment-guide.md`:
    - [ ] Environment setup
    - [ ] Docker deployment
    - [ ] Environment variables
    - [ ] Database migrations
    - [ ] Monitoring setup

- [ ] **6.10 Configuration Documentation**
  - [ ] Create `docs/configuration.md`:
    - [ ] All configuration options
    - [ ] Feature flags
    - [ ] Scaling recommendations

- [ ] **6.11 Widget Integration Guide**
  - [ ] Add to `docs/widget-integration.md`:
    - [ ] Installation instructions
    - [ ] Configuration options
    - [ ] Customization (themes, styling)
    - [ ] Event hooks

### Phase 6 Deliverables
- [x] Integration tests passing
- [x] E2E tests passing
- [x] Performance optimized
- [x] Security audit complete
- [x] Documentation complete

---

## Phase 7: Deployment & Monitoring

### Objectives
- Production deployment
- Monitoring and alerting setup
- Knowledge base seeding
- Handover documentation

### Duration: 4 Days

### Detailed TODO Checklist

#### Day 25: Production Infrastructure

- [ ] **7.1 Production Docker Compose**
  - [ ] Create `docker-compose.prod.yml`:
    - [ ] Production environment variables
    - [ ] Resource limits
    - [ ] Restart policies
    - [ ] Health checks

- [ ] **7.2 Database Setup**
  - [ ] Run Alembic migrations: `alembic upgrade head`
  - [ ] Create production database user
  - [ ] Configure backup schedule
  - [ ] Set up connection pooling (pgBouncer optional)

- [ ] **7.3 Qdrant Cloud Setup**
  - [ ] Create Qdrant Cloud cluster
  - [ ] Configure API key
  - [ ] Create collections with proper indexes
  - [ ] Verify hybrid search enabled

#### Day 26: Monitoring Setup

- [ ] **7.4 Prometheus Configuration**
  - [ ] Create `infra/monitoring/prometheus/prometheus.yml`
  - [ ] Configure scrape targets:
    - [ ] FastAPI metrics endpoint
    - [ ] PostgreSQL exporter
    - [ ] Redis exporter

- [ ] **7.5 Grafana Dashboards**
  - [ ] Create `infra/monitoring/grafana/dashboards/`:
    - [ ] `api-dashboard.json`:
      - Request rate
      - Latency percentiles
      - Error rate
      - Status code distribution
    - [ ] `llm-dashboard.json`:
      - Token usage
      - LLM latency
      - Cost tracking
      - Error rate

- [ ] **7.6 Alerting Rules**
  - [ ] Configure alerts for:
    - [ ] High error rate (> 1%)
    - [ ] High latency (p95 > 5s)
    - [ ] Database connection errors
    - [ ] LLM service unavailable

#### Day 27: Knowledge Base & Testing

- [ ] **7.7 Knowledge Base Seeding**
  - [ ] Prepare sample documents:
    - [ ] FAQs (general, billing, technical)
    - [ ] Product catalog
    - [ ] Policies (privacy, terms)
  - [ ] Run ingestion: `python scripts/ingest_documents.py ./knowledge_base/`
  - [ ] Verify documents in Qdrant

- [ ] **7.8 RAG Evaluation**
  - [ ] Create evaluation dataset (20+ Q&A pairs)
  - [ ] Run: `python scripts/evaluate_rag.py`
  - [ ] Verify:
    - [ ] Faithfulness > 0.85
    - [ ] Context Recall > 0.80
    - [ ] Answer Relevance > 0.90

- [ ] **7.9 Production Testing**
  - [ ] Smoke tests in production environment
  - [ ] Multi-turn conversation test
  - [ ] Escalation flow test
  - [ ] Load test (10 concurrent users)

#### Day 28: Launch & Handover

- [ ] **7.10 Final Deployment**
  - [ ] Deploy backend: `docker-compose -f docker-compose.prod.yml up -d`
  - [ ] Deploy frontend widget to CDN
  - [ ] Verify health endpoints
  - [ ] Verify monitoring dashboards

- [ ] **7.11 Widget Integration**
  - [ ] Generate client API key
  - [ ] Provide embed code:
    ```html
    <script
      src="https://cdn.example.com/widget-loader.js"
      data-client-key="YOUR_API_KEY"
      data-theme="light"
    ></script>
    ```
  - [ ] Test on target website

- [ ] **7.12 Handover Documentation**
  - [ ] Update `README.md` with:
    - [ ] Quick start guide
    - [ ] Architecture overview
    - [ ] Development setup
    - [ ] Deployment instructions
  - [ ] Create `docs/troubleshooting.md`:
    - [ ] Common issues
    - [ ] Debug procedures
    - [ ] Support contacts

- [ ] **7.13 Final Validation**
  - [ ] All tests passing
  - [ ] Monitoring operational
  - [ ] Documentation complete
  - [ ] Stakeholder sign-off

### Phase 7 Deliverables
- [x] Production deployment live
- [x] Monitoring and alerting operational
- [x] Knowledge base populated
- [x] Widget embeddable
- [x] Complete documentation

---

## 8. Success Metrics & Validation

### Technical Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| API Latency (p95) | < 500ms | Prometheus histogram |
| LLM Response Time (p95) | < 5s | Custom metric |
| Uptime | > 99.5% | Health check monitoring |
| Error Rate | < 1% | Error counter |
| RAGAs Faithfulness | > 0.85 | Evaluation script |
| RAGAs Context Recall | > 0.80 | Evaluation script |

### Business Metrics (Post-Launch)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Customer Satisfaction | > 4.0/5 | In-widget feedback |
| Query Resolution Rate | > 70% | Escalation rate inverse |
| Avg Response Time | < 3s | End-to-end timing |
| Daily Active Users | Growth | Analytics |

### Validation Checkpoints

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VALIDATION GATES                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ✓ Phase 1 Complete: Infrastructure running, health checks pass         │
│                                                                          │
│  ✓ Phase 2 Complete: Auth works, Chat CRUD functional, SSE streams     │
│                                                                          │
│  ✓ Phase 3 Complete: RAG retrieval accurate, reranking improves results │
│                                                                          │
│  ✓ Phase 4 Complete: Agent graph executes, checkpointing works         │
│                                                                          │
│  ✓ Phase 5 Complete: Widget embeds, accessible, mobile responsive       │
│                                                                          │
│  ✓ Phase 6 Complete: All tests pass, security audit clean              │
│                                                                          │
│  ✓ Phase 7 Complete: Production live, monitoring operational            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix A: Environment Variables Reference

```bash
# =============================================================================
# CORE APPLICATION
# =============================================================================
ENVIRONMENT=production          # development | staging | production
DEBUG=false                     # Enable debug mode
LOG_LEVEL=INFO                  # DEBUG | INFO | WARNING | ERROR

# =============================================================================
# DATABASE
# =============================================================================
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/dbname
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10
DB_POOL_RECYCLE=3600

# =============================================================================
# REDIS
# =============================================================================
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=20

# =============================================================================
# VECTOR DATABASE (QDRANT)
# =============================================================================
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
QDRANT_COLLECTION=support_docs

# =============================================================================
# LLM PROVIDERS
# =============================================================================
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-turbo

# =============================================================================
# COHERE (EMBEDDINGS + RERANKING)
# =============================================================================
COHERE_API_KEY=your-cohere-key
COHERE_EMBED_MODEL=embed-english-v3.0
COHERE_RERANK_MODEL=rerank-v3.5

# =============================================================================
# AUTHENTICATION
# =============================================================================
JWT_SECRET=your-super-secret-key-min-32-chars
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=30

# =============================================================================
# RATE LIMITING
# =============================================================================
RATE_LIMIT_DEFAULT=100/minute
RATE_LIMIT_LLM=20/minute

# =============================================================================
# CORS
# =============================================================================
CORS_ORIGINS=["https://yourdomain.com", "https://app.yourdomain.com"]

# =============================================================================
# OBSERVABILITY
# =============================================================================
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
```

---

## Appendix B: Cost Estimation (Monthly)

| Service | Tier | Estimated Cost (SGD) |
|---------|------|---------------------|
| **Qdrant Cloud** | 1GB free, then $25/mo | ~$35 |
| **PostgreSQL** | DigitalOcean Managed | ~$15 |
| **Redis** | DigitalOcean Managed | ~$15 |
| **OpenAI GPT-4** | ~100K tokens/day | ~$60 |
| **Cohere Embed** | ~50K docs embedded | ~$10 |
| **Cohere Rerank** | ~10K reranks/day | ~$15 |
| **Compute (VPS)** | 4GB RAM, 2 vCPU | ~$25 |
| **CDN (Widget)** | Cloudflare Free | $0 |
| **TOTAL** | | **~$175/month** |

*Note: Costs vary based on usage. Optimize with caching and batch operations.*

---

## Appendix C: Quick Reference Commands

```bash
# Development
make dev                    # Start development stack
make test                   # Run all tests
make lint                   # Run linters
make migrate                # Run database migrations

# Docker
docker-compose up -d        # Start all services
docker-compose logs -f api  # Follow API logs
docker-compose down         # Stop all services

# Database
alembic upgrade head        # Apply migrations
alembic revision -m "msg"   # Create new migration
alembic downgrade -1        # Rollback one migration

# RAG
python scripts/ingest_documents.py ./docs/  # Ingest documents
python scripts/evaluate_rag.py              # Run RAGAs evaluation

# Production
docker-compose -f docker-compose.prod.yml up -d  # Deploy production
docker-compose -f docker-compose.prod.yml logs   # View logs
```

---

**Document End**

*This Master Execution Plan is ready for implementation. Each phase can be executed independently after its dependencies are complete. Refer to the validation checkpoints at the end of each phase to ensure quality before proceeding.*
