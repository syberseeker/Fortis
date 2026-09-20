# Quick Reference Cheat Sheet

## Architecture at a Glance

```
┌─────────────┐      ┌──────────────────────┐      ┌─────────────┐
│  Open WebUI │◄────►│  Backend (FastAPI)   │◄────►│   Ollama    │
│  :3000      │ pipe │  :8010               │      │  :11434     │
│  chat UI    │      │  ingestion           │      │  LLM+GPU    │
└─────────────┘      │  embeddings (CPU)    │      └─────────────┘
                     │  ChromaDB            │
                     │  RAG retrieval       │
                     │  report generation   │
                     └──────────────────────┘
```

## Data Flow

```
Upload → extract_text() → chunk_text() → embed_texts() → ChromaDB
                                                                      ↓
Query → embed_query() → ChromaDB similarity search → context block
                                                                      ↓
Context + persona + user message → Ollama LLM → response
                                                                      ↓
Report: all chunks → analysis.py (flat/map-reduce) → SecurityReport → DOCX/PPTX/PDF
```

## File Map

```
fortis/
├── docker-compose.yml          # 4 services: ollama, backend, openwebui, pipelines
├── .env                        # OLLAMA_MODEL, EMBEDDING_MODEL, etc.
│
├── backend/
│   ├── Dockerfile              # Python 3.11-slim + system deps
│   ├── requirements.txt        # FastAPI, ChromaDB, sentence-transformers, etc.
│   └── app/
│       ├── main.py             # FastAPI app, startup hooks, /health endpoint
│       ├── config.py           # Settings class (env vars → typed config)
│       ├── ingestion.py        # File parsing (PDF/DOCX/PPTX/XLSX/CSV/code) + chunking
│       ├── embeddings.py       # CPU embedding wrapper (sentence-transformers or hash-stub)
│       ├── vectorstore.py      # ChromaDB wrapper (user_documents + security_frameworks)
│       ├── llm.py              # Ollama HTTP client (chat + streaming)
│       ├── rag.py              # RAG context builder + consultant persona
│       ├── analysis.py         # Report generation (flat + map-reduce)
│       ├── token_utils.py      # ~4 chars/token heuristic
│       ├── store.py            # SQLite engagement persistence
│       │
│       ├── frameworks/         # Built-in reference corpus
│       │   ├── __init__.py     # seed_all_frameworks()
│       │   ├── loader.py       # Load official PDFs as verbatim reference
│       │   ├── nist_csf.json   # NIST CSF 2.0 (28 controls)
│       │   ├── owasp_top10.json# OWASP Top 10:2025 (10 categories)
│       │   ├── cis_controls.json# CIS Controls v8.1 (18 controls)
│       │   ├── mitre_attack.json# MITRE ATT&CK tactics+techniques
│       │   └── cis_benchmarks.json# CIS Benchmarks sub-controls
│       │
│       ├── reports/            # Structured report renderers
│       │   ├── schema.py       # Pydantic models (Finding, SecurityReport)
│       │   ├── docx_report.py  # Word document renderer
│       │   ├── pptx_report.py  # PowerPoint renderer
│       │   └── pdf_report.py   # PDF renderer (ReportLab)
│       │
│       ├── routers/            # FastAPI route handlers
│       │   ├── upload.py       # POST /upload, GET/DELETE engagement files
│       │   ├── chat.py         # POST /chat, POST /chat/stream
│       │   ├── report.py       # POST /report/generate, GET /report/download
│       │   └── engagements.py  # CRUD clients + engagements + active tracking
│       │
│       └── tests/
│           ├── conftest.py     # Env vars + hash-stub backend
│           └── test_integration.py  # Full HTTP API tests (offline)
│
├── openwebui/
│   ├── Dockerfile              # Custom favicon
│   └── favicon.png
│
└── openwebui_pipeline/
    ├── fortis_pipe.py   # Main pipeline (routing, guardrails, reports)
    └── fortis_skill.py  # Finding validator skill
```

## Key API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | System health check |
| POST | `/engagements` | Create client + engagement |
| GET | `/engagements` | List all engagements |
| GET | `/engagements/{id}` | Get engagement detail + files |
| POST | `/engagements/active` | Set active engagement for user |
| GET | `/engagements/active/{user_id}` | Get user's active engagement |
| POST | `/upload` | Upload + ingest a file |
| GET | `/upload/engagement/{id}/files` | List files for engagement |
| POST | `/chat` | RAG chat (synchronous) |
| POST | `/chat/stream` | RAG chat (streaming) |
| POST | `/report/generate` | Generate DOCX/PPTX/PDF report |
| GET | `/report/download/{filename}` | Download generated report |

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_MODEL` | `qwen2.5:3b` | LLM model to use |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `MAX_CONTEXT_TOKENS` | `6000` | Max tokens in LLM context |
| `CHUNK_SIZE_TOKENS` | `500` | Chunk size for ingestion |
| `CHUNK_OVERLAP_TOKENS` | `75` | Overlap between chunks |
| `TOP_K_USER_DOC` | `6` | User doc chunks retrieved per query |
| `TOP_K_FRAMEWORK` | `4` | Framework chunks retrieved per query |
| `HIERARCHICAL_THRESHOLD_TOKENS` | `5000` | Threshold to trigger map-reduce |
| `MAP_BATCH_TOKENS` | `2800` | Max tokens per map batch |
| `OLLAMA_NUM_PARALLEL` | `1` | Concurrent LLM requests |
| `EMBEDDING_BACKEND` | `sentence-transformers` | `hash-stub` for testing |
| `CHROMA_PERSIST_DIR` | `/data/chroma` | ChromaDB data directory |
| `DB_PATH` | `/data/engagements.sqlite3` | SQLite database path |

## Docker Commands

```bash
# Start everything
docker compose up -d --build

# Stop (preserves data)
docker compose down

# Stop and delete ALL data
docker compose down -v

# View logs
docker logs fortis-backend -f
docker logs fortis-ollama -f

# Rebuild after code changes
docker compose build backend && docker compose up -d backend

# Pull/update model
docker exec fortis-ollama ollama pull qwen2.5:3b

# Load official framework document
docker exec fortis-backend \
  python -m app.frameworks.loader NIST_CSF /path/to/document.pdf
```

## Python Import Chain

```
main.py
  ├── routers/upload.py → ingestion.py → vectorstore.py → embeddings.py
  ├── routers/chat.py   → rag.py → vectorstore.py + llm.py
  ├── routers/report.py → analysis.py → vectorstore.py + llm.py + reports/
  ├── routers/engagements.py → store.py (SQLite)
  └── frameworks/__init__.py → vectorstore.seed_framework_chunks()
```

## Severity Scale

| Level | Criteria |
|-------|----------|
| **Critical** | Immediate exploitation risk, full compromise likely |
| **High** | Significant vulnerability, exploitable with moderate effort |
| **Medium** | Notable weakness, requires specific conditions to exploit |
| **Low** | Minor issue, defense-in-depth improvement |
| **Informational** | Observation, best practice suggestion |

## Report Schema (Pydantic)

```python
SecurityReport:
  title: str
  client_context: str
  scope: str
  executive_summary: str
  findings: List[Finding]
  overall_risk_rating: str
  recommendations_summary: List[str]

Finding:
  title: str
  severity: str
  description: str
  evidence: str
  source_file: Optional[str]
  framework: Optional[str]     # NIST_CSF | OWASP_TOP10 | CIS_CONTROLS
  control_id: Optional[str]
  remediation: str
```
