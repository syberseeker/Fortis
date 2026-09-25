# Quick Reference Cheat Sheet

## Architecture at a Glance

```
┌─────────────┐      ┌──────────────────────┐      ┌─────────────────┐
│  Chat UI    │◄────►│  server/ (FastAPI)   │◄────►│  engine/        │
│  desktop or │ http │  - orchestration     │      │  llama.cpp GGUF │
│  browser    │      │  - upload/chat/report│      │  local, GPU/CPU │
└─────────────┘      └──────────────────────┘      └─────────────────┘
                              │
                      ┌───────▼──────────────┐
                      │  core/               │
                      │  - ChromaDB          │
                      │  - BM25 + rerank     │
                      │  - RAG + persona     │
                      │  - report renderers  │
                      └──────────────────────┘
```

## File Map

```
fortis/
├── .env                        # FORTIS_LLM_BACKEND, model path, etc.
│
├── core/                       # RAG backend library
│   ├── config.py               # Settings class (env vars → typed config)
│   ├── ingestion.py            # File parsing (PDF/DOCX/PPTX/XLSX/CSV/code) + chunking
│   ├── embeddings.py           # CPU embedding wrapper (sentence-transformers or hash-stub)
│   ├── vectorstore.py          # ChromaDB wrapper (user_documents + security_frameworks)
│   ├── bm25.py                 # BM25 keyword index + reciprocal rank fusion
│   ├── condense.py             # Follow-up query condensation
│   ├── rerank.py               # Cross-encoder reranking
│   ├── rag.py                  # RAG context builder + consultant persona
│   ├── analysis.py             # Report generation (flat + map-reduce)
│   ├── token_utils.py          # ~4 chars/token heuristic
│   ├── store.py                # SQLite engagement persistence
│   ├── frameworks/             # Built-in reference corpus (NIST/OWASP/CIS/MITRE JSONs + loader)
│   └── reports/                # Structured report renderers (schema, docx, pptx, pdf)
│
├── engine/                     # LLM runtime: llama-cpp-python backend, stub backend, hardware detection
├── server/                     # FastAPI app + orchestration + static chat UI
├── desktop/app.py              # pywebview launcher
├── desktop/tests/              # Offline integration suite
└── training/                   # Workshop guides + lab documents
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
| GET | `/chat/history/{engagement_id}` | Stored chat turns, oldest first (newest 500) |
| DELETE | `/chat/history/{engagement_id}?confirm=true` | Clear an engagement's chat history |
| POST | `/report/generate` | Generate DOCX/PPTX/PDF report |
| GET | `/report/download/{filename}` | Download generated report |
| PATCH | `/engagements/{id}/status` | Set engagement status (active/closed) |
| DELETE | `/engagements/{id}` | Delete engagement + its docs/chat history |

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `FORTIS_LLM_BACKEND` | `auto` | `llama` \| `stub` \| `auto` |
| `FORTIS_MODEL_PATH` | (tier default) | Override GGUF model file path |
| `RAG_LEVEL` | `standard` | `basic` \| `standard` \| `full` retrieval tiers |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `EMBEDDING_BACKEND` | `sentence-transformers` | `hash-stub` for testing |
| `MAX_CONTEXT_TOKENS` | `6000` | Max tokens in LLM context |
| `CHUNK_SIZE_TOKENS` | `500` | Chunk size for ingestion |
| `CHUNK_OVERLAP_TOKENS` | `75` | Overlap between chunks |
| `TOP_K_USER_DOC` | `6` | User doc chunks retrieved per query |
| `TOP_K_FRAMEWORK` | `4` | Framework chunks retrieved per query |
| `HIERARCHICAL_THRESHOLD_TOKENS` | `5000` | Threshold to trigger map-reduce |
| `MAP_BATCH_TOKENS` | `2800` | Max tokens per map batch |
| `RERANK_BACKEND` | `sentence-transformers` | `none` disables reranking |
| `BM25_RRF_K` | `60` | Reciprocal rank fusion constant |
| `CHROMA_PERSIST_DIR` | `%APPDATA%\Fortis\chroma` | ChromaDB data directory |
| `DB_PATH` | `%APPDATA%\Fortis\engagements.sqlite3` | SQLite database (engagements + chat history) |

Data root: `%APPDATA%\Fortis` on Windows, `~/.local/share/Fortis` on
macOS/Linux — override both with `FORTIS_DATA_DIR`.

## Python Import Chain

```
server/app.py
  ├── server/orchestration.py → core/rag.py → core/vectorstore.py + core/bm25.py
  ├── routers/upload.py → core/ingestion.py → core/vectorstore.py → core/embeddings.py
  ├── routers/chat.py → core/report_intent.py + routers/report.py + core/store.py
  ├── routers/report.py → core/analysis.py → core/vectorstore.py + core/reports/
  ├── routers/engagements.py → core/store.py (SQLite)
  └── core/frameworks/__init__.py → vectorstore.seed_framework_chunks()
```

## Slash Commands

| Command | Effect |
|---------|--------|
| `/new-engagement Client :: Name` | Create and activate an engagement |
| `/engagements` | List all engagements |
| `/use <id>` | Switch the active engagement |
| `/whoami` | Show current engagement + files |
| `/close-engagement` | Mark the active engagement closed |
| `/report [docx\|pptx\|pdf]` | Generate a report for the active engagement (defaults to docx) |

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
  diagram: Optional[str]       # Mermaid source, when present

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
