# Workshop: How to Build Your Own AI Cybersecurity Advisor

> **Builder-level guide.** This document walks through the architecture and
> code. For the end-user
> half-day workshop (no coding), follow `03-syllabus.md` with labs from
> `01-lab-exercises.md` instead.

**Duration:** 3-4 hours (with breaks)
**Level:** Intermediate — assumes basic Python and networking knowledge
**Goal:** Build a fully local, GPU-accelerated cybersecurity advisory tool that ingests documents, maps findings to security frameworks, and generates professional reports.

---

## What You Will Build

A system called **Fortis** — an AI cybersecurity consultant that:

- Accepts uploaded documents (configs, policies, code, logs, architecture diagrams)
- Retrieves relevant context using vector search (RAG)
- Maps findings to NIST CSF 2.0, OWASP Top 10:2025, and CIS Controls v8.1
- Generates consultant-grade reports as DOCX, PPTX, or PDF
- Runs entirely on your local machine — no cloud APIs, no data leaves your network

### Architecture Overview

```
```
┌──────────────────┐      ┌──────────────────────┐      ┌───────────────┐
│  Chat UI         │◄────►│  server/ (FastAPI)   │◄────►│  engine/      │
│  desktop window  │ http │  - orchestration     │      │  llama.cpp    │
│  or browser      │      │  - upload/chat/report│      │  Qwen3.5 GGUF │
└──────────────────┘      └──────────────────────┘      └───────────────┘
                                   │
                          ┌────────▼─────────┐
                          │  core/           │
                          │  - ChromaDB      │
                          │  - BM25 + rerank │
                          │  - RAG + persona │
                          │  - report output │
                          └──────────────────┘
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 6+ cores |
| RAM | 16 GB | 32 GB |
| GPU | 4 GB VRAM (RTX 3050) | 8+ GB VRAM |
| Disk | 20 GB free | 50 GB free |

---

## Workshop Agenda

### Module 1: Foundations (45 min)
1. Why local AI for cybersecurity? (privacy, compliance, air-gapped environments)
2. RAG architecture explained — why not just use ChatGPT?
3. Component tour: engine, FastAPI, ChromaDB, chat UI

### Module 2: Infrastructure Setup (30 min)
4. Local stack setup and launch (setup.ps1 / setup.sh)
5. Pulling the LLM and configuring the embedding model
6. Verifying the stack is running

### Module 3: RAG Core Deep Dive
7. **File Ingestion Pipeline** — parsing PDFs, DOCX, configs, code
8. **Embeddings & Vector Store** — how documents become searchable
9. **RAG Retrieval** — building context for the LLM
10. **The LLM Engine** — talking to llama.cpp
11. **Analysis Engine** — flat vs. map-reduce for large documents
12. **Report Generation** — structured output into DOCX/PPTX/PDF

### Module 4: Frontend & Integration (30 min)
13. Chat UI and persona injection
14. Orchestration — connecting UI to the RAG core
15. Engagement management (persistent client sessions)

### Module 5: Hands-On Labs (45 min)
16. Lab 1: Upload and analyze a sample firewall config
17. Lab 2: Generate a security report
18. Lab 3: Extend the system with your own framework

### Module 6: Advanced Topics (30 min)
19. Map-reduce analysis for large documents
20. Custom framework ingestion (loading official PDFs)
21. Testing without a GPU (offline test suite)
22. Deployment considerations and hardening

---

## Module 1: Foundations

### Why Local AI for Cybersecurity?

Security data is sensitive. Uploading a firewall configuration, IAM policy, or source code to a cloud AI service raises serious concerns:

- **Data sovereignty** — client data cannot leave their network
- **Compliance** — SOC 2, HIPAA, ITAR may prohibit sending data to third-party APIs
- **Air-gapped environments** — military, government, critical infrastructure
- **IP protection** — proprietary code and configurations

A local RAG system gives you the power of LLM-assisted analysis while keeping all data on your own hardware.

### What is RAG?

**Retrieval-Augmented Generation** solves the LLM's biggest limitation: it doesn't know about your specific documents.

```
Without RAG:                     With RAG:
                                 
User: "Review my firewall"      User: "Review my firewall"
   ↓                                ↓
LLM: "I don't have your         Retriever: finds relevant chunks
     firewall config"            from uploaded firewall config
   ↓                                ↓
Response: generic advice        LLM: sees your actual config
                                   + framework references
                                ↓
                                Response: specific, evidence-based
```

The flow:
1. **Ingest** — parse documents, split into chunks, embed into vectors
2. **Store** — keep vectors in a database (ChromaDB)
3. **Retrieve** — when user asks a question, find the most relevant chunks
4. **Generate** — give the LLM the retrieved context + a persona prompt
5. **Respond** — grounded, cited, framework-mapped answer

### Component Tour

| Component | Role | Technology |
|-----------|------|------------|
| LLM | Text generation | llama.cpp + Qwen3.5 GGUF (local) |
| Embeddings | Convert text to vectors | bge-small-en-v1.5 (CPU) |
| Vector Store | Store and search vectors | ChromaDB (persistent, local) |
| RAG core | Ingestion, RAG, reports | core/ (Python) |
| Frontend | Chat interface, persona | FastAPI static UI + pywebview |
| Orchestration | Routing, guardrails, reports | server/orchestration.py |

---

## Module 2: Local Setup

### How the Stack Runs

Everything is a plain Python process — no Docker:

```
core/     RAG library: ingestion → ChromaDB → BM25/rerank → context
engine/   llama.cpp runtime: local Qwen3.5 GGUF, GPU offload auto-detected
server/   FastAPI app + orchestration + static chat UI
desktop/  pywebview launcher (the "app window")
```

Key design decisions:
- **LLM runs in-process** — llama.cpp inside the Python app, no HTTP hop
- **GPU offload auto-detected** — falls back to CPU on modest hardware
- **Local data only** — ChromaDB, uploads, and SQLite store live in `data/`

### Step-by-Step Setup

```powershell
# 1. One-command setup + launch (creates .venv, installs deps, downloads model)
powershell -ExecutionPolicy Bypass -File setup.ps1

# 2. Verify — the desktop window opens, or headless:
python -m server.app --port 8757
curl http://127.0.0.1:8757/health
# Should return: {"status": "ok", "frameworks_loaded": ["NIST_CSF", ...]}
```

### Understanding the .env Configuration

```bash
FORTIS_LLM_BACKEND=auto            # llama | stub | auto
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5  # Embedding model (runs on CPU)
MAX_CONTEXT_TOKENS=6000            # Max tokens sent to LLM per request
RAG_LEVEL=standard                 # basic | standard | full retrieval tiers
```

## Module 3: RAG Core Deep Dive

### 3.1 File Ingestion Pipeline

**File:** `core/ingestion.py`

The ingestion pipeline converts any uploaded file into plain text, then chunks it for embedding.

```
Upload → Extract Text → Chunk Text → Embed → Store in ChromaDB
```

**Supported formats:**
| Format | Parser | Notes |
|--------|--------|-------|
| PDF | PyMuPDF (fitz) | Extracts per-page with page markers |
| DOCX | python-docx | Paragraphs + table rows |
| PPTX | python-pptx | Slides with text frames + tables |
| XLSX | openpyxl | Sheet-by-sheet with row pipes |
| CSV | stdlib csv | Pipe-delimited rows |
| Code/Config | plaintext | .py, .js, .yaml, .json, .conf, etc. |
| TXT/MD | plaintext | Direct read |

**Chunking strategy** (token-aware sliding window):
- Default chunk size: 500 tokens (~2000 characters)
- Overlap: 75 tokens (~300 characters)
- Splits on paragraph boundaries to preserve code/config structure
- Oversized paragraphs get hard-split by character width

**Key code walkthrough:**

```python
# ingestion.py — chunk_text()
def chunk_text(text, chunk_size_tokens=500, overlap_tokens=75):
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks, current, current_tokens = [], [], 0
    
    for para in paragraphs:
        para_tokens = count_tokens(para)
        
        if current_tokens + para_tokens > chunk_size_tokens:
            # Flush current chunk
            chunks.append("\n\n".join(current).strip())
            # Start new chunk with overlap from tail of previous
            current = [chunks[-1][-overlap_chars:]] if chunks else []
            current_tokens = count_tokens(overlap_text) if overlap_text else 0
        
        current.append(para)
        current_tokens += para_tokens
    
    return [c for c in chunks if c.strip()]
```

**Why this matters:** Poor chunking breaks RAG quality. If you cut a firewall rule mid-sentence, the retriever finds a meaningless fragment. Paragraph-aware chunking preserves semantic units.

### 3.2 Embeddings & Vector Store

**Files:** `core/embeddings.py`, `core/vectorstore.py`

**Embeddings** convert text into numerical vectors (384-dimensional for bge-small-en-v1.5) that capture semantic meaning. Similar concepts get similar vectors.

```
"password stored in plaintext"  → [0.12, -0.34, 0.56, ...]  (384 floats)
"hardcoded credential in config" → [0.11, -0.33, 0.55, ...]  (similar!)
"the weather is nice today"      → [0.89, 0.12, -0.76, ...]  (different)
```

**ChromaDB** stores these vectors with metadata and supports similarity search:

```python
# Two collections:
_user_docs     # Uploaded client documents (scoped by engagement_id)
_frameworks    # NIST CSF, OWASP, CIS Controls, MITRE ATT&CK
```

**Query flow:**
```python
def query_user_documents(engagement_id, query, top_k=6):
    query_vector = embed_query(query)          # Embed the question
    results = _user_docs.query(                # Vector similarity search
        query_embeddings=[query_vector],
        n_results=top_k,
        where={"engagement_id": engagement_id} # Scoped to this engagement
    )
    return results  # Top 6 most relevant chunks
```

**Two collections, two purposes:**
- `user_documents` — your client's uploaded files, scoped per engagement
- `security_frameworks` — always-available reference controls (NIST, OWASP, CIS)

### 3.3 RAG Retrieval — Building Context

**File:** `core/rag.py`

The RAG system builds a context block from two retrieval sources:

```python
def build_context_block(engagement_id, query):
    # 1. Retrieve relevant chunks from user's uploaded documents
    user_hits = vectorstore.query_user_documents(engagement_id, query, top_k=6)
    
    # 2. Retrieve relevant security framework controls
    framework_hits = vectorstore.query_frameworks(query, top_k=4)
    
    # 3. Assemble into a structured context block
    parts = []
    parts.append("=== Retrieved excerpts from uploaded documents ===")
    for hit in user_hits:
        parts.append(f"[{filename} — chunk {index}]\n{text}")
    
    parts.append("\n=== Relevant framework controls ===")
    for hit in framework_hits:
        parts.append(f"[{framework} {control_id}] {text}")
    
    return "\n\n".join(parts)
```

**The system persona** (in `rag.py`) defines the consultant behavior:

```
You are Fortis, a senior cybersecurity consultant AI.
- Ground every finding in retrieved document excerpts and framework references
- Cite which uploaded file a finding came from
- Map findings to NIST CSF, OWASP, or CIS Controls when applicable
- Rate severity (Critical/High/Medium/Low/Informational)
- Give concrete, actionable remediation
- Refuse non-cybersecurity questions
```

### 3.4 The LLM Engine

**File:** `engine/llama_engine.py`

A local llama.cpp backend running a Qwen3.5 GGUF in-process — no Ollama,
no HTTP hop. The chat template is applied by llama.cpp via
`create_chat_completion`;

```python
# engine/llama_engine.py — chat entry point (simplified)
def chat(messages, temperature=0.2, max_tokens=None):
    model = _get_or_load_model()   # lazy-load inside a lock, with warm-up
    out = model.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=_STOP_TOKENS,
    )
    return out["choices"][0]["message"]["content"]
```

**Two modes:**
- `chat()` — single completion, used for report generation
- `chat_stream()` — generator, used for streaming chat responses to the UI

### 3.5 Analysis Engine — Flat vs. Map-Reduce

**File:** `core/analysis.py`

This is the most sophisticated component. Report generation uses two strategies:

#### Flat Path (small documents)
When total uploaded content fits in one context window (< ~5000 tokens):

```
All chunks + Framework references → Single LLM call → SecurityReport JSON
```

Simple, fast, cheap.

#### Map-Reduce Path (large documents)
When content exceeds the threshold — a 200-page policy document won't be silently truncated:

```
┌─────────────────────────────────────────┐
│ MAP PHASE                               │
│                                         │
│ Document chunks → Batch 1 → LLM → Findings 1
│                → Batch 2 → LLM → Findings 2
│                → Batch 3 → LLM → Findings 3
│                ...                      │
│ (each batch analyzed independently)     │
└────────────────────┬────────────────────┘
                     ↓
┌─────────────────────────────────────────┐
│ REDUCE PHASE                            │
│                                         │
│ All findings + Framework context        │
│        → Single LLM call                │
│        → Dedupe, map frameworks,        │
│          write executive summary        │
│        → SecurityReport JSON            │
└─────────────────────────────────────────┘
```

**Key design decision:** Both paths return the same `SecurityReport` schema. The report router doesn't know or care which path ran.

```python
# analysis.py — the decision point
async def generate_security_report(engagement_id, focus_instructions=""):
    total_tokens = sum(len(chunk) for chunks in all_chunks)
    
    if total_tokens <= settings.hierarchical_threshold_tokens:
        return await _generate_flat(engagement_id, focus_instructions)
    else:
        return await _generate_map_reduce(engagement_id, focus_instructions)
```

### 3.6 Report Generation

**Files:** `core/reports/`

The structured `SecurityReport` Pydantic model is rendered into three formats:

| Format | Library | Output |
|--------|---------|--------|
| DOCX | python-docx | Color-coded severity, heading structure, bullet lists |
| PPTX | python-pptx | Title slide, executive summary, one slide per finding |
| PDF | ReportLab | Page-formatted with colors, labels, lists |

**The schema** (`reports/schema.py`):
```python
class Finding(BaseModel):
    title: str
    severity: str        # Critical | High | Medium | Low | Informational
    description: str
    evidence: str        # Quoted from source document
    source_file: str     # Which uploaded file
    framework: str       # NIST_CSF | OWASP_TOP10 | CIS_CONTROLS
    control_id: str      # e.g., PR.AA, A01:2025, CIS-6
    remediation: str     # Actionable fix

class SecurityReport(BaseModel):
    title: str
    client_context: str
    scope: str
    executive_summary: str
    findings: List[Finding]
    overall_risk_rating: str
    recommendations_summary: List[str]
```

**Important:** Reports are rendered by code, not by the LLM. The LLM produces the data; the Python renderers produce the document. This guarantees structurally valid output every time.

---

## Module 4: UI & Integration

### 4.1 The Chat UI

**File:** `server/static/index.html`

The chat UI is a static page served by FastAPI (or wrapped by pywebview in
the desktop app). Streaming responses, markdown tables, and the engagement
sidebar are rendered client-side; vendor libraries are bundled locally.

### 4.2 Orchestration — Connecting UI to the RAG Core

**File:** `server/orchestration.py`

The orchestration layer is the glue between the chat route and the RAG core:

1. **Intercepts** the user message before it reaches the LLM
2. **Routes** engagement commands (`/new-engagement`, `/use`, etc.)
3. **Ingests** attached files into the RAG core
4. **Enforces** the cybersecurity-only guardrail (rejects off-topic queries)
5. **Detects** report generation intent ("generate a docx report")
6. **Condenses** follow-up questions before retrieval
7. **Calls** the core for chat or report generation

```python
# Priority order in orchestration:
# 1. Engagement commands (always allowed)
# 2. Must have active engagement
# 3. Ingest any attached files
# 4. Cybersecurity topic guardrail
# 5. Report generation detection
# 6. Query condensation
# 7. Normal RAG chat
```

### 4.3 Engagement Management

**Files:** `core/store.py`, `core/routers/engagements.py`

Engagements provide persistent scoping — documents and context survive across chat sessions:

```
Client: "Acme Corp"
  └── Engagement: "Q3 2026 Config Review" (active)
        ├── Uploaded: firewall-config.txt, iam-policy.json
        ├── Chat history: [relevant excerpts stored]
        └── Reports: [generated DOCX/PPTX/PDF]
  
  └── Engagement: "Annual Pentest Prep" (closed)
        └── ...
```

**Storage:** SQLite database (`engagements.sqlite3`) with three tables:
- `clients` — organization name, slug-based ID
- `engagements` — name, status, notes, linked to client
- `active_engagement` — per-user pointer to current engagement

**Chat commands** (typed as messages):
| Command | Effect |
|---------|--------|
| `/new-engagement Client :: Name` | Create and activate |
| `/engagements` | List all |
| `/use <id>` | Switch active |
| `/whoami` | Show current + files |
| `/close-engagement` | Mark closed |

---

## Module 5: Hands-On Labs

See `01-lab-exercises.md` for detailed step-by-step instructions.

---

## Module 6: Advanced Topics

### 6.1 Custom Framework Ingestion

The built-in framework corpus is paraphrased. For client-facing reports citing exact standard language:

```bash
python -m core.frameworks.loader NIST_CSF /path/to/NIST.CSWP.29.pdf
```

This replaces the paraphrased seed with verbatim text. Works for any framework name — custom standards are picked up automatically.

### 6.2 Testing Without a GPU

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

The test suite uses:
- `EMBEDDING_BACKEND=hash-stub` — deterministic pseudo-embeddings (no HuggingFace download)
- Monkeypatched LLM — returns canned responses (no model needed)
- Temporary directories for Chroma and SQLite

This validates the wiring (routing, schema, map-reduce triggering) without model quality.

### 6.3 What This Is / Isn't

**IS:**
- Document and config review advisor
- Framework mapping (NIST, OWASP, CIS, MITRE)
- Consultant-style report generation
- Fully local, air-gap capable

**ISN'T:**
- Malware sandbox
- Network scanner
- PCAP deep-inspection tool
- Real-time SIEM

The persona is explicitly instructed to say so if asked to do live/binary analysis.

---

## Key Takeaways

1. **RAG grounds LLM responses in your actual data** — no hallucinated findings
2. **Local deployment keeps sensitive data on your network** — critical for security work
3. **Framework mapping adds credibility** — findings tied to NIST/OWASP/CIS aren't just opinions
4. **Structured output schemas** ensure consistent, renderable reports
5. **Map-reduce handles real-world document sizes** without silent truncation
6. **Engagement persistence** makes this a practical consulting tool, not a demo

---

## Resources

- **NIST CSF 2.0:** https://www.nist.gov/cyberframework
- **OWASP Top 10:2025:** https://owasp.org/Top10/2025/
- **CIS Controls v8.1:** https://www.cisecurity.org/controls
- **MITRE ATT&CK:** https://attack.mitre.org/
- **llama.cpp:** https://github.com/ggml-org/llama.cpp
- **ChromaDB:** https://www.trychroma.com/
