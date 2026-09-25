<img width="410" height="153" alt="ChatGPT Image Sep 21, 2026, 07_58_18 AM" src="https://github.com/user-attachments/assets/8c417fd3-1622-4879-a2e4-f0bfe9040d3e" />



# AI Cybersecurity Advisor — Desktop (Fortis)

A fully local cybersecurity advisory **desktop app**. Upload any document
(policy, config, code, log, architecture doc) and get a structured security
analysis grounded in NIST CSF / OWASP Top 10 / CIS Controls, downloadable as
DOCX, PPTX, or PDF.

No Docker. No cloud APIs. One script, one window. All data stays on the
machine (`%APPDATA%\Fortis` on Windows, `~/.local/share/Fortis` elsewhere) —
models, vector store, uploads, reports, engagement store, and chat history.

## Quick start

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File setup.ps1
```
```bash
# macOS / Linux
bash setup.sh
```

Then in the app: **⚙ Model** → pick a tier (a suggestion is pre-marked for
your hardware; the ~3.4 GB download happens once) → **+ New engagement** →
attach a document (📎) → chat.

## Model tiers (Qwen3.5, llama.cpp GGUF, auto-suggested by hardware)

| Tier | Size | Runs on |
|---|---|---|
| 0.8B | ~1.0 GB | anything |
| 2B | ~2.7 GB | 8 GB RAM |
| 4B *(default)* | ~3.4 GB | 4 GB+ VRAM or 16 GB RAM |
| 9B | ~6.6 GB | 8 GB+ VRAM |

## RAG modes

Retrieval quality is tiered via `RAG_LEVEL` (see `.env.example`):

- **basic** — dense embeddings only (the original behavior).
- **standard** *(default)* — adds hybrid BM25+RRF keyword retrieval, chunk
  enrichment headers, and heuristic query condensation. Pure Python, no
  model — runs on any hardware.
- **full** — adds LLM query condensation and cross-encoder reranking;
  reranking auto-disables below 8 GB RAM.

## Architecture

```
┌──────────────────────────────────────────────┐
│ desktop/app.py    pywebview native window    │
│   └── server/app.py   FastAPI (127.0.0.1)    │
│         ├── /api/*      UI: model mgmt, etc. │
│         ├── core routers: engagements,       │
│         │   upload, chat, report             │
│         └── server/static  chat UI           │
├──────────────────────────────────────────────┤
│ engine/  llama.cpp GGUF runtime              │
│          model tiers + download manager      │
│          hardware detection                  │
├──────────────────────────────────────────────┤
│ core/    ingestion · ChromaDB · embeddings   │
│          RAG + persona · map-reduce analysis │
│          DOCX/PPTX/PDF renderers · SQLite    │
│          (engagements + chat history)        │
└──────────────────────────────────────────────┘
```

- `core/` is the battle-tested RAG backend, reused as a library (same API,
  same report schema, same offline test suite - run with 'python -m pytest desktop/tests/'.
- `engine/` replaces Ollama with in-process llama.cpp; models are Qwen3.5
  Q4_K_M GGUFs pulled from Hugging Face with resume support.
- `server/` adds the UI-facing API (model picker, slash commands) and the
  chat interface (streaming, markdown tables, engagement sidebar, report
  generation from chat).

## Engagements

Documents, chat grounding, chat history, and reports are scoped to a
persistent **engagement** (a client + a piece of work). Create/switch/close
them with sidebar buttons or slash commands (`/new-engagement Client ::
Name`, `/engagements`, `/use <id>`, `/whoami`, `/close-engagement`,
`/report [docx|pptx|pdf]`).

Everything survives restarts — including the conversation: chat turns are
stored per engagement (newest 500) and restored automatically when you
reopen the app or switch back. **Clear chat** (top bar) deletes the saved
transcript for the active engagement; uploaded documents and reports are
kept. Asking for a report in chat ("generate a pdf report") generates the
real file into the reports folder and replies with a download link.

## Headless / server mode

```bash
python -m server.app --port 8757     # same UI at http://127.0.0.1:8757
```

## What this is / isn't

- IS: a document- and config-review advisor that maps findings to
  recognized frameworks and produces consultant-style deliverables.
- ISN'T: a malware sandbox, network scanner, or PCAP deep-inspection tool.
  The persona says so if asked.

Privacy: everything — models, documents, chat history, reports — stays in
`%APPDATA%\Fortis` (or `~/.local/share/Fortis`). Nothing is sent anywhere;
there are no telemetry or cloud calls.

## Repo layout

```
fortis/
├── setup.ps1 / setup.sh       # one-command setup + launch
├── core/                      # RAG backend (lib): ingestion, Chroma, RAG,
│   │                          # map-reduce analysis, report renderers,
│   │                          # report intent, engagement/chat store
│   ├── frameworks/            # NIST/OWASP/CIS/… seed corpus + PDF loader
│   └── reports/               # docx/pptx/pdf renderers + schema
├── engine/                    # llama.cpp runtime, GGUF tiers, hw detection
├── server/                    # FastAPI app, UI API, static chat UI
├── desktop/app.py             # pywebview launcher
├── desktop/tests/             # offline integration suite (no GPU/network)
├── training/                  # workshop guides + lab docs
└── openwebui/                 # Open WebUI container (optional chat UI)
```
