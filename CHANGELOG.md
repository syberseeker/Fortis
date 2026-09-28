# Changelog

All notable changes to Fortis, traced from the initial commit (2026-09-21)
to the current build. Version numbers are retroactive labels for the dated
commit clusters; no git tags are cut yet. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/).

## [0.4.1] — 2026-09-25

### Fixed
- **`UnboundLocalError` crashed every llama-backend chat call.** The
  executor closure in `engine/llama_engine.py::chat` assigned to
  `max_tokens`, which made the name closure-local; the first read
  (`if max_tokens is None`) raised before any assignment. Latent since
  0.3.0's hardware-aware inference change, surfaced by chat-driven report
  generation. The default (2048 plain / 3072 JSON) is now resolved outside
  the closure; explicit `max_tokens` passes through untouched. Added a
  six-test regression file covering both defaults, explicit passthrough
  (including `max_tokens=0`), JSON-fence extraction, and reasoning-fence
  stripping. Offline suite: 239 tests passing.

## [0.4.0] — 2026-09-25

### Added
- **Chat-driven report generation.** Asking "generate a docx/pptx/pdf
  report" in chat now renders a real file into `%APPDATA%\Fortis\reports`
  and replies with a download link (which opens in the system browser from
  the desktop shell). This rewires the previously dead path: the chat UI
  posts to `/chat/stream`, which never invoked report rendering — the only
  working route was `/report/generate`, which the UI never called.
- **Shared report pipeline** (`core/report_intent.py`): one source of truth
  for intent detection, rendering into the reports folder, the
  placeholder-data gate message, and reply formatting. `/report/generate`,
  `/chat`, `/chat/stream`, the orchestrator, and the new slash command all
  share a single builder (`generate_report_result`).
- **`/report [docx|pptx|pdf]` slash command** for keyboard-only report
  generation against the active engagement.
- **Chat history persistence.** New `chat_messages` SQLite table scoped to
  the engagement (newest 500 turns, auto-pruned, indexed, cascade-deleted
  with the engagement). Both sides of every turn are stored on `/chat` and
  `/chat/stream` — normal chat, structured output, report replies, and the
  stub-gate message included. The UI restores the transcript on app start
  and engagement switch; a top-bar **Clear chat** button deletes an
  engagement's transcript behind a confirm dialog.
- **`GET /chat/history/{engagement_id}`** and
  **`DELETE /chat/history/{engagement_id}?confirm=true`** endpoints.
- **LLM context guard:** `build_chat_messages` clamps whatever history it
  receives to the most recent 16 messages, so a long stored archive can
  never blow the model's context window.
- 13 regression tests for chat-driven export and 15 for history
  persistence (suite grew 205 → 233).

### Fixed
- Report error surfaces now carry proper status codes (400 invalid format,
  404 unknown engagement, 409 stub gate) consistently across the direct
  route, `/api/report/save`, and chat paths. In chat surfaces the stub gate
  replies conversationally instead of throwing an HTTP error at the UI.

### Changed
- README documents the new behaviors; tightened the report-intent detector
  so "make a summary of this document" stays a chat request rather than
  triggering a report export (mirrored in the UI's progress-bar heuristic).

## [0.3.1] — 2026-09-25

### Fixed
- `setup.ps1` no longer fails when uninstalling a broken `llama_cpp`
  install that is not present.

## [0.3.0] — 2026-09-24

### Added
- **Report progress tracking** (`core/progress.py`): thread-safe
  stage/done/total snapshots published by the analysis pipeline, polled by
  the UI at 800 ms so long map-reduce runs on CPU show "Analyzing document
  batches / Merging findings / Building document" instead of appearing hung.
- **Placeholder-report gate:** with no model loaded, report generation is
  refused (HTTP 409) unless explicitly acknowledged, so stub output can
  never be delivered to a client accidentally.
- Hardware status surfaced in the chat UI (GPU offload line, CPU-mode
  hint on report progress).

### Changed
- **Hardware-aware inference:** partial GPU offload via an OOM retry
  ladder, CPU thread/batch tuning (physical cores preferred over
  hyperthreads), slow-backend map-batch doubling for report pipelines.

### Hardened
- API rejects unpaired UTF-16 surrogates in chat messages, engagement
  names, and upload filenames (a JSON `\ud800` escape previously 500'd the
  request); valid text is byte-identical. Adversarial-input coverage:
  emoji/CJK/RTL floods, control characters, 100k single-line messages.

## [0.2.0] — 2026-09-22/23

### Fixed
- Fresh-clone setup failures: dependency lists had drifted from what the
  code imports; `setup.ps1` now installs a consistent set.
- llama.cpp engine install fallbacks on fresh clones; a broken
  `llama_cpp` install is uninstalled before wheel fallbacks are tried.
- Error handling in `setup.ps1` centralized through an `Invoke-Native`
  helper so native-command failures are caught reliably.
- Housekeeping: ignore local `.opencode` directory; README refresh.

## [0.1.0] — 2026-09-21

### Added
- **Initial release of Fortis**, the fully local AI cybersecurity advisory
  desktop app: upload a document (policy, config, code, log) and get a
  structured security analysis grounded in NIST CSF / OWASP Top 10 /
  CIS Controls, exportable as DOCX, PPTX, or PDF. No Docker, no cloud
  APIs; all data stays under one local data root.
- **Engagement data model** (`core/store.py`, SQLite): clients,
  engagements, per-user active pointer — documents, chat grounding, and
  reports scope to an engagement rather than a chat window.
- **RAG core** (`core/`): ingestion with per-type text extraction, chunking
  (~500 tokens, 75 overlap, enriched headers), ChromaDB vector store with
  hash-stub offline embeddings for tests, framework seed corpus, and the
  map-reduce analysis pipeline (flat pass under ~5000 tokens, token-bounded
  batches plus a reduce/merge pass above).
- **Report renderers** (`core/reports/`): one `SecurityReport` schema
  rendered by python-docx / python-pptx / ReportLab with severity colors,
  findings tables, and charts — documents are rendered by code, never by
  the LLM.
- **Engine abstraction** (`engine/`): llama-cpp-python on local Qwen3.5
  GGUF tiers (0.8B–9B, auto-suggested by hardware, resumable HF downloads)
  or a deterministic stub backend for offline tests; selection via
  `FORTIS_LLM_BACKEND`.
- **Server + desktop shell:** FastAPI app with engagement/upload/chat/
  report routes and the static streaming chat UI (markdown, Mermaid
  diagrams with strict allowlist, engagement sidebar, persona presets,
  cybersecurity guardrail); pywebview launcher with browser fallback.
- **Tiered RAG upgrade** (same day): `RAG_LEVEL` basic/standard/full —
  hybrid BM25+RRF keyword retrieval, chunk enrichment, heuristic query
  condensation, optional LLM condensation and cross-encoder reranking
  (auto-disabled under 8 GB RAM).
- **Offline test suite** (`desktop/tests/`): no GPU, no network, no model —
  hash-stub embeddings plus monkeypatched LLM.
- Removed the legacy containerized stack (predecessor ran as four Docker
  containers behind an Open WebUI pipeline); the orchestration behaviors
  were ported into in-process calls.

---

## Development arc

- **Day 1 (Sep 21):** born as a desktop rewrite of a Docker-era advisor —
  the container stack was deleted and its pipeline logic ported into
  `core/` as a library, with the llama.cpp engine replacing Ollama.
- **Day 2–3 (Sep 22–23):** setup hardening — getting a fresh clone to a
  working desktop app on one command proved to be the real work.
- **Day 4 (Sep 24):** robustness — hostile-input hardening, hardware-aware
  GPU offload, and the first UX for long report runs (progress bar, stub
  gate).
- **Day 5 (Sep 25):** the two features that make the product coherent —
  reports actually export from chat, and conversations actually persist —
  plus the closure bug fix that un-blocked real inference for chat.
