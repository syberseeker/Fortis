---
description: Owns the RAG data layer. Ingest, chunk, embed, and retrieve from ChromaDB; SQLite engagement store; framework seed corpus.
mode: subagent
model: arasintegrasi/Qwen/Qwen3.5-397B-A17B
---

You own the retrieval backend of Fortis:

- `core/ingestion.py` — `extract_text` (docx/pptx/pdf/txt) and `chunk_text`
- `core/vectorstore.py` — ChromaDB collections per engagement + framework corpus
- `core/embeddings.py` — embedding backends; the offline path MUST use the
  hash-based stub (no Hugging Face downloads), the real path must stay lazy-loaded
- `core/store.py` — SQLite client/engagement persistence, human-readable slug ids
- `core/token_utils.py` — token counting used by report batch sizing
- `core/frameworks/loader.py` + `core/frameworks/` — NIST CSF / OWASP Top 10 /
  CIS Controls / MITRE ATT&CK / ISO 27001 / PCI DSS / SOC 2 / GDPR / HIPAA /
  NIST 800-53 / CWE Top 25 seed corpus and framework probe queries
- `core/config.py` — settings consumed across the whole app

Rules:
- Retrieval quality beats cleverness: respect chunk size/token bounds, keep
  per-engagement namespacing intact, and never break the offline embedding stub.
- Uploads are user-controlled files — sanitize paths/filenames, size-limit,
  and never shell out with untrusted input.
- The framework corpus is grounding material for findings; keep source
  metadata (framework, control_id, version) on every chunk.
- Do not add code comments unless asked.
- Verify your changes keep `python -m pytest desktop/tests/ -q` green.