"""
Sets environment variables BEFORE the app package is imported anywhere, since
app/config.py reads them at module-import time. This lets the full backend
(FastAPI routes, SQLite store, Chroma vector store, ingestion, report
rendering) run in tests with no GPU, no Ollama, and no network access to
Hugging Face:

- EMBEDDING_BACKEND=hash-stub: deterministic offline pseudo-embeddings
  (see app/embeddings.py) instead of downloading sentence-transformers
  weights. Not semantically meaningful, but exercises the full Chroma
  add/query code path.
- RAG_LEVEL=standard / RERANK_BACKEND=none: the standard tier exercises the
  hybrid BM25 path (hash-stub embeddings are not semantically meaningful,
  but BM25 keyword scoring is) and RERANK_BACKEND=none keeps the
  cross-encoder reranker offline.
- HIERARCHICAL_THRESHOLD_TOKENS / MAP_BATCH_TOKENS are set low so tests can
  trigger the map-reduce analysis path with small fixtures instead of
  needing genuinely huge documents.
- The LLM itself (core.llm.chat) is monkeypatched per-test in
  test_integration.py, not stubbed here, since different tests need
  different canned responses.
"""
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="cyber-advisor-test-")

os.environ["EMBEDDING_BACKEND"] = "hash-stub"
os.environ["DB_PATH"] = os.path.join(_tmp, "engagements.sqlite3")
os.environ["CHROMA_PERSIST_DIR"] = os.path.join(_tmp, "chroma")
os.environ["UPLOAD_DIR"] = os.path.join(_tmp, "uploads")
os.environ["REPORT_DIR"] = os.path.join(_tmp, "reports")
os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:1"  # deliberately unreachable
os.environ["HIERARCHICAL_THRESHOLD_TOKENS"] = "200"
os.environ["MAP_BATCH_TOKENS"] = "80"
os.environ["RAG_LEVEL"] = "standard"
os.environ["RERANK_BACKEND"] = "none"

import pytest  # noqa: E402
