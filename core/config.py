import os
import sys


def _env_int(name: str, default: str) -> int:
    """int(os.getenv(...)) with graceful fallback; a malformed value must not
    kill the app at import time."""
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


# Data root: everything persistent (Chroma, SQLite, uploads, reports, models)
# lives under one directory. Override with FORTIS_DATA_DIR.
def _default_data_dir() -> str:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, "Fortis")
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "Fortis")


DATA_DIR = os.getenv("FORTIS_DATA_DIR", _default_data_dir())


class Settings:
    # ---- Engine (llama.cpp) ----
    # Models are GGUF files managed by engine/model_manager.py, stored under
    # DATA_DIR/models. OLLAMA_* env vars are no longer used.
    models_dir: str = os.getenv("FORTIS_MODELS_DIR", os.path.join(DATA_DIR, "models"))
    default_model_tier: str = os.getenv("FORTIS_MODEL_TIER", "auto")
    n_ctx: int = _env_int("FORTIS_N_CTX", "8192")
    n_gpu_layers: int = _env_int("FORTIS_N_GPU_LAYERS", "-1")  # -1 = engine decides

    # ---- Embeddings ----
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # ---- Storage ----
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", os.path.join(DATA_DIR, "chroma"))
    upload_dir: str = os.getenv("UPLOAD_DIR", os.path.join(DATA_DIR, "uploads"))
    report_dir: str = os.getenv("REPORT_DIR", os.path.join(DATA_DIR, "reports"))
    db_path: str = os.getenv("DB_PATH", os.path.join(DATA_DIR, "engagements.sqlite3"))

    # ---- RAG budgets ----
    max_context_tokens: int = _env_int("MAX_CONTEXT_TOKENS", "6000")
    chunk_size_tokens: int = _env_int("CHUNK_SIZE_TOKENS", "500")
    chunk_overlap_tokens: int = _env_int("CHUNK_OVERLAP_TOKENS", "75")
    top_k_user_doc: int = _env_int("TOP_K_USER_DOC", "6")
    top_k_framework: int = _env_int("TOP_K_FRAMEWORK", "4")

    # Above this many tokens of total uploaded content, report generation
    # switches from a single flat LLM pass to a map-reduce pipeline: each
    # document is analyzed in token-bounded batches (map), then the
    # candidate findings are merged, deduped, and framework-mapped in a
    # final pass (reduce). Below this, a flat pass is faster and simpler.
    hierarchical_threshold_tokens: int = _env_int("HIERARCHICAL_THRESHOLD_TOKENS", "5000")
    map_batch_tokens: int = _env_int("MAP_BATCH_TOKENS", "2800")

    # ---- RAG upgrade ----
    # Tiers: "basic" = dense-only retrieval (original behavior);
    # "standard" = + hybrid BM25/RRF, chunk enrichment, heuristic query
    # condensation (no extra LLM call); "full" = + LLM condensation and
    # cross-encoder reranking (auto-disabled below rerank_min_ram_gb).
    rag_level: str = os.getenv("RAG_LEVEL", "standard")
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    rerank_backend: str = os.getenv("RERANK_BACKEND", "sentence-transformers")  # "none" disables
    rerank_candidate_multiplier: int = _env_int("RERANK_CANDIDATE_MULTIPLIER", "4")
    rerank_min_ram_gb: float = _env_float("RERANK_MIN_RAM_GB", "8")
    bm25_rrf_k: int = _env_int("BM25_RRF_K", "60")


def _normalize_rag_level(value: str) -> str:
    value = (value or "").strip().lower()
    return value if value in ("basic", "standard", "full") else "standard"


settings = Settings()
settings.rag_level = _normalize_rag_level(settings.rag_level)

for d in (settings.chroma_persist_dir, settings.upload_dir, settings.report_dir, settings.models_dir):
    os.makedirs(d, exist_ok=True)
os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
