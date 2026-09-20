"""
Cross-encoder reranking of retrieved chunks.

Pluggable and fail-safe like the embeddings wrapper: the reranker is
skipped when disabled via RERANK_BACKEND, and any load or scoring
failure falls back to the original retrieval order.
"""
import logging
from functools import lru_cache

from .config import settings

_RERANK_CHECKED: bool = False
_RERANK_AVAILABLE: bool = False
_RERANK_BROKEN: bool = False


@lru_cache(maxsize=1)
def _get_cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        getattr(settings, "rerank_model", "BAAI/bge-reranker-base"),
        device="cpu",
    )


def is_available() -> bool:
    """Once-per-process check: backend enabled and enough RAM."""
    global _RERANK_CHECKED, _RERANK_AVAILABLE
    if _RERANK_CHECKED:
        return _RERANK_AVAILABLE
    _RERANK_CHECKED = True
    try:
        # No engine.hardware.detect() here (core parity note): this legacy
        # containerized stack has no engine package to consult, and
        # containerized deployments control their own resource limits via
        # RERANK_BACKEND / RERANK_MIN_RAM_GB anyway. "none" disables outright.
        backend = getattr(settings, "rerank_backend", "sentence-transformers")
        _RERANK_AVAILABLE = backend != "none"
    except Exception:
        _RERANK_AVAILABLE = False
    return _RERANK_AVAILABLE


def rerank(query: str, hits: list, top_k: int) -> list:
    """Reorder hits by cross-encoder score, or fall back to input order."""
    global _RERANK_BROKEN
    if not is_available():
        return hits[:top_k]
    if len(hits) <= top_k:
        return hits
    if _RERANK_BROKEN:
        return hits[:top_k]
    try:
        model = _get_cross_encoder()
        pairs = [(query, hit["text"]) for hit in hits]
        scores = model.predict(pairs)
        ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
        return [hit for hit, _ in ranked[:top_k]]
    except Exception:
        _RERANK_BROKEN = True
        logging.getLogger(__name__).warning(
            "Cross-encoder reranking failed; falling back to retrieval order"
        )
        return hits[:top_k]
