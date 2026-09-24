"""
Cross-encoder reranking of retrieved chunks.

Pluggable and fail-safe like the embeddings wrapper: the reranker is
skipped on machines below the RAM threshold, and any load or scoring
failure falls back to the original retrieval order.

A transient failure (first-predict OOM, disk hiccup) does not disable
reranking permanently: the model is retried until _RERANK_MAX_FAILURES
load/score failures have accumulated for the process.

Device: the cross-encoder runs on CUDA when engine.hardware reports a
usable NVIDIA GPU; a CUDA failure at load or predict permanently downgrades
the process to CPU.
"""
import logging
from functools import lru_cache

from .config import settings

logger = logging.getLogger(__name__)

_RERANK_CHECKED: bool = False
_RERANK_AVAILABLE: bool = False
_RERANK_FAILURES: int = 0
_RERANK_MAX_FAILURES: int = 2

_resolved_device: str = ""
_cuda_broken: bool = False


def _resolve_device() -> str:
    """Pick the reranker device once per process. 'cuda' when an NVIDIA GPU
    is present, else 'cpu'. Downgrades permanently to 'cpu' after a CUDA
    failure at model load."""
    global _resolved_device, _cuda_broken
    if _resolved_device:
        return _resolved_device
    device = "cpu"
    try:
        from engine.hardware import can_use_cuda

        if can_use_cuda() and not _cuda_broken:
            device = "cuda"
    except Exception:
        device = "cpu"
    _resolved_device = device
    return device


def _demote_to_cpu() -> None:
    global _resolved_device, _cuda_broken
    _cuda_broken = True
    _resolved_device = "cpu"
    if hasattr(_load_cross_encoder, "cache_clear"):
        _load_cross_encoder.cache_clear()


@lru_cache(maxsize=1)
def _load_cross_encoder(device: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        getattr(settings, "rerank_model", "BAAI/bge-reranker-base"),
        device=device,
    )


def _get_cross_encoder():
    device = _resolve_device()
    try:
        return _load_cross_encoder(device)
    except Exception:
        if device != "cuda":
            raise
        _demote_to_cpu()
        logger.warning(
            "CUDA cross-encoder load failed (%s); falling back to CPU",
            device,
            exc_info=True,
        )
        return _load_cross_encoder("cpu")


def is_available() -> bool:
    """Once-per-process check: backend enabled and enough RAM."""
    global _RERANK_CHECKED, _RERANK_AVAILABLE
    if _RERANK_CHECKED:
        return _RERANK_AVAILABLE
    _RERANK_CHECKED = True
    try:
        from engine.hardware import detect

        backend = getattr(settings, "rerank_backend", "sentence-transformers")
        min_ram_gb = getattr(settings, "rerank_min_ram_gb", 8.0)
        _RERANK_AVAILABLE = backend != "none" and detect().ram_gb >= min_ram_gb
    except Exception:
        _RERANK_AVAILABLE = False
    return _RERANK_AVAILABLE


def rerank(query: str, hits: list, top_k: int) -> list:
    """Reorder hits by cross-encoder score, or fall back to input order."""
    global _RERANK_FAILURES
    if not is_available():
        return hits[:top_k]
    if len(hits) <= top_k:
        return hits
    if _RERANK_FAILURES >= _RERANK_MAX_FAILURES:
        return hits[:top_k]
    try:
        model = _get_cross_encoder()
        pairs = [(query, hit["text"]) for hit in hits]
        scores = model.predict(pairs)
        ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
        return [hit for hit, _ in ranked[:top_k]]
    except Exception:
        if _resolved_device == "cuda":
            _demote_to_cpu()
        _RERANK_FAILURES += 1
        logger.warning(
            "Cross-encoder reranking failed; falling back to retrieval order"
        )
        return hits[:top_k]


def reset_retry_state() -> None:
    """Reset the failure counter and clear the cached model — for tests or
    when the desktop launcher wants to re-enable reranking."""
    global _RERANK_FAILURES
    _RERANK_FAILURES = 0
    if hasattr(_load_cross_encoder, "cache_clear"):
        _load_cross_encoder.cache_clear()
