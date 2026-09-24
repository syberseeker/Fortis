"""
Embedding wrapper for retrieval.

Backend is pluggable via EMBEDDING_BACKEND:
- "sentence-transformers" (default): real embeddings, downloads model weights
  on first use. This is what docker-compose uses in production.
- "hash-stub": fast, deterministic, offline pseudo-embeddings with no model
  download. Not semantically meaningful -- retrieval quality is unusable for
  real work -- but lets the full upload -> retrieve -> report pipeline be
  exercised in CI/tests/dev environments without GPU or network access to
  Hugging Face. See backend/tests/.
"""
import hashlib
import logging
import os
from functools import lru_cache
from typing import List

from .config import settings

logger = logging.getLogger(__name__)

_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")

_resolved_device: str = ""
_cuda_broken: bool = False


def _resolve_device() -> str:
    """Pick the encode device once per process. 'cuda' when an NVIDIA GPU is
    present, else 'cpu'. Downgrades permanently to 'cpu' after a CUDA failure
    at model load or first encode (e.g. driver mismatch or CUDA wheel missing
    for torch)."""
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
    if hasattr(_load_model, "cache_clear"):
        _load_model.cache_clear()


@lru_cache(maxsize=1)
def _load_model(device: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.embedding_model, device=device)


def _get_model():
    device = _resolve_device()
    try:
        return _load_model(device)
    except Exception:
        if device != "cuda":
            raise
        _demote_to_cpu()
        logger.warning(
            "CUDA embedding model load failed (%s); falling back to CPU",
            device,
            exc_info=True,
        )
        return _load_model("cpu")


def _hash_vector(text: str, dim: int = 384) -> List[float]:
    """Deterministic pseudo-embedding derived from a text hash. Same input
    always yields the same vector; unrelated to semantic content. Test/dev
    only -- see module docstring."""
    raw = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
    while len(raw) < dim:
        raw += hashlib.sha256(raw).digest()
    values = [(b / 255.0) - 0.5 for b in raw[:dim]]
    norm = sum(v * v for v in values) ** 0.5
    if norm == 0:
        return values
    return [v / norm for v in values]


def _encode(texts: List[str], model) -> List[List[float]]:
    vectors = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vectors.tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    if _BACKEND == "hash-stub":
        return [_hash_vector(t) for t in texts]

    model = _get_model()
    try:
        return _encode(texts, model)
    except Exception:
        if _resolved_device != "cuda":
            raise
        _demote_to_cpu()
        logger.warning(
            "CUDA encoding failed; falling back to CPU embeddings", exc_info=True
        )
        return _encode(texts, _load_model("cpu"))


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]
