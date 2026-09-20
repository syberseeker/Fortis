"""
CPU-only embedding wrapper. Kept off the GPU intentionally so the RTX 4050's
6GB VRAM is fully available to the LLM in Ollama.

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
import os
from functools import lru_cache
from typing import List

from .config import settings

_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.embedding_model, device="cpu")


def _hash_vector(text: str, dim: int = 384) -> List[float]:
    """Deterministic pseudo-embedding derived from a text hash. Same input
    always yields the same vector; unrelated to semantic content. Test/dev
    only -- see module docstring."""
    raw = hashlib.sha256(text.encode("utf-8")).digest()
    while len(raw) < dim:
        raw += hashlib.sha256(raw).digest()
    values = [(b / 255.0) - 0.5 for b in raw[:dim]]
    norm = sum(v * v for v in values) ** 0.5
    if norm == 0:
        return values
    return [v / norm for v in values]


def embed_texts(texts: List[str]) -> List[List[float]]:
    if _BACKEND == "hash-stub":
        return [_hash_vector(t) for t in texts]

    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vectors.tolist()


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]
