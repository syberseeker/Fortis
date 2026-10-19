"""Opt-in evaluation instrumentation.

When FORTIS_EVAL_LOG=1, each chat turn appends one JSON line to
<DATA_DIR>/eval_log/turns-YYYYMMDD.jsonl for offline retrieval/answer
quality analysis. When disabled every hook is a no-op with zero file IO.
build_env_manifest() captures the environment snapshot (packages, git
commit, settings, model file, hardware) that makes eval runs reproducible;
it is import-safe without llama_cpp installed. Run `python -m core.telemetry`
to print it as pretty JSON.
"""
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import settings

logger = logging.getLogger(__name__)

_PACKAGE_NAMES = (
    "fastapi", "uvicorn", "pydantic", "chromadb", "sentence-transformers",
    "rank-bm25", "pymupdf", "reportlab", "python-docx", "python-pptx",
    "llama-cpp-python",
)

_PROVENANCE_META_KEYS = (
    "filename", "chunk_index", "page_label", "start_char", "end_char",
    "framework", "control_id",
)

_MANIFEST_SETTING_KEYS = (
    "rag_level", "top_k_user_doc", "top_k_framework", "chunk_size_tokens",
    "chunk_overlap_tokens", "hierarchical_threshold_tokens",
    "map_batch_tokens", "max_context_tokens", "bm25_rrf_k",
    "rerank_backend", "rerank_model", "seed", "embedding_model", "eval_log",
)


def _eval_log_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(settings.db_path)), "eval_log")


def _retrieval_summary(retrieved: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for key in ("user_hits", "framework_hits"):
        for hit in (retrieved or {}).get(key) or []:
            entry: Dict[str, Any] = {
                "id": hit.get("id"),
                "distance": hit.get("distance"),
                "rank": hit.get("rank"),
            }
            meta = hit.get("metadata") or {}
            for field in _PROVENANCE_META_KEYS:
                if field in meta:
                    entry[field] = meta.get(field)
            out.append(entry)
    return out


def log_chat_turn(
    *,
    engagement_id: str,
    query: str,
    condensed_query: str,
    retrieved: Optional[Dict[str, Any]],
    sources: Optional[List[Dict[str, Any]]],
    usage: Optional[Dict[str, Any]],
    model_id: Optional[str],
    reply: str,
) -> None:
    """Append one eval event for a completed chat turn. No-op unless
    eval_log is enabled; never raises into the chat pipeline."""
    if not getattr(settings, "eval_log", False):
        return
    try:
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "engagement_id": engagement_id,
            "role": "assistant",
            "rag_level": settings.rag_level,
            "query": query,
            "condensed_query": condensed_query,
            "retrieval": _retrieval_summary(retrieved),
            "sources": sources,
            "usage": usage,
            "model_id": model_id,
            "reply_chars": len(reply or ""),
        }
        directory = _eval_log_dir()
        os.makedirs(directory, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        with open(os.path.join(directory, f"turns-{day}.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        logger.debug("eval log write failed", exc_info=True)


def _package_versions() -> Dict[str, str]:
    from importlib import metadata

    versions: Dict[str, str] = {}
    for name in _PACKAGE_NAMES:
        try:
            versions[name] = metadata.version(name)
        except Exception:
            continue
    return versions


def _git_commit() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        return None
    return None


def _sha256_of_file(path: str) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except Exception:
        return None


def build_env_manifest() -> Dict[str, Any]:
    """Environment snapshot for reproducible evaluation. Every probe is
    fail-safe (None/empty on error) so the manifest always builds."""
    model_path: Optional[str] = None
    backend_wheel: Optional[str] = None
    try:
        from engine.llama_engine import current_model_path, _backend_wheel

        model_path = current_model_path()
        backend_wheel = _backend_wheel()
    except Exception:
        pass

    hardware: Optional[Dict[str, Any]] = None
    try:
        from engine.hardware import detect

        hw = detect()
        hardware = {
            "ram_gb": getattr(hw, "ram_gb", None),
            "vram_gb": getattr(hw, "vram_gb", None),
            "has_nvidia": getattr(hw, "has_nvidia", None),
        }
    except Exception:
        pass

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "git_commit": _git_commit(),
        "settings": {key: getattr(settings, key, None) for key in _MANIFEST_SETTING_KEYS},
        "model_path": model_path,
        "model_sha256": _sha256_of_file(model_path) if model_path else None,
        "backend_wheel": backend_wheel,
        "hardware": hardware,
    }


if __name__ == "__main__":
    print(json.dumps(build_env_manifest(), indent=2, ensure_ascii=False, sort_keys=True))
