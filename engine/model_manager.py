"""
GGUF model catalog + download manager for Fortis Desktop.

Tiers are GGUFs from unsloth's HF repos, which publish one file per tier —
small enough to fetch individually, with HTTP resume support via huggingface_hub.

    0.8b      -> unsloth/Qwen3.5-0.8B-GGUF              (~1.0 GB)
    2b        -> unsloth/Qwen3.5-2B-GGUF                (~2.7 GB)
    4b        -> unsloth/Qwen3.5-4B-GGUF                (~3.4 GB)   default tier
    3b-llama  -> bartowski/Llama-3.2-3B-Instruct-GGUF     (~2.0 GB)
    1.5b-coder-> unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF (~1.0 GB)
    3b-coder  -> unsloth/Qwen2.5-Coder-3B-Instruct-GGUF (~1.9 GB)
    8b-llama  -> unsloth/Llama-3.1-8B-Instruct-GGUF      (~3.9 GB)
    7b-coder  -> unsloth/Qwen2.5-Coder-7B-Instruct-GGUF (~3.7 GB)
    7b-r1     -> unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF (~3.6 GB)

Downloads go to core.config.settings.models_dir. The repo file list is
queried at download time so we always grab the actual quantized filename even
if it changes upstream. The active model is recorded in a small JSON state
file next to the weights so the engine knows what to load after restart.
"""
import json
import logging
import os
import threading
from typing import Callable, Dict, List, Optional

from core.config import settings

logger = logging.getLogger(__name__)

TIERS: Dict[str, dict] = {
    "0.8b": {
        "repo": "unsloth/Qwen3.5-0.8B-GGUF",
        "quant": "q4_k_m",
        "size_gb": 1.0,
        "min_vram_gb": 0.0,
        "n_ctx": 8192,
        "roles": ["chat"],
        "label": "Qwen3.5 0.8B (fastest, any laptop)",
        "match": ["0.8b"],
    },
    "2b": {
        "repo": "unsloth/Qwen3.5-2B-GGUF",
        "quant": "q4_k_m",
        "size_gb": 2.7,
        "min_vram_gb": 0.0,
        "n_ctx": 8192,
        "roles": ["chat", "osint"],
        "label": "Qwen3.5 2B (light)",
        "match": ["-2b", "_2b"],
    },
    "4b": {
        "repo": "unsloth/Qwen3.5-4B-GGUF",
        "quant": "q4_k_m",
        "size_gb": 3.4,
        "min_vram_gb": 3.5,
        "n_ctx": 8192,
        "roles": ["chat", "grc", "osint"],
        "label": "Qwen3.5 4B (recommended default)",
        "match": ["-4b", "_4b"],
    },
    "3b-llama": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "quant": "q4_k_m",
        "size_gb": 1.9,
        "min_vram_gb": 2.0,
        "n_ctx": 8192,
        "roles": ["grc", "chat"],
        "label": "Llama 3.2 3B (low-VRAM laptops, long policy context)",
        "match": ["llama-3.2-3b"],
    },
    "1.5b-coder": {
        "repo": "unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        "quant": "q4_k_m",
        "size_gb": 1.0,
        "min_vram_gb": 0.0,
        "n_ctx": 8192,
        "roles": ["code"],
        "label": "Qwen2.5-Coder 1.5B (code review on CPU-only machines)",
        "match": ["coder-1.5b"],
    },
    "3b-coder": {
        "repo": "unsloth/Qwen2.5-Coder-3B-Instruct-GGUF",
        "quant": "q4_k_m",
        "size_gb": 1.9,
        "min_vram_gb": 2.0,
        "n_ctx": 8192,
        "roles": ["code"],
        "label": "Qwen2.5-Coder 3B (code review, 2GB+ VRAM)",
        "match": ["coder-3b"],
    },
    "8b-llama": {
        "repo": "unsloth/Llama-3.1-8B-Instruct-GGUF",
        "quant": "q3_k_m",
        "size_gb": 3.9,
        "min_vram_gb": 5.5,
        "n_ctx": 12288,
        "roles": ["chat", "grc", "cti"],
        "label": "Llama 3.1 8B Q3 (RTX 4050-class, 12k context)",
        "match": ["llama-3.1-8b"],
    },
    "7b-coder": {
        "repo": "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF",
        "quant": "q3_k_m",
        "size_gb": 3.7,
        "min_vram_gb": 5.5,
        "n_ctx": 12288,
        "roles": ["code"],
        "label": "Qwen2.5-Coder 7B Q3 (deep code analysis)",
        "match": ["coder-7b"],
    },
    "7b-r1": {
        "repo": "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "quant": "q3_k_m",
        "size_gb": 3.6,
        "min_vram_gb": 5.5,
        "n_ctx": 8192,
        "roles": ["osint", "cti"],
        "label": "DeepSeek-R1 Distill 7B Q3 (OSINT/CTI reasoning)",
        "match": ["r1-distill-qwen-7b"],
    },
}

_STATE_FILE = os.path.join(settings.models_dir, "active_model.json")
_DOWNLOADING: dict = {"tier": None, "progress": 0.0, "error": None, "done": True}
_LOCK = threading.Lock()


# ---- Active model state ---------------------------------------------------

def current_selection() -> Optional[str]:
    """Absolute path of the selected GGUF, if it exists on disk."""
    try:
        with open(_STATE_FILE) as f:
            path = json.load(f).get("path")
        if path and os.path.isfile(path):
            return path
    except Exception:
        pass
    return None


def current_tier() -> Optional[str]:
    path = current_selection()
    if not path:
        return None
    name = os.path.basename(path).lower()
    for tier_id, info in TIERS.items():
        if any(m in name for m in info["match"]):
            return tier_id
    return "custom"


def set_selection(path: str) -> None:
    with open(_STATE_FILE, "w") as f:
        json.dump({"path": os.path.abspath(path)}, f)


def list_local_models() -> list:
    """All GGUFs already downloaded (recursive scan)."""
    models = []
    try:
        for root, dirs, files in os.walk(settings.models_dir):
            for f in files:
                if f.lower().endswith(".gguf"):
                    full_path = os.path.join(root, f)
                    models.append({
                        "path": full_path,
                        "name": f,
                        "size_gb": round(os.path.getsize(full_path) / 1024 ** 3, 2),
                    })
        return sorted(models, key=lambda m: m["name"])
    except OSError:
        return []


def tier_status() -> dict:
    """tier_id -> {label, size_gb, downloaded, active, roles, min_vram_gb, n_ctx} for the UI picker."""
    local = {m["name"].lower() for m in list_local_models()}
    active = current_selection()
    sel = active and os.path.basename(active).lower()
    out = {}
    for tier_id, info in TIERS.items():
        downloaded = any(all(m in name for m in info["match"]) for name in local)
        active_match = False
        if sel and downloaded:
            active_match = any(m in sel for m in info["match"])
        out[tier_id] = {
            "label": info["label"],
            "size_gb": info["size_gb"],
            "downloaded": downloaded,
            "active": bool(active_match),
            "roles": info["roles"],
            "min_vram_gb": info["min_vram_gb"],
            "n_ctx": info["n_ctx"],
        }
    return out


# ---- Download --------------------------------------------------------------

def _pick_gguf_file(repo_id: str, quant: str = "q4_k_m") -> str:
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo_id, repo_type="model")
    quant_lower = quant.lower()
    candidates = [
        f for f in files
        if f.lower().endswith(".gguf") and quant_lower in f.lower() and "imatrix" not in f.lower()
    ]
    if not candidates:
        candidates = [f for f in files if f.lower().endswith(".gguf")]
    if not candidates:
        raise RuntimeError(f"No GGUF files found in {repo_id}")
    candidates.sort(key=len)
    return candidates[0]


def download_progress() -> dict:
    return dict(_DOWNLOADING)


def download_tier(tier_id: str, progress_cb: Optional[Callable[[float], None]] = None) -> str:
    """Downloads the tier's GGUF (resumable) and selects it. Blocking; call
    from a worker thread. Returns the local path."""
    if tier_id not in TIERS:
        raise ValueError(f"Unknown tier '{tier_id}'. Valid: {list(TIERS)}")
    repo = TIERS[tier_id]["repo"]
    with _LOCK:
        _DOWNLOADING.update(tier=tier_id, progress=0.0, error=None, done=False)
    try:
        from huggingface_hub import hf_hub_download

        filename = _pick_gguf_file(repo, TIERS[tier_id]["quant"])

        def hook(progress) -> None:
            try:
                frac = progress.fraction_completed if hasattr(progress, "fraction_completed") else 0.0
            except Exception:
                frac = 0.0
            _DOWNLOADING["progress"] = frac
            if progress_cb:
                progress_cb(frac)

        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=settings.models_dir,
            resume_download=True,
            tqdm_class=None,
        )
        set_selection(path)
        _DOWNLOADING.update(progress=1.0, done=True)
        logger.info("Downloaded %s -> %s", filename, path)
        return path
    except Exception as e:
        _DOWNLOADING.update(error=str(e), done=True)
        logger.exception("Download failed for tier %s", tier_id)
        raise


def remove_model(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
    if current_selection() == path:
        try:
            os.remove(_STATE_FILE)
        except OSError:
            pass
