"""
Real inference backend: llama-cpp-python on a local Qwen3.5 GGUF.

Loaded lazily — importing this module does not import llama_cpp or load
weights. The model loads on first chat call (or explicitly via load_model
from the desktop launcher / model manager), inside a lock, with a warm-up
generation so the first user message doesn't absorb load time.

Uses create_chat_completion (not raw prompt strings) so the model's chat
template is applied by llama.cpp; Qwen3.5 emits clean content without
reasoning tags at these temperatures.
"""
import asyncio
import json
import logging
import os
import threading
import time
from typing import List, Dict, Optional

from core.config import settings

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_STATE: dict = {
    "model": None, "path": None, "error": None, "loading": False,
    "offload_level": None,
}

_STOP_TOKENS = ["<|im_end|>", "</think>"]


def current_model_path() -> Optional[str]:
    """Path of the model that should be used, or None if none is available.
    Priority: FORTIS_MODEL_PATH env var, then the model manager's selection,
    then the newest GGUF in the models dir."""
    env = os.getenv("FORTIS_MODEL_PATH")
    if env and os.path.isfile(env):
        return env
    try:
        from engine.model_manager import current_selection
        sel = current_selection()
        if sel:
            return sel
    except Exception:
        pass
    try:
        ggufs = [
            f for f in os.listdir(settings.models_dir) if f.lower().endswith(".gguf")
        ]
        if ggufs:
            newest = max(
                ggufs,
                key=lambda f: os.path.getmtime(os.path.join(settings.models_dir, f)),
            )
            return os.path.join(settings.models_dir, newest)
    except OSError:
        pass
    return None


def model_status() -> Optional[str]:
    """None if a model is loaded or loadable, else a human-readable warning."""
    if _STATE["model"] is not None:
        return None
    if _STATE["error"]:
        return f"Model failed to load: {_STATE['error']}"
    if current_model_path() is None:
        return (
            "No model file found. Use the model picker in the app to download "
            "a Qwen3.5 GGUF (tiers: 0.8B / 2B / 4B / 9B), or set FORTIS_MODEL_PATH."
        )
    return None  # loadable on demand


def _backend_wheel() -> str:
    """Which llama-cpp wheel flavor is usable: "cuda" when the verified CUDA
    marker exists and llama_cpp imports, "cpu" when llama_cpp imports without
    the marker, "none" when there is no llama_cpp at all."""
    try:
        import llama_cpp  # noqa: F401
    except Exception:
        return "none"
    try:
        from engine.hardware import cuda_marker_exists
        return "cuda" if cuda_marker_exists() else "cpu"
    except Exception:
        return "cpu"


def hardware_status(hw: Optional["Hardware"] = None) -> Dict[str, object]:
    """Hardware/offload snapshot for /api/model/status. `offload` is
    "unknown" before the first load; a zero layer count is reported as "cpu"
    even when CUDA is available so the UI can warn about failed offload."""
    if hw is None:
        try:
            from engine.hardware import detect
            hw = detect()
        except Exception:
            hw = None
    level = _STATE.get("offload_level")
    if level is None:
        offload = "unknown"
    elif level >= 999:
        offload = "full"
    elif level > 0:
        offload = "partial"
    else:
        offload = "cpu"
    return {
        "vram_gb": round(getattr(hw, "vram_gb", 0.0), 1),
        "ram_gb": round(getattr(hw, "ram_gb", 0.0), 1),
        "has_nvidia": bool(getattr(hw, "has_nvidia", False)),
        "offload": offload,
        "offload_layers": level,
        "backend_wheel": _backend_wheel(),
    }


def _auto_gpu_layers(path: Optional[str] = None) -> int:
    """Chooses n_gpu_layers when settings.n_gpu_layers is -1 (auto).

    An explicit value >= 0 is passed through verbatim. With a usable CUDA
    build, full offload (999) requires free VRAM to cover the model file plus
    an estimated KV cache of kv_gb = (n_ctx / 8192) * model_gb * 0.35 with a
    0.5 GB safety margin; otherwise a proportional layer count
    max(1, int(999 * free_vram / (model_gb + kv_gb))) clamped to 95% is used
    (llama.cpp scales its numeric layer parameter against the true layer
    count, so a fraction of 999 offloads roughly that fraction of layers).
    Without CUDA this is always 0.
    """
    try:
        if settings.n_gpu_layers >= 0:
            return settings.n_gpu_layers
        from engine.hardware import can_use_cuda, free_vram_gb
        if not can_use_cuda():
            return 0
        if path is None:
            path = current_model_path()
        model_gb = _model_size_gb(path)
        kv_gb = (settings.n_ctx / 8192.0) * model_gb * 0.35
        need_gb = model_gb + kv_gb
        free_gb = free_vram_gb()
        if free_gb >= need_gb + 0.5:
            return 999
        fraction = max(0.0, min(0.95, free_gb / need_gb if need_gb > 0 else 0.0))
        return max(1, int(999 * fraction))
    except Exception:
        return 0


def _model_size_gb(path: Optional[str]) -> float:
    """GGUF size in GiB; 0.0 when the file is missing/unreadable."""
    if not path:
        return 0.0
    try:
        return os.path.getsize(path) / (1024 ** 3)
    except OSError:
        return 0.0


def _resolved_threads() -> int:
    """Worker thread count for llama.cpp: settings.n_threads when set > 0,
    else the hyperthread-adjusted physical core estimate."""
    if settings.n_threads > 0:
        return settings.n_threads
    try:
        from engine.hardware import physical_core_estimate
        return physical_core_estimate()
    except Exception:
        logical = os.cpu_count() or 2
        return max(1, logical // 2 if logical >= 16 else logical)


def _oom_ladder(attempted: int) -> List[int]:
    """Fallback n_gpu_layers attempts after `attempted` fails: ~75%, ~50%,
    then CPU-only. Empty when the first attempt is already CPU."""
    if attempted <= 0:
        return []
    steps = []
    prev = attempted
    for frac in (0.75, 0.5, 0.0):
        nxt = int(attempted * frac)
        if nxt < prev:
            steps.append(nxt)
            prev = nxt
    return steps


def _load_sync() -> None:
    """Blocking load. Caller must hold _LOCK."""
    path = current_model_path()
    if path is None:
        _STATE["error"] = "no model file available"
        return
    if _STATE["model"] is not None and _STATE["path"] == path:
        return  # already loaded
    try:
        from llama_cpp import Llama

        n_threads = _resolved_threads()
        known = _STATE.get("offload_level")
        if known is not None:
            first = known
        else:
            first = settings.n_gpu_layers
            if first == -1:
                first = _auto_gpu_layers(path)
        attempts = [first] + _oom_ladder(first)

        llm = None
        last_error: Optional[Exception] = None
        for i, n_gpu in enumerate(attempts):
            if i > 0:
                logger.warning(
                    "GPU offload failed at n_gpu_layers=%d (%s); retrying with n_gpu_layers=%d",
                    attempts[i - 1], last_error, n_gpu,
                )
            logger.info(
                "Loading GGUF: %s (n_gpu_layers=%s, n_ctx=%d, n_threads=%d, n_batch=%d)",
                os.path.basename(path), n_gpu, settings.n_ctx, n_threads, settings.n_batch,
            )
            t0 = time.time()
            try:
                llm = Llama(
                    model_path=path,
                    n_ctx=settings.n_ctx,
                    n_gpu_layers=n_gpu,
                    n_threads=n_threads,
                    n_batch=settings.n_batch,
                    verbose=False,
                )
                break
            except Exception as e:
                last_error = e
                llm = None
                if n_gpu <= 0 or not _is_cuda_oom_error(e):
                    raise
        if llm is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("model load failed")
        # warm-up so the first real message is not the compile/load hit
        try:
            llm.create_chat_completion(
                messages=[{"role": "user", "content": "OK"}],
                max_tokens=1, temperature=0.0,
            )
        except Exception:
            pass
        _STATE["model"] = llm
        _STATE["path"] = path
        _STATE["error"] = None
        _STATE["offload_level"] = n_gpu
        logger.info("Model ready in %.1fs (n_gpu_layers=%d)", time.time() - t0, n_gpu)
    except Exception as e:
        _STATE["model"] = None
        _STATE["path"] = None
        _STATE["error"] = str(e)
        _STATE["offload_level"] = None
        logger.exception("Failed to load model %s", path)
        raise


def load_model(path: Optional[str] = None, n_ctx: Optional[int] = None) -> None:
    """Explicit load (model switch in the UI). Raises on failure."""
    if path:
        os.environ["FORTIS_MODEL_PATH"] = path
    if n_ctx:
        settings.n_ctx = n_ctx
    with _LOCK:
        _STATE["model"] = None
        _STATE["path"] = None
        _STATE["error"] = None
        _load_sync()


def ensure_loaded() -> None:
    with _LOCK:
        if _STATE["model"] is None:
            _load_sync()


def _clean(text: str) -> str:
    """Strips reasoning fences if a model emits them despite chat templating."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    if text.strip().startswith("</think>") and "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def _extract_json_text(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


def _complete(llm, messages, temperature, max_tokens, stream=False):
    return llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=_STOP_TOKENS,
        stream=stream,
    )


async def chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    json_mode: bool = False,
    max_tokens: int | None = None,
) -> str:
    ensure_loaded()
    llm = _STATE["model"]
    loop = asyncio.get_running_loop()
    # Resolve the default outside the executor closure: assigning to
    # max_tokens inside run() would make it a closure-local name and the
    # first read would raise UnboundLocalError.
    limit = max_tokens if max_tokens is not None else (3072 if json_mode else 2048)

    def run() -> str:
        out = _complete(llm, messages, temperature, limit)
        content = out["choices"][0]["message"].get("content", "") or ""
        content = _clean(content)
        if json_mode:
            content = _extract_json_text(content)
        return content

    return await loop.run_in_executor(None, run)


async def chat_stream(
    messages: List[Dict[str, str]], temperature: float = 0.2
):
    ensure_loaded()
    llm = _STATE["model"]
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def run():
        try:
            stream = _complete(llm, messages, temperature, 2048, stream=True)
            in_reasoning = False
            for part in stream:
                delta = part["choices"][0].get("delta", {})
                piece = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or ""
                if piece and "</think>" in piece:
                    piece = piece.split("</think>")[-1]
                if piece:
                    loop.call_soon_threadsafe(queue.put_nowait, ("t", piece))
                elif reasoning and not in_reasoning:
                    in_reasoning = True  # suppress reasoning from the UI stream
            loop.call_soon_threadsafe(queue.put_nowait, ("done", ""))
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))

    threading.Thread(target=run, daemon=True).start()
    while True:
        kind, payload = await queue.get()
        if kind == "t":
            yield payload
        elif kind == "error":
            raise RuntimeError(payload)
        else:
            break


def _is_cuda_oom_error(e: Exception) -> bool:
    text = str(e).lower()
    return any(
        k in text
        for k in (
            "out of memory",
            "cuda_error",
            "cudaerror",
            "failed to allocate",
            "ggml_backend_cuda_buffer",
        )
    )


def unload() -> None:
    with _LOCK:
        _STATE["model"] = None
        _STATE["path"] = None
        _STATE["error"] = None
        _STATE["offload_level"] = None
