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
_STATE: dict = {"model": None, "path": None, "error": None, "loading": False}

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


def _auto_gpu_layers() -> int:
    """Offload all layers when a usable CUDA build is present, else CPU."""
    try:
        from engine.hardware import Hardware, can_use_cuda
        return 999 if can_use_cuda() else 0
    except Exception:
        return 0


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

        n_gpu = settings.n_gpu_layers
        if n_gpu == -1:
            n_gpu = _auto_gpu_layers()

        logger.info("Loading GGUF: %s (n_gpu_layers=%s, n_ctx=%d)",
                    os.path.basename(path), n_gpu, settings.n_ctx)
        t0 = time.time()
        llm = Llama(
            model_path=path,
            n_ctx=settings.n_ctx,
            n_gpu_layers=n_gpu,
            verbose=False,
        )
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
        logger.info("Model ready in %.1fs", time.time() - t0)
    except Exception as e:
        _STATE["model"] = None
        _STATE["path"] = None
        _STATE["error"] = str(e)
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
) -> str:
    ensure_loaded()
    llm = _STATE["model"]
    loop = asyncio.get_running_loop()

    def run() -> str:
        out = _complete(llm, messages, temperature, 3072 if json_mode else 2048)
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


def unload() -> None:
    with _LOCK:
        _STATE["model"] = None
        _STATE["path"] = None
        _STATE["error"] = None


def _extract_json_text(raw: str) -> str:
    """Lenient JSON extraction (kept for reference by chat(); json output
    cleaning happens there)."""
    text = raw.strip()
    if text.startswith("```"):
        lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text
