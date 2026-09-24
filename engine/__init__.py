"""
LLM engine abstraction for Fortis Desktop.

Two implementations behind one interface:

- llama backend: real inference via llama-cpp-python on a local Qwen3.5 GGUF
  (engine/llama_engine.py). Uses create_chat_completion so the model's own
  chat template (and any reasoning tags) are handled by llama.cpp.
- stub backend: deterministic offline replies for tests/no-model demos
  (engine/stub_backend.py).

Selection: FORTIS_LLM_BACKEND env var (llama|stub|auto). In auto mode, llama
is used when a model file is available, otherwise stub. Importing this
module never loads llama-cpp-python or any weights.
"""
import os
from typing import AsyncIterator, List, Dict

_ACTIVE_BACKEND: str = os.getenv("FORTIS_LLM_BACKEND", "auto")  # auto | llama | stub


def set_backend(name: str) -> None:
    """Forces a backend. Used by tests ('stub') and the desktop launcher."""
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = name


def get_backend() -> str:
    return _ACTIVE_BACKEND


def _resolve() -> str:
    if _ACTIVE_BACKEND != "auto":
        return _ACTIVE_BACKEND
    try:
        from engine.llama_engine import current_model_path
        if current_model_path():
            return "llama"
    except Exception:
        pass
    return "stub"


async def chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    json_mode: bool = False,
    max_tokens: int | None = None,
) -> str:
    backend = _resolve()
    if backend == "llama":
        from engine.llama_engine import chat as llama_chat
        return await llama_chat(messages, temperature=temperature, json_mode=json_mode, max_tokens=max_tokens)
    from engine.stub_backend import chat as stub_chat
    return await stub_chat(messages, temperature=temperature, json_mode=json_mode, max_tokens=max_tokens)


async def chat_stream(
    messages: List[Dict[str, str]], temperature: float = 0.2
) -> AsyncIterator[str]:
    backend = _resolve()
    if backend == "llama":
        from engine.llama_engine import chat_stream as llama_stream
        async for token in llama_stream(messages, temperature):
            yield token
        return
    from engine.stub_backend import chat_stream as stub_stream
    async for token in stub_stream(messages, temperature):
        yield token


async def check_model_available():
    """Returns None if OK, otherwise a human-readable warning string."""
    backend = _resolve()
    if backend == "stub":
        return "Using built-in stub responses (no model loaded). Reports will be placeholder data."
    from engine.llama_engine import model_status
    return model_status()
