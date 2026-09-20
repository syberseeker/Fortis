import json
from typing import List, Dict, Optional

import httpx

from .config import settings

_TIMEOUT = httpx.Timeout(300.0, connect=10.0)


async def chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    json_mode: bool = False,
) -> str:
    """Calls Ollama's /api/chat. If json_mode is True, requests the model
    constrain output to valid JSON (Ollama's `format: "json"`)."""
    payload = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{settings.ollama_base_url}/api/chat", json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]


async def chat_stream(messages: List[Dict[str, str]], temperature: float = 0.2):
    """Async generator yielding text chunks, for streaming to Open WebUI."""
    payload = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": True,
        "think": False,
        "options": {"temperature": temperature},
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        async with client.stream(
            "POST", f"{settings.ollama_base_url}/api/chat", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
                if chunk.get("done"):
                    break


async def check_model_available() -> Optional[str]:
    """Returns None if OK, otherwise an error message."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if settings.ollama_model not in models:
                return (
                    f"Model '{settings.ollama_model}' not pulled yet. Run: "
                    f"docker exec -it fortis-ollama ollama pull {settings.ollama_model}"
                )
            return None
    except httpx.HTTPError as e:
        return f"Could not reach Ollama at {settings.ollama_base_url}: {e}"
