"""
Condenses multi-turn chat history into a standalone retrieval query.

The standard tier uses a free heuristic (the last user turn prepended to
the new message); the full tier spends a single LLM call rewriting the
question and falls back to the heuristic on any failure.
"""


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _has_user_turn(history: list) -> bool:
    return any(
        isinstance(entry, dict) and entry.get("role") == "user"
        for entry in (history or [])
    )


def heuristic_condense(history: list, message: str) -> str:
    """Standalone query = last user turn prepended to the new message."""
    last_user_content = ""
    for entry in reversed(history or []):
        if isinstance(entry, dict) and entry.get("role") == "user":
            last_user_content = _collapse(str(entry.get("content", "")))
            break
    if not last_user_content:
        return message
    if len(last_user_content) > 200:
        last_user_content = last_user_content[:200] + "..."
    return f"{last_user_content} {message}"


def _transcript(history: list) -> str:
    lines = []
    for entry in (history or [])[-6:]:
        role = "User" if entry.get("role") == "user" else "Assistant"
        content = _collapse(str(entry.get("content", "")))[:300]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _strip_quotes(text: str) -> str:
    text = text.strip()
    while len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()
    return text


CONDENSE_SYSTEM_PROMPT = (
    "You rewrite a follow-up question as a standalone search query. "
    "Rewrite the latest question as a single standalone search query that "
    "includes any necessary context from the conversation. "
    "Reply with the query text ONLY: no quotes, no explanation, under 80 words."
)


async def condense_query(history: list, message: str, level: str = "standard") -> str:
    """Full tier: one LLM rewrite with graceful fallback to the heuristic."""
    if level != "full" or not _has_user_turn(history):
        return heuristic_condense(history, message)
    try:
        from engine import chat

        reply = await chat(
            [
                {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _transcript(history) + "\n\nLatest question: " + message,
                },
            ],
            temperature=0.0,
        )
        query = _strip_quotes(reply or "")
        if not query or len(query) > 600:
            raise ValueError("condense reply was empty or oversized")
        return query
    except Exception:
        import logging

        logging.getLogger(__name__).warning(
            "Query condensation failed; using heuristic fallback", exc_info=True
        )
        return heuristic_condense(history, message)
