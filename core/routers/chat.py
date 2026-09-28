from typing import List, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import rag, store
from .. import report_intent as ri
from .. import guardrail
from ..config import settings
from ..condense import condense_query
from ..rag import ROLE_PRESETS
from ..report_service import trim_focus_text, generate_report_result
from ..structured import FORMAT_INSTRUCTIONS, REPAIR_INSTRUCTIONS, detect_structured_output_request, reply_has_valid_diagram, is_valid_gfm_table
from engine import chat, chat_stream

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    engagement_id: str
    message: str
    history: Optional[List[Dict[str, str]]] = None
    role: Optional[str] = None


def _require_engagement(engagement_id: str) -> None:
    if not store.get_engagement(engagement_id):
        raise HTTPException(404, f"Engagement '{store.clean(engagement_id)}' not found. Create or select one first.")


async def _persist_turn(engagement_id: str, user_text: str, assistant_text: str) -> None:
    """Best-effort persistence of a chat turn; a failed write must never break
    the chat reply itself."""
    try:
        store.append_chat_message(engagement_id, "user", user_text)
        store.append_chat_message(engagement_id, "assistant", assistant_text)
    except Exception:
        pass


async def run_structured_turn(engagement_id: str, message: str, history, intent: str, role: Optional[str] = None) -> str:
    if role is not None and role not in ROLE_PRESETS:
        role = None
    messages = rag.build_chat_messages(engagement_id, message, history, role=role)
    messages[-1]["content"] += "\n\n" + FORMAT_INSTRUCTIONS[intent]
    reply = await chat(messages)
    
    valid = reply_has_valid_diagram(reply) if intent == "diagram" else is_valid_gfm_table(reply)
    if not valid:
        repair_history = history[-12:] + [{"role": "assistant", "content": reply}]
        messages = rag.build_chat_messages(engagement_id, REPAIR_INSTRUCTIONS[intent], repair_history, role=role)
        messages[-1]["content"] += "\n\nOriginal request: " + message
        reply = await chat(messages)
    
    return reply


@router.post("")
async def chat_endpoint(req: ChatRequest):
    _require_engagement(req.engagement_id)
    req.message = store.clean(req.message)
    if req.history:
        for entry in req.history:
            entry["content"] = store.clean(entry.get("content", ""))
    kind, refusal = guardrail.classify(req.message)
    if kind == "offtopic":
        await _persist_turn(req.engagement_id, req.message, refusal)
        return {"reply": refusal}
    fmt = ri.detect_report_request(req.message)
    if fmt:
        if ri.is_stub_backend():
            reply = ri.chat_stub_gate_reply()
            await _persist_turn(req.engagement_id, req.message, reply)
            return {"reply": reply}
        result = await generate_report_result(req.engagement_id, fmt, trim_focus_text(req.message))
        if "error" in result:
            raise HTTPException(result.get("status", 400), result["error"])
        return {"reply": ri.format_report_reply(result, ri.is_stub_backend())}
    role = req.role if req.role in ROLE_PRESETS else None
    retrieval_query = await condense_query(req.history, req.message, settings.rag_level)
    intent = detect_structured_output_request(req.message)
    if intent:
        reply = await run_structured_turn(req.engagement_id, req.message, req.history or [], intent, role)
        await _persist_turn(req.engagement_id, req.message, reply)
        return {"reply": reply}
    messages = rag.build_chat_messages(req.engagement_id, req.message, req.history, role=role, retrieval_query=retrieval_query)
    reply = await chat(messages)
    if guardrail.is_offtopic_refusal(reply):
        messages = rag.build_chat_messages(
            req.engagement_id, guardrail.forced_answer_prompt(req.message), req.history, role=role, retrieval_query=retrieval_query
        )
        reply = await chat(messages)
    await _persist_turn(req.engagement_id, req.message, reply)
    return {"reply": reply}


@router.post("/stream")
async def chat_stream_endpoint(req: ChatRequest):
    _require_engagement(req.engagement_id)
    req.message = store.clean(req.message)
    if req.history:
        for entry in req.history:
            entry["content"] = store.clean(entry.get("content", ""))
    kind, refusal = guardrail.classify(req.message)
    if kind == "offtopic":
        async def refusal_generator():
            await _persist_turn(req.engagement_id, req.message, refusal)
            yield refusal
        return StreamingResponse(refusal_generator(), media_type="text/plain")
    fmt = ri.detect_report_request(req.message)
    if fmt:
        if ri.is_stub_backend():
            async def gate_generator():
                reply = ri.chat_stub_gate_reply()
                await _persist_turn(req.engagement_id, req.message, reply)
                yield reply
            return StreamingResponse(gate_generator(), media_type="text/plain")
        result = await generate_report_result(req.engagement_id, fmt, trim_focus_text(req.message))
        if "error" in result:
            raise HTTPException(result.get("status", 400), result["error"])
        reply = ri.format_report_reply(result, ri.is_stub_backend())
        await _persist_turn(req.engagement_id, req.message, reply)
        async def generator():
            yield reply
        return StreamingResponse(generator(), media_type="text/plain")
    role = req.role if req.role in ROLE_PRESETS else None
    retrieval_query = await condense_query(req.history, req.message, settings.rag_level)
    intent = detect_structured_output_request(req.message)
    if intent:
        reply = await run_structured_turn(req.engagement_id, req.message, req.history or [], intent, role)
        await _persist_turn(req.engagement_id, req.message, reply)
        async def generator():
            yield reply
        return StreamingResponse(generator(), media_type="text/plain")
    messages = rag.build_chat_messages(req.engagement_id, req.message, req.history, role=role, retrieval_query=retrieval_query)

    async def generator():
        acc = []
        async for token in chat_stream(messages):
            acc.append(token)
            yield token
        full = "".join(acc)
        if guardrail.is_offtopic_refusal(full):
            retry = rag.build_chat_messages(
                req.engagement_id, guardrail.forced_answer_prompt(req.message), req.history, role=role, retrieval_query=retrieval_query
            )
            async for token in chat_stream(retry):
                yield token
            return
        await _persist_turn(req.engagement_id, req.message, full)

    return StreamingResponse(generator(), media_type="text/plain")


@router.get("/history/{engagement_id}")
async def chat_history(engagement_id: str, limit: int = 500):
    _require_engagement(engagement_id)
    try:
        turns = store.get_chat_history(engagement_id, limit=limit)
    except (KeyError, ValueError):
        raise HTTPException(404, f"Engagement '{store.clean(engagement_id)}' not found.")
    return {"engagement_id": store.clean(engagement_id), "turns": turns}


@router.delete("/history/{engagement_id}")
async def chat_history_clear(engagement_id: str, confirm: bool = False):
    _require_engagement(engagement_id)
    if not confirm:
        raise HTTPException(400, "Pass confirm=true to delete chat history for this engagement.")
    removed = store.clear_chat_history(engagement_id)
    return {"engagement_id": store.clean(engagement_id), "removed": removed}
