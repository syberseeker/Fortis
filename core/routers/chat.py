import os
import re
import uuid
from datetime import datetime
from typing import List, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .. import rag, store
from .. import report_intent as ri
from .. import guardrail
from ..config import settings
from ..condense import condense_query
from ..rag import ROLE_PRESETS
from ..reports import chat_export
from ..report_service import trim_focus_text, generate_report_result
from ..structured import FORMAT_INSTRUCTIONS, REPAIR_INSTRUCTIONS, detect_structured_output_request, reply_has_valid_diagram, is_valid_gfm_table
from engine import chat, chat_stream

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    engagement_id: str
    message: str
    history: Optional[List[Dict[str, str]]] = None
    role: Optional[str] = None


class ChatExportRequest(BaseModel):
    engagement_id: str
    format: str = "md"
    message_id: Optional[int] = None
    latest_assistant: bool = False
    transcript: bool = False


def _require_engagement(engagement_id: str) -> None:
    if not store.get_engagement(engagement_id):
        raise HTTPException(404, f"Engagement '{store.clean(engagement_id)}' not found. Create or select one first.")


async def _persist_turn(engagement_id: str, user_text: str, assistant_text: str) -> None:
    """Best-effort persistence of a chat turn; a failed write must never break
    the chat reply itself."""
    try:
        await run_in_threadpool(store.append_chat_message, engagement_id, "user", user_text)
        await run_in_threadpool(store.append_chat_message, engagement_id, "assistant", assistant_text)
    except Exception:
        pass


async def run_structured_turn(engagement_id: str, message: str, history, intent: str, role: Optional[str] = None) -> str:
    if role is not None and role not in ROLE_PRESETS:
        role = None
    messages = await run_in_threadpool(rag.build_chat_messages, engagement_id, message, history, role=role)
    messages[-1]["content"] += "\n\n" + FORMAT_INSTRUCTIONS[intent]
    reply = await chat(messages)
    
    valid = reply_has_valid_diagram(reply) if intent == "diagram" else is_valid_gfm_table(reply)
    if not valid:
        repair_history = history[-12:] + [{"role": "assistant", "content": reply}]
        messages = await run_in_threadpool(rag.build_chat_messages, engagement_id, REPAIR_INSTRUCTIONS[intent], repair_history, role=role)
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
    messages = await run_in_threadpool(rag.build_chat_messages, req.engagement_id, req.message, req.history, role=role, retrieval_query=retrieval_query)
    reply = await chat(messages)
    if guardrail.is_offtopic_refusal(reply):
        messages = await run_in_threadpool(
            rag.build_chat_messages,
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
    messages = await run_in_threadpool(rag.build_chat_messages, req.engagement_id, req.message, req.history, role=role, retrieval_query=retrieval_query)

    async def generator():
        acc = []
        async for token in chat_stream(messages):
            acc.append(token)
            yield token
        full = "".join(acc)
        if guardrail.is_offtopic_refusal(full):
            retry = await run_in_threadpool(
                rag.build_chat_messages,
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
        turns = await run_in_threadpool(store.get_chat_history, engagement_id, limit=limit)
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


def _compose_transcript(turns) -> str:
    parts = []
    for turn in turns:
        speaker = "User" if turn["role"] == "user" else "Fortis"
        parts.append(f"### {speaker}\n\n{_sanitize_transcript_turn(turn['content'])}")
    return "\n\n---\n\n".join(parts)


_FENCE_OPEN_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_HEADING_FORGE_RE = re.compile(r"^(\s*)(#{1,6})(\s|$)")
_HR_FORGE_RE = re.compile(r"^(\s*)([-*_])([-*_\s]{2,})$")


def _sanitize_transcript_turn(text: str) -> str:
    """Escapes forged speaker headers (leading `#`/hr lines) in a turn so a
    hostile reply cannot add fake `### User` sections to a transcript export;
    fenced code blocks are preserved verbatim and dangling fences are closed
    so a turn cannot swallow the following speaker header."""
    out = []
    fence_marker = None
    for line in text.splitlines() or []:
        m = _FENCE_OPEN_RE.match(line)
        if m is not None:
            marker = m.group(1)
            if fence_marker is None:
                fence_marker = marker[0] * len(marker)
            elif marker[0] == fence_marker[0] and len(marker) >= len(fence_marker):
                fence_marker = None
            out.append(line)
            continue
        if fence_marker is None:
            line = _HEADING_FORGE_RE.sub(r"\1\\\2\3", line)
            line = _HR_FORGE_RE.sub(r"\1\\\2\3", line)
        out.append(line)
    if fence_marker is not None:
        out.append(fence_marker)
    return "\n".join(out)


_MAX_EXPORT_CHARS = 1_000_000


def _render_chat_export(md_text: str, output_path: str, title: str, fmt: str) -> str:
    return chat_export.REPLY_RENDERERS[fmt](md_text, output_path, title)


@router.post("/export")
async def chat_export_endpoint(req: ChatExportRequest):
    engagement = store.get_engagement(req.engagement_id)
    if not engagement:
        raise HTTPException(404, f"Engagement '{store.clean(req.engagement_id)}' not found. Create or select one first.")
    fmt = (req.format or "").lower()
    if fmt not in chat_export.REPLY_RENDERERS:
        raise HTTPException(400, f"format must be one of {list(chat_export.REPLY_RENDERERS)}")

    try:
        turns = await run_in_threadpool(store.get_chat_history, req.engagement_id)
    except (KeyError, ValueError):
        raise HTTPException(404, f"Engagement '{store.clean(req.engagement_id)}' not found.")
    if req.message_id is not None:
        turn = next((t for t in turns if t.get("id") == req.message_id), None)
        if turn is None:
            raise HTTPException(404, "message not found in this engagement's history")
        md_text = turn["content"]
        message_count = 1
    elif req.latest_assistant or not req.transcript:
        turn = next((t for t in reversed(turns) if t["role"] == "assistant"), None)
        if turn is None:
            raise HTTPException(404, "No assistant reply in this engagement's history yet.")
        md_text = turn["content"]
        message_count = 1
    else:
        if not any((t["content"] or "").strip() for t in turns):
            raise HTTPException(400, "Nothing to export.")
        md_text = _compose_transcript(turns)
        message_count = len(turns)

    if not (md_text or "").strip():
        raise HTTPException(400, "Nothing to export.")
    if len(md_text) > _MAX_EXPORT_CHARS:
        raise HTTPException(
            400,
            f"Export too large: {len(md_text)} characters (limit {_MAX_EXPORT_CHARS}). "
            "Export individual turns instead.",
        )

    title = f"{engagement['client_name']} — {engagement['name']} — chat export"
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base = ri._FILE_BASE_RE.sub("-", (req.engagement_id or "chat").lower())[:40].strip("-") or "chat"
    filename = f"{base}_chat_{stamp}_{uuid.uuid4().hex[:6]}.{fmt}"
    output_path = os.path.join(settings.report_dir, filename)
    try:
        await run_in_threadpool(_render_chat_export, md_text, output_path, title, fmt)
    except Exception:
        try:
            os.remove(output_path)
        except OSError:
            pass
        raise HTTPException(500, "Export rendering failed.")
    return {
        "download_url": f"/report/download/{filename}",
        "filename": filename,
        "format": fmt,
        "message_count": message_count,
    }
