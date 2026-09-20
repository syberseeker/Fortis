from typing import List, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import rag, store
from ..config import settings
from ..condense import condense_query
from ..rag import ROLE_PRESETS
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
        raise HTTPException(404, f"Engagement '{engagement_id}' not found. Create or select one first.")


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
    role = req.role if req.role in ROLE_PRESETS else None
    retrieval_query = await condense_query(req.history, req.message, settings.rag_level)
    intent = detect_structured_output_request(req.message)
    if intent:
        reply = await run_structured_turn(req.engagement_id, req.message, req.history or [], intent, role)
        return {"reply": reply}
    messages = rag.build_chat_messages(req.engagement_id, req.message, req.history, role=role, retrieval_query=retrieval_query)
    reply = await chat(messages)
    return {"reply": reply}


@router.post("/stream")
async def chat_stream_endpoint(req: ChatRequest):
    _require_engagement(req.engagement_id)
    role = req.role if req.role in ROLE_PRESETS else None
    retrieval_query = await condense_query(req.history, req.message, settings.rag_level)
    intent = detect_structured_output_request(req.message)
    if intent:
        reply = await run_structured_turn(req.engagement_id, req.message, req.history or [], intent, role)
        async def generator():
            yield reply
        return StreamingResponse(generator(), media_type="text/plain")
    messages = rag.build_chat_messages(req.engagement_id, req.message, req.history, role=role, retrieval_query=retrieval_query)

    async def generator():
        async for token in chat_stream(messages):
            yield token

    return StreamingResponse(generator(), media_type="text/plain")
