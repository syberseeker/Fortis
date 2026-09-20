from typing import List, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import rag, store
from ..config import settings
from ..condense import condense_query
from ..llm import chat, chat_stream

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    engagement_id: str
    message: str
    history: Optional[List[Dict[str, str]]] = None


def _require_engagement(engagement_id: str) -> None:
    if not store.get_engagement(engagement_id):
        raise HTTPException(404, f"Engagement '{engagement_id}' not found. Create or select one first.")


@router.post("")
async def chat_endpoint(req: ChatRequest):
    _require_engagement(req.engagement_id)
    retrieval_query = await condense_query(req.history, req.message, settings.rag_level)
    messages = rag.build_chat_messages(req.engagement_id, req.message, req.history, retrieval_query=retrieval_query)
    reply = await chat(messages)
    return {"reply": reply}


@router.post("/stream")
async def chat_stream_endpoint(req: ChatRequest):
    _require_engagement(req.engagement_id)
    retrieval_query = await condense_query(req.history, req.message, settings.rag_level)
    messages = rag.build_chat_messages(req.engagement_id, req.message, req.history, retrieval_query=retrieval_query)

    async def generator():
        async for token in chat_stream(messages):
            yield token

    return StreamingResponse(generator(), media_type="text/plain")
