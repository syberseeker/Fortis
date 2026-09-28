import os
import json
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from .. import store, vectorstore
from ..config import settings

router = APIRouter(prefix="/engagements", tags=["engagements"])


# ---- Clients --------------------------------------------------------------

class ClientCreate(BaseModel):
    name: str


@router.post("/clients")
async def create_client(req: ClientCreate):
    return store.get_or_create_client(req.name)


@router.get("/clients")
async def list_clients():
    return store.list_clients()


# ---- Engagements ------------------------------------------------------------

class EngagementCreate(BaseModel):
    client_name: str
    engagement_name: str
    notes: str = ""


@router.post("")
async def create_engagement(req: EngagementCreate):
    if not req.client_name.strip() or not req.engagement_name.strip():
        raise HTTPException(400, "client_name and engagement_name are both required.")
    return store.create_engagement(req.client_name, req.engagement_name, req.notes)


@router.get("")
async def list_engagements(client_id: Optional[str] = None):
    return store.list_engagements(client_id)


@router.get("/{engagement_id}")
async def get_engagement(engagement_id: str):
    engagement = store.get_engagement(engagement_id)
    if not engagement:
        raise HTTPException(404, "Engagement not found.")
    engagement = dict(engagement)
    engagement["files"] = vectorstore.list_engagement_files(engagement_id)
    return engagement


class EngagementStatusUpdate(BaseModel):
    status: str  # active | closed


@router.patch("/{engagement_id}/status")
async def update_status(engagement_id: str, req: EngagementStatusUpdate):
    if not store.get_engagement(engagement_id):
        raise HTTPException(404, "Engagement not found.")
    if req.status not in ("active", "closed"):
        raise HTTPException(400, "status must be 'active' or 'closed'.")
    store.set_engagement_status(engagement_id, req.status)
    return {"engagement_id": engagement_id, "status": req.status}


@router.delete("/{engagement_id}")
async def delete_engagement(engagement_id: str):
    if not store.get_engagement(engagement_id):
        raise HTTPException(404, "Engagement not found.")
    vectorstore.clear_engagement(engagement_id)
    store.delete_engagement(engagement_id)
    return {"engagement_id": engagement_id, "status": "deleted"}


@router.get("/{engagement_id}/export")
async def export_engagement(engagement_id: str):
    """Bundles the engagement's metadata, chat transcript, indexed document
    list, and generated report files (matched by filename slug prefix) into a
    downloadable zip archive."""
    import io
    import zipfile
    from datetime import datetime

    engagement = store.get_engagement(engagement_id)
    if not engagement:
        raise HTTPException(404, "Engagement not found.")

    try:
        turns = store.get_chat_history(engagement_id)
    except (KeyError, ValueError):
        turns = []
    documents = vectorstore.list_engagement_files(engagement_id)

    slug = engagement_id.lower()
    report_names = []
    if os.path.isdir(settings.report_dir):
        report_names = sorted(
            f for f in os.listdir(settings.report_dir)
            if f.lower().startswith(slug) and os.path.isfile(os.path.join(settings.report_dir, f))
        )

    bundle = {
        "fortis_export_version": 1,
        "exported_at": datetime.utcnow().isoformat(),
        "engagement": engagement,
        "documents": documents,
        "chat_turns": turns,
        "report_files": report_names,
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("engagement.json", json.dumps(bundle, indent=2, ensure_ascii=False))
        for name in report_names:
            zf.write(os.path.join(settings.report_dir, name), f"reports/{name}")

    stamp = datetime.now().strftime("%Y%m%d")
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{slug}-export-{stamp}.zip"'},
    )


# ---- Active engagement per user --------------------------------------------

class ActiveEngagementSet(BaseModel):
    user_id: str
    engagement_id: str


@router.post("/active")
async def set_active_engagement(req: ActiveEngagementSet):
    if not store.get_engagement(req.engagement_id):
        raise HTTPException(404, "Engagement not found.")
    store.set_active_engagement(req.user_id, req.engagement_id)
    return store.get_active_engagement(req.user_id)


@router.get("/active/{user_id}")
async def get_active_engagement(user_id: str):
    engagement = store.get_active_engagement(user_id)
    if not engagement:
        raise HTTPException(404, "No active engagement set for this user.")
    return engagement


@router.delete("/active/{user_id}")
async def clear_active_engagement(user_id: str):
    store.clear_active_engagement(user_id)
    return {"user_id": user_id, "status": "cleared"}
