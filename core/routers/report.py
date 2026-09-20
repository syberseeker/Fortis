import os
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..config import settings
from .. import store
from ..analysis import generate_security_report
from ..reports.docx_report import render_docx
from ..reports.pptx_report import render_pptx
from ..reports.pdf_report import render_pdf

router = APIRouter(prefix="/report", tags=["report"])

_RENDERERS = {
    "docx": (render_docx, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    "pptx": (render_pptx, "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    "pdf": (render_pdf, "application/pdf"),
}


class ReportRequest(BaseModel):
    engagement_id: str
    format: str = "docx"  # docx | pptx | pdf
    focus_instructions: str = ""


@router.post("/generate")
async def generate_report(req: ReportRequest):
    fmt = req.format.lower()
    if fmt not in _RENDERERS:
        raise HTTPException(400, f"format must be one of {list(_RENDERERS)}")

    engagement = store.get_engagement(req.engagement_id)
    if not engagement:
        raise HTTPException(404, f"Engagement '{req.engagement_id}' not found. Create or select one first.")

    try:
        report = await generate_security_report(req.engagement_id, req.focus_instructions)
    except ValueError as e:
        raise HTTPException(400, str(e))

    renderer, media_type = _RENDERERS[fmt]
    filename = f"{req.engagement_id}_{uuid.uuid4().hex[:6]}.{fmt}"
    output_path = os.path.join(settings.report_dir, filename)
    renderer(report, output_path)

    return {
        "download_url": f"/report/download/{filename}",
        "filename": filename,
        "engagement": {"client_name": engagement["client_name"], "engagement_name": engagement["name"]},
        "findings_count": len(report.findings),
        "overall_risk_rating": report.overall_risk_rating,
    }


@router.get("/download/{filename}")
async def download_report(filename: str):
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "Invalid filename")
    path = os.path.join(settings.report_dir, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "Report not found. Generate it first via /report/generate.")

    ext = filename.rsplit(".", 1)[-1]
    _, media_type = _RENDERERS.get(ext, (None, "application/octet-stream"))
    return FileResponse(path, media_type=media_type, filename=filename)
