import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import report_intent
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
for _fmt, (_renderer, _media) in _RENDERERS.items():
    report_intent.register_renderer(_fmt, _renderer)


def _media_type_for(fmt: str) -> str:
    return _RENDERERS.get(fmt, (None, "application/octet-stream"))[1]


class ReportRequest(BaseModel):
    engagement_id: str
    format: str = "docx"  # docx | pptx | pdf
    focus_instructions: str = ""
    allow_stub: bool = False


def _is_stub_backend() -> bool:
    return report_intent.is_stub_backend()


_STUB_GATE_MSG = report_intent.stub_gate_message()


def _enforce_stub_gate(allow_stub: bool) -> None:
    """Single chokepoint for the placeholder-report gate; every report entry
    path (HTTP router, /api/report/save, chat intent) must call this."""
    if _is_stub_backend() and not allow_stub:
        raise HTTPException(409, _STUB_GATE_MSG)


async def generate_report_result(
    engagement_id: str,
    fmt: str,
    focus_text: str = "",
    allow_stub: bool = False,
) -> dict:
    """Single builder shared by /report/generate, the chat report intent, and
    the orchestrator: validates format/engagement, runs the analysis, renders
    into settings.report_dir, and returns the download metadata. Error results
    carry {"error": str, "status": int}."""
    fmt = (fmt or "").lower()
    if fmt not in _RENDERERS:
        return {"error": f"format must be one of {list(_RENDERERS)}", "status": 400}
    engagement = store.get_engagement(engagement_id)
    if not engagement:
        return {
            "error": f"Engagement '{engagement_id}' not found. Create or select one first.",
            "status": 404,
        }
    try:
        report_intent.enforce_stub_gate(allow_stub)
        report = await generate_security_report(engagement_id, focus_text)
        filename = report_intent.render_to_report_dir(report, fmt, engagement_id)
    except ValueError as e:
        message = str(e)
        is_gate = "allow_stub" in message
        return {"error": message, "status": 409 if is_gate else 400, "stub_gate": is_gate}
    return {
        "download_url": f"/report/download/{filename}",
        "filename": filename,
        "format": fmt,
        "findings_count": len(report.findings),
        "overall_risk_rating": report.overall_risk_rating,
        "engagement": {
            "client_name": engagement["client_name"],
            "engagement_name": engagement["name"],
        },
    }


@router.post("/generate")
async def generate_report(req: ReportRequest):
    # Router-level gate keeps this endpoint's own seam (and its 409 contract);
    # the builder call then runs with the gate satisfied.
    _enforce_stub_gate(req.allow_stub)
    result = await generate_report_result(
        req.engagement_id,
        req.format,
        req.focus_instructions,
        allow_stub=True,
    )
    if "error" in result:
        raise HTTPException(result.get("status", 400), result["error"])

    return result


@router.get("/download/{filename}")
async def download_report(filename: str):
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "Invalid filename")
    path = os.path.join(settings.report_dir, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "Report not found. Generate it first via /report/generate.")

    ext = filename.rsplit(".", 1)[-1]
    media_type = _media_type_for(ext)
    return FileResponse(path, media_type=media_type, filename=filename)
