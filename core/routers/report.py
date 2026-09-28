import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import report_intent
from ..config import settings
from ..report_service import generate_report_result

router = APIRouter(prefix="/report", tags=["report"])


def _media_type_for(fmt: str) -> str:
    return {
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "pdf": "application/pdf",
    }.get(fmt, "application/octet-stream")


class ReportRequest(BaseModel):
    engagement_id: str
    format: str = "docx"  # docx | pptx | pdf
    focus_instructions: str = ""
    allow_stub: bool = False


def _is_stub_backend() -> bool:
    return report_intent.is_stub_backend()


_STUB_GATE_MSG = report_intent.stub_gate_message()


def _enforce_stub_gate(allow_stub: bool) -> None:
    """Router-level gate keeps this endpoint's 409 contract and the
    report_router._is_stub_backend test seam; the service re-checks via
    report_intent."""
    if _is_stub_backend() and not allow_stub:
        raise HTTPException(409, _STUB_GATE_MSG)


@router.post("/generate")
async def generate_report(req: ReportRequest):
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
    return FileResponse(path, media_type=_media_type_for(ext), filename=filename)
