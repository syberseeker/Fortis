import os
import re
import uuid

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from ..config import settings
from ..ingestion import extract_text, chunk_text_enriched
from .. import vectorstore, store

router = APIRouter(prefix="/upload", tags=["upload"])

MAX_FILE_SIZE_MB = 40
_FILENAME_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]+")
_MAX_FILENAME_LENGTH = 120


def _clean_filename(name: str) -> str:
    """Filenames flow into chroma metadata, the disk filename, and prompt
    context. Hostile input can carry unpaired surrogates (a JSON \\ud800
    escape decodes to one), which utf-8 encoding rejects outright; replace
    them so a crafted filename cannot 500 the upload. Valid text is
    byte-identical."""
    return store.clean(name or "")


def _safe_disk_filename(name: str) -> str:
    base = os.path.basename((name or "").replace("\\", "/"))
    base = _FILENAME_CONTROL_CHARS_RE.sub("", base).strip()
    return base[:_MAX_FILENAME_LENGTH] or "upload.bin"


@router.post("")
async def upload_file(
    file: UploadFile = File(...),
    engagement_id: str = Form(...),
):
    if not store.get_engagement(engagement_id):
        raise HTTPException(404, f"Engagement '{store.clean(engagement_id)}' not found. Create or select one first.")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(413, f"File exceeds {MAX_FILE_SIZE_MB}MB limit ({size_mb:.1f}MB).")

    safe_name = f"{uuid.uuid4().hex[:8]}_{_safe_disk_filename(_clean_filename(file.filename))}"
    save_path = os.path.join(settings.upload_dir, safe_name)
    with open(save_path, "wb") as f:
        f.write(contents)

    try:
        text = extract_text(save_path)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {type(e).__name__}: {e}")

    if not text.strip():
        raise HTTPException(422, "No extractable text found in this file.")

    chunks = chunk_text_enriched(text, _clean_filename(file.filename))
    n_added = vectorstore.add_user_document_chunks(engagement_id, _clean_filename(file.filename), chunks)

    return {
        "filename": file.filename,
        "chunks_indexed": n_added,
        "characters_extracted": len(text),
        "engagement_id": engagement_id,
    }


@router.get("/engagement/{engagement_id}/files")
async def engagement_files(engagement_id: str):
    return {"engagement_id": engagement_id, "files": vectorstore.list_engagement_files(engagement_id)}


@router.delete("/engagement/{engagement_id}")
async def clear_engagement_documents(engagement_id: str):
    """Clears uploaded documents/vectors for an engagement without deleting the
    engagement record itself (client history, past reports stay intact)."""
    vectorstore.clear_engagement(engagement_id)
    return {"engagement_id": engagement_id, "status": "documents_cleared"}
