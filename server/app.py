"""
Fortis Desktop server: the core FastAPI app (engagements, upload, chat,
report) plus a small /api layer for the UI (slash commands, model
management, report saving), and the static chat UI itself.

Run headless with:  python -m server.app --port 8757
(the desktop launcher in desktop/app.py imports create_server instead).
"""
import argparse
import os
import urllib.parse
import subprocess
import sys
import threading
import webbrowser

from fastapi import APIRouter, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core.config import settings
from core.main import app as core_app  # FastAPI app with all core routers
import engine
import engine.llama_engine
import engine.model_manager as model_manager
from engine.hardware import detect, suggest_tier, can_use_cuda, suggest_tiers
from server.orchestration import Orchestrator

PORT = int(os.getenv("FORTIS_PORT", "8757"))

orch = Orchestrator()

api = APIRouter(prefix="/api")


# ---- UI helpers --------------------------------------------------------------

class CommandRequest(BaseModel):
    text: str


@api.post("/command")
async def command(req: CommandRequest):
    """Slash-command passthrough from the UI (also usable headless)."""
    reply = orch.handle_command(req.text)
    if reply is None:
        raise HTTPException(400, "Not a recognized command. Try /engagements, /new-engagement, /use, /whoami, /close-engagement.")
    return {"text": reply}


# ---- model management ---------------------------------------------------------

@api.get("/model/status")
async def model_status():
    hw = detect()
    warning = await engine.check_model_available()
    hardware_info = engine.llama_engine.hardware_status(hw)
    hardware_info["can_cuda"] = can_use_cuda()
    return {
        "hardware": hardware_info,
        "suggested_tier": suggest_tier(hw),
        "suggested_tiers": suggest_tiers(hw),
        "tiers": model_manager.tier_status(),
        "active_model": model_manager.current_selection(),
        "loaded": engine.llama_engine._STATE["model"] is not None,
        "load_error": engine.llama_engine._STATE["error"],
        "warning": warning,
    }


class TierRequest(BaseModel):
    tier: str


@api.post("/model/download/{tier}")
async def model_download(tier: str):
    if tier not in model_manager.TIERS:
        raise HTTPException(400, f"Unknown tier '{tier}'")
    # already downloaded? just select it
    for m in model_manager.list_local_models():
        if f"-{tier}-" in m["name"] or f"-{tier}." in m["name"]:
            model_manager.set_selection(m["path"])
            return {"status": "already_downloaded", "path": m["path"]}
    prog = model_manager.download_progress()
    if not prog["done"] and prog["tier"] == tier:
        return {"status": "in_progress"}
    threading.Thread(target=_download_worker, args=(tier,), daemon=True).start()
    return {"status": "started"}


def _download_worker(tier: str):
    try:
        model_manager.download_tier(tier)
    except Exception as e:
        print(f"[model download] failed: {e}", file=sys.stderr)


@api.get("/model/download/progress")
async def model_download_progress():
    return model_manager.download_progress()


@api.get("/report/progress")
async def report_progress():
    from core import progress
    return progress.snapshot()


class SelectRequest(BaseModel):
    tier: str = ""
    path: str = ""


@api.post("/model/select")
async def model_select(req: SelectRequest):
    if req.tier:
        for m in model_manager.list_local_models():
            name_lower = m["name"].lower()
            if req.tier in model_manager.TIERS:
                match_patterns = model_manager.TIERS[req.tier].get("match", [f"-{req.tier}-", f"-{req.tier}."])
                if any(p in name_lower for p in match_patterns):
                    req.path = m["path"]
                    break
            elif f"-{req.tier}-" in name_lower or f"-{req.tier}." in name_lower:
                req.path = m["path"]
                break
        else:
            raise HTTPException(404, f"Tier '{req.tier}' is not downloaded yet.")
    if not req.path or not os.path.isfile(req.path):
        raise HTTPException(404, "Model file not found.")
    basename = os.path.basename(req.path).lower()
    for tier_id, info in model_manager.TIERS.items():
        if "match" in info and any(m in basename for m in info["match"]):
            if "n_ctx" in info:
                settings.n_ctx = info["n_ctx"]
            break
    model_manager.set_selection(req.path)
    engine.llama_engine.unload()
    return {"status": "selected", "path": req.path, "n_ctx": settings.n_ctx}


# ---- reports -------------------------------------------------------------------

@api.post("/report/save")
async def report_save(req: dict):
    engagement_id = req.get("engagement_id", "")
    fmt = req.get("format", "docx")
    focus = req.get("focus_instructions", "")
    allow_stub = bool(req.get("allow_stub", False))
    result = orch.generate_report_download(
        engagement_id, fmt, focus, allow_stub=allow_stub, enforce_stub_gate=True
    )
    if "error" in result:
        raise HTTPException(409 if result.get("stub_gate") else 400, result["error"])
    return result


class OpenExternalRequest(BaseModel):
    url: str


@api.post("/open-external")
async def open_external(req: OpenExternalRequest):
    url = req.url.strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        raise HTTPException(400, "URL must be a valid http:// or https:// URL")
    threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    return {"status": "opened", "url": url}


@api.post("/open-path")
async def open_path(req: dict):
    path = req.get("path", "")
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Path not found")
    try:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return {"status": "opened"}
    except Exception as e:
        raise HTTPException(500, str(e))


@api.post("/reports/open-folder")
async def open_reports_folder():
    try:
        if sys.platform == "win32":
            os.startfile(os.path.normpath(settings.report_dir))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", settings.report_dir])
        else:
            subprocess.Popen(["xdg-open", settings.report_dir])
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"path": settings.report_dir}


# ---- wiring ---------------------------------------------------------------------

app = core_app
app.include_router(api)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")


@app.get("/")
async def index():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


def create_server(host: str = "127.0.0.1", port: int = PORT):
    import uvicorn
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    return uvicorn.Server(config)


def main():
    parser = argparse.ArgumentParser(description="Fortis desktop server (headless)")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
