"""
Desktop orchestration layer: engagement slash commands, per-user helpers,
and file ingestion — thin wrappers over core, deliberately NOT going over
HTTP. The per-message chat pipeline (guardrail, report intent, structured
output, persistence) lives in core.routers.chat, which serves both the UI
and any API client.
"""
import os
import re
from typing import Dict, Optional

from core import store, vectorstore
from core.ingestion import extract_text, chunk_text_enriched
from core.report_intent import format_report_reply
from core.report_service import generate_report_result

_LOCAL_USER = "local"

_RUN_EXECUTOR = None


def _run_async(coro):
    """Runs an async core function from sync code. When already inside an
    event loop (FastAPI endpoint), runs it on a shared worker thread instead
    of failing with 'asyncio.run() cannot be called from a running event
    loop'. One executor is reused for the process lifetime."""
    import asyncio
    global _RUN_EXECUTOR
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    if _RUN_EXECUTOR is None:
        _RUN_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="fortis-orch"
        )
    return _RUN_EXECUTOR.submit(asyncio.run, coro).result()


def _is_stub_backend() -> bool:
    from core.report_intent import is_stub_backend

    return is_stub_backend()


def core_gate_message() -> str:
    from core.report_intent import stub_gate_message

    return stub_gate_message()


class Orchestrator:

    # ---- engagement helpers (direct store calls, no HTTP) -------------------

    def active_engagement(self) -> Optional[Dict]:
        eng = store.get_active_engagement(_LOCAL_USER)
        if eng:
            eng = dict(eng)
            eng["files"] = vectorstore.list_engagement_files(eng["id"])
        return eng

    def create_engagement(self, client_name: str, engagement_name: str, notes: str = "") -> Dict:
        eng = store.create_engagement(client_name, engagement_name, notes)
        store.set_active_engagement(_LOCAL_USER, eng["id"])
        return eng

    def activate(self, engagement_id: str) -> Dict:
        if not store.get_engagement(engagement_id):
            raise KeyError(engagement_id)
        store.set_active_engagement(_LOCAL_USER, engagement_id)
        return store.get_engagement(engagement_id)

    # ---- slash commands ------------------------------------------------------

    def handle_command(self, text: str) -> Optional[str]:
        t = text.strip()
        low = t.lower()
        if low.startswith("/engagements"):
            return self._list_engagements()
        if low.startswith("/new-engagement"):
            return self._create_engagement_cmd(t[len("/new-engagement"):])
        if low.startswith("/use"):
            m = re.match(r"/use\s+(\S+)", t, re.IGNORECASE)
            if not m:
                return "Usage: `/use <engagement_id>` — see `/engagements` for IDs."
            try:
                eng = self.activate(m.group(1))
            except KeyError:
                return f"No engagement with ID `{m.group(1)}`. See `/engagements`."
            return f"Switched to **{eng['client_name']} — {eng['name']}**."
        if low.startswith("/whoami"):
            return self._whoami()
        if low.startswith("/close-engagement"):
            return self._close_active()
        if low.startswith("/report"):
            return self._report_cmd(t[len("/report"):])
        return None

    def _report_cmd(self, body: str) -> str:
        parts = body.strip().split()
        fmt = (parts[0].lower() if parts else "docx")
        if fmt in ("pptx", "powerpoint"):
            fmt = "pptx"
        if fmt not in ("docx", "pptx", "pdf"):
            fmt = "docx"
        eng = self.active_engagement()
        if not eng:
            return "No active engagement. Use `/new-engagement Client :: Name` or the + button."
        if _is_stub_backend():
            from core.report_intent import chat_stub_gate_reply

            return chat_stub_gate_reply()
        result = self.generate_report_download(eng["id"], fmt)
        if "error" in result:
            return f"Report generation failed: {result['error']}"
        return format_report_reply(result, _is_stub_backend())

    def _list_engagements(self) -> str:
        engagements = store.list_engagements()
        if not engagements:
            return "No engagements yet. Create one with `/new-engagement Client Name :: Engagement Name` or the + button."
        lines = ["**Clients & engagements:**"]
        for e in engagements:
            lines.append(f"- `{e['id']}` — **{e['client_name']}** / {e['name']} ({e['status']})")
        lines.append("\nSwitch with `/use <engagement_id>`.")
        return "\n".join(lines)

    def _create_engagement_cmd(self, body: str) -> str:
        parts = [p.strip() for p in body.split("::")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            return (
                "Usage: `/new-engagement Client Name :: Engagement Name [:: notes]`\n\n"
                "Example: `/new-engagement Acme Corp :: Q3 2026 Config Review`"
            )
        eng = self.create_engagement(parts[0], parts[1], parts[2] if len(parts) > 2 else "")
        client = store.get_engagement(eng["id"])["client_name"]
        return f"Created and activated **{client} — {eng['name']}** (`{eng['id']}`)."

    def _whoami(self) -> str:
        eng = self.active_engagement()
        if not eng:
            return "No active engagement. Use `/new-engagement Client :: Name` or the + button."
        files = eng.get("files") or []
        files_line = ", ".join(files) if files else "none yet"
        return (
            f"**Active engagement:** {eng['client_name']} — {eng['name']}\n"
            f"- ID: `{eng['id']}`\n- Status: {eng['status']}\n"
            f"- Documents uploaded: {files_line}"
        )

    def _close_active(self) -> str:
        eng = self.active_engagement()
        if not eng:
            return "No active engagement to close."
        store.set_engagement_status(eng["id"], "closed")
        return f"Closed **{eng['client_name']} — {eng['name']}**."

    # ---- uploads -------------------------------------------------------------

    def ingest_file(self, file_path: str, engagement_id: str) -> str:
        filename = os.path.basename(file_path)
        if not store.get_engagement(engagement_id):
            return f"Failed to ingest {filename}: no such engagement."
        try:
            text = extract_text(file_path)
        except Exception as e:
            return f"Failed to parse {filename}: {e}"
        if not text.strip():
            return f"Failed to ingest {filename}: no extractable text."
        chunks = chunk_text_enriched(text, filename)
        n = vectorstore.add_user_document_chunks(engagement_id, filename, chunks)
        return f"Indexed **{filename}** ({n} chunks)."

    # ---- reports ----------------------------------------------------------------

    async def _agenerate_report_download(
        self,
        engagement_id: str,
        fmt: str,
        focus_text: str = "",
        allow_stub: bool = False,
        enforce_stub_gate: bool = False,
    ) -> Dict:
        # enforce_stub_gate is accepted for call-site compatibility; the gate
        # is always enforced inside generate_report_result now.
        return await generate_report_result(engagement_id, fmt, focus_text, allow_stub=allow_stub)

    def generate_report_download(
        self,
        engagement_id: str,
        fmt: str,
        focus_text: str = "",
        allow_stub: bool = False,
        enforce_stub_gate: bool = False,
    ) -> Dict:
        return _run_async(
            self._agenerate_report_download(
                engagement_id, fmt, focus_text, allow_stub=allow_stub, enforce_stub_gate=enforce_stub_gate
            )
        )
