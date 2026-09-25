"""
Single-user orchestration layer for the desktop app.

Ports the Open WebUI pipeline's per-message behavior (engagement slash
commands, file ingestion, cybersecurity guardrail, report-intent detection,
refusal retry) to direct function calls into core — deliberately NOT over
HTTP, since self-requests from inside async endpoints deadlock the event
loop.
"""
import os
import re
from typing import Dict, List, Optional

from core import store, vectorstore, rag
from core.config import settings
from core.ingestion import extract_text, chunk_text_enriched
from core.report_intent import (
    detect_report_request,
    format_report_reply,
    is_stub_backend,
    stub_gate_message,
)
from core.routers.chat import run_structured_turn
from core.routers.report import generate_report_result
from core.structured import detect_structured_output_request
import engine

_LOCAL_USER = "local"

_CYBERSECURITY_KEYWORDS = [
    r"\b(security|cyber(?:security)?|infosec|application\s*security|appsec)\b",
    r"\b(vulnerab(?:ility|le)|exploit|att(?:ack|&?ck)|threat|malware|ransomware|phishing)\b",
    r"\b(risk|compliance|audit|governance|policy|incident|breach|compromise)\b",
    r"\b(nist|owasp|mitre|cis\s*controls?|iso\s*27001|soc\s*2|pci[\s-]*dss|gdpr|hipaa)\b",
    r"\b(firewall|ids|ips|siem|soc|encryption|authentication|authorization)\b",
    r"\b(access\s*control|iam|identity|privilege|zero\s*trust|vpn|tls|ssl)\b",
    r"\b(patch|hardening|configuration|misconfiguration|baseline)\b",
    r"\b(log|monitoring|detect|respond|recover|protect|identify)\b",
    r"\b(sql|sqli|injection|xss|csrf|ssrf|deserializ|payload|webshell|privilege\s*escap)\b",
    r"\b(iv|traversal|directory\s*traversal|code\s*injection|command\s*injection)\b",
    r"\b(reverse\s*shell|shell|buffer\s*overflow|malicious|trojan|rootkit|botnet)\b",
    r"\b(report|finding|remediat|recommendation|executive\s*summary|risk\s*rating)\b",
    r"\b(upload|document|config|code\s*review|architecture|network\s*diagram)\b",
]

_OFF_TOPIC_RESPONSE = (
    "I'm Fortis, a cybersecurity advisory assistant. I can only help with "
    "security-related topics such as:\n\n"
    "- Document, configuration, and code security reviews\n"
    "- Framework mapping (NIST CSF, OWASP Top 10, CIS Controls)\n"
    "- Risk assessment and remediation advice\n"
    "- Security report generation\n\n"
    "Please ask a cybersecurity question or upload a document for review."
)

_REFUSAL_MARKERS = (
    "i can only help with security-related topics",
    "please ask a cybersecurity question",
)

def _is_stub_backend() -> bool:
    return is_stub_backend()


def core_gate_message() -> str:
    return stub_gate_message()


def _run_async(coro):
    """Runs an async core function from sync code. When already inside an
    event loop (FastAPI endpoint), runs it on a worker thread instead of
    failing with 'asyncio.run() cannot be called from a running event loop'."""
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def is_cybersecurity_related(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in _CYBERSECURITY_KEYWORDS)


def is_offtopic_refusal(reply: str) -> bool:
    lowered = reply.lower()
    return any(m in lowered for m in _REFUSAL_MARKERS)


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

    # ---- the main turn --------------------------------------------------------

    def chat_turn(
        self,
        user_text: str,
        history: List[Dict[str, str]] = None,
        attachment_paths: List[str] = None,
        role: Optional[str] = None,
    ) -> Dict:
        command = self.handle_command(user_text)
        if command is not None:
            return {"kind": "command", "text": command}

        engagement = self.active_engagement()
        if not engagement:
            return {
                "kind": "no_engagement",
                "text": "No active engagement. Create one with `/new-engagement Client :: Name` "
                        "or the **+ New engagement** button in the sidebar.",
            }

        engagement_id = engagement["id"]
        notes = [self.ingest_file(p, engagement_id) for p in attachment_paths or []]
        upload_note = "\n".join(n for n in notes if n)

        if not upload_note and not is_cybersecurity_related(user_text):
            return {"kind": "offtopic", "text": _OFF_TOPIC_RESPONSE}

        fmt = detect_report_request(user_text)
        if fmt:
            return {"kind": "report", "text": self.generate_report(engagement_id, fmt, user_text)}

        intent = detect_structured_output_request(user_text)
        if intent:
            effective_role = role if role in rag.ROLE_PRESETS else None
            reply = _run_async(run_structured_turn(engagement_id, user_text, history or [], intent, effective_role))
            return {"kind": "chat", "text": reply, "upload_note": upload_note}

        reply = self._chat(engagement_id, user_text, history or [], role)
        if is_offtopic_refusal(reply) and is_cybersecurity_related(user_text):
            forced = (
                "This is a cybersecurity topic. Answer the user's question directly "
                "and completely. Do not refuse. Original request: " + user_text
            )
            reply = self._chat(engagement_id, forced, history or [], role)
        return {"kind": "chat", "text": reply, "upload_note": upload_note}

    def _chat(self, engagement_id: str, message: str, history: List[Dict[str, str]], role: Optional[str] = None) -> str:
        return _run_async(self._achat(engagement_id, message, history, role))

    async def _achat(self, engagement_id: str, message: str, history: List[Dict[str, str]], role: Optional[str] = None) -> str:
        from core.condense import condense_query
        from core.config import settings as core_settings
        retrieval_query = await condense_query(history, message, core_settings.rag_level)
        effective_role = role if role in rag.ROLE_PRESETS else None
        messages = rag.build_chat_messages(
            engagement_id, message, history, role=effective_role, retrieval_query=retrieval_query
        )
        return await engine.chat(messages)

    # ---- reports ----------------------------------------------------------------

    def generate_report(self, engagement_id: str, fmt: str, focus_text: str = "") -> str:
        result = _run_async(generate_report_result(engagement_id, fmt, focus_text))
        if "error" in result:
            return f"Report generation failed: {result['error']}"
        return format_report_reply(result, _is_stub_backend())

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
