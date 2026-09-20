"""
title: Cybersecurity Advisor (Fortis)
author: cyber-advisor
version: 0.3.0
description: >
  Routes chat + file uploads to the local RAG cybersecurity backend, scoped
  to a persistent client/engagement (not a throwaway chat session), and
  exposes report generation as a DOCX/PPTX/PDF download.

  Includes cybersecurity-only guardrails: off-topic queries are rejected
  before reaching the LLM to keep the assistant focused on security advisory.

Engagement commands (type these as a chat message):
  /engagements                          list all clients & engagements
  /new-engagement Client Name :: Engagement Name [:: notes]
                                         create (or reuse) an engagement and
                                         make it active for you
  /use <engagement_id>                  switch your active engagement
  /whoami                               show your currently active engagement
  /close-engagement                     mark the active engagement closed

Everything else (chat messages, file uploads, "generate a docx report")
applies to whichever engagement you currently have active. The active
engagement is stored backend-side per Open WebUI user, so it survives
across chats, browser restarts, and container restarts -- switching Open
WebUI conversations does NOT reset your documents or history the way a
per-chat-session design would.
"""

import re
from typing import List, Dict, Generator, Iterator, Union, Optional

import requests
from pydantic import BaseModel, Field


# ── Cybersecurity topic guardrail ────────────────────────────────────────────
# Keywords and patterns that indicate a cybersecurity-related query.
# The check is intentionally broad; borderline queries are allowed through
# because the RAG context + persona will handle nuance.
_CYBERSECURY_KEYWORDS = [
    # core security domains
    r"\b(security|cyber(?:security)?|infosec|application\s*security|appsec)\b",
    r"\b(vulnerab(?:ility|le)|exploit|att(?:ack|&?ck)|threat|malware|ransomware|phishing)\b",
    r"\b(risk|compliance|audit|governance|policy|incident|breach|compromise)\b",
    # frameworks & standards
    r"\b(nist|owasp|mitre|cis\s*controls?|iso\s*27001|soc\s*2|pci[\s-]*dss|gdpr|hipaa)\b",
    # technical security
    r"\b(firewall|ids|ips|siem|soc|encryption|authentication|authorization)\b",
    r"\b(access\s*control|iam|identity|privilege|zero\s*trust|vpn|tls|ssl)\b",
    r"\b(patch|hardening|configuration|misconfiguration|baseline)\b",
    r"\b(log|monitoring|detect|respond|recover|protect|identify)\b",  # NIST CSF functions
    # web / app exploitation
    r"\b(sql|sqli|injection|xss|csrf|ssrf|deserializ|payload|webshell|privilege\s*escap)\b",
    r"\b(iv|traversal|directory\s*traversal|code\s*injection|command\s*injection)\b",
    r"\b(reverse\s*shell|shell|buffer\s*overflow|malicious|trojan|rootkit|botnet)\b",
    # report / engagement context
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


def _is_cybersecurity_related(text: str) -> bool:
    """Return True if the text is plausibly about cybersecurity."""
    lower = text.lower()
    for pattern in _CYBERSECURY_KEYWORDS:
        if re.search(pattern, lower):
            return True
    return False


class Pipeline:
    class Valves(BaseModel):
        BACKEND_URL: str = Field(
            default="http://backend:8010",
            description="Base URL of the cyber-advisor FastAPI backend. "
                        "Use http://localhost:8010 if running Open WebUI outside Docker.",
        )
        DEFAULT_REPORT_FORMAT: str = Field(
            default="docx", description="docx | pptx | pdf"
        )
        REQUEST_TIMEOUT: int = Field(default=300)

    def __init__(self):
        self.id = "cyber_advisor"
        self.name = "Cybersecurity Advisor (Fortis)"
        self.valves = self.Valves()

    def pipes(self) -> List[Dict[str, str]]:
        return [{"id": "fortis-cyber-advisor", "name": self.name}]

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    def pipe(
        self,
        body: Dict,
        __user__: Optional[Dict] = None,
        __files__: Optional[List[Dict]] = None,
        __id__: Optional[str] = None,
        **kwargs,
    ) -> Union[str, Generator, Iterator]:
        user_id = (__user__ or {}).get("id", "anon")
        messages = body.get("messages", [])
        if not messages:
            return "No message received."

        latest = messages[-1]
        user_text = self._extract_text(latest).strip()

        # 1. Engagement management commands take priority over everything else.
        command_reply = self._handle_engagement_command(user_id, user_text)
        if command_reply is not None:
            return command_reply

        # 2. Resolve the active engagement. Required before uploads/chat/reports.
        engagement = self._get_active_engagement(user_id)
        if not engagement:
            return (
                "You don't have an active engagement yet. Start one with:\n\n"
                "`/new-engagement Client Name :: Engagement Name`\n\n"
                "or switch to an existing one — see `/engagements` for the list."
            )

        engagement_id = engagement["id"]

        # 3. Ingest any newly attached files into the backend RAG store.
        upload_notes = self._ingest_files(__files__, engagement_id)

        # 4. Cybersecurity topic guardrail — reject off-topic queries.
        if not _is_cybersecurity_related(user_text):
            return _OFF_TOPIC_RESPONSE

        # 5. Report-generation intent detection.
        report_format = self._detect_report_request(user_text)
        if report_format:
            return self._generate_report(engagement_id, report_format, user_text)

        # 6. Otherwise, normal RAG chat turn.
        history = self._to_backend_history(messages[:-1])
        reply = self._chat(engagement_id, user_text, history)
        reply = self._retry_refusal(reply, user_text, engagement_id, history)
        reply = self._postprocess_reply(reply, user_text, engagement_id, history)

        header = f"_[{engagement['client_name']} — {engagement['name']}]_\n\n"
        if upload_notes:
            reply = f"{upload_notes}\n\n{reply}"
        return header + reply

    # ---- Engagement commands -------------------------------------------------

    def _handle_engagement_command(self, user_id: str, text: str) -> Optional[str]:
        if text.lower().startswith("/engagements"):
            return self._list_engagements()

        if text.lower().startswith("/new-engagement"):
            return self._create_and_activate_engagement(user_id, text)

        if text.lower().startswith("/use"):
            match = re.match(r"/use\s+(\S+)", text, re.IGNORECASE)
            if not match:
                return "Usage: `/use <engagement_id>` — see `/engagements` for IDs."
            return self._activate_engagement(user_id, match.group(1))

        if text.lower().startswith("/whoami"):
            engagement = self._get_active_engagement(user_id)
            if not engagement:
                return "No active engagement set. Use `/new-engagement Client :: Engagement` to start one."
            files = engagement.get("files", [])
            files_line = ", ".join(files) if files else "none yet"
            return (
                f"**Active engagement:** {engagement['client_name']} — {engagement['name']}\n"
                f"- ID: `{engagement['id']}`\n"
                f"- Status: {engagement['status']}\n"
                f"- Documents uploaded: {files_line}"
            )

        if text.lower().startswith("/close-engagement"):
            engagement = self._get_active_engagement(user_id)
            if not engagement:
                return "No active engagement to close."
            try:
                resp = requests.patch(
                    f"{self.valves.BACKEND_URL}/engagements/{engagement['id']}/status",
                    json={"status": "closed"},
                    timeout=self.valves.REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                return f"Closed engagement **{engagement['client_name']} — {engagement['name']}**."
            except Exception as e:
                return f"⚠️ Could not close engagement: {e}"

        return None

    def _list_engagements(self) -> str:
        try:
            resp = requests.get(
                f"{self.valves.BACKEND_URL}/engagements",
                timeout=self.valves.REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            engagements = resp.json()
        except Exception as e:
            return f"⚠️ Could not reach the backend: {e}"

        if not engagements:
            return "No engagements yet. Create one with `/new-engagement Client Name :: Engagement Name`."

        lines = ["**Clients & engagements:**"]
        for e in engagements:
            lines.append(
                f"- `{e['id']}` — **{e['client_name']}** / {e['name']} ({e['status']})"
            )
        lines.append("\nSwitch with `/use <engagement_id>`.")
        return "\n".join(lines)

    def _create_and_activate_engagement(self, user_id: str, text: str) -> str:
        body = text[len("/new-engagement"):].strip()
        parts = [p.strip() for p in body.split("::")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            return (
                "Usage: `/new-engagement Client Name :: Engagement Name [:: notes]`\n\n"
                "Example: `/new-engagement Acme Corp :: Q3 2026 Config Review`"
            )
        client_name, engagement_name = parts[0], parts[1]
        notes = parts[2] if len(parts) > 2 else ""

        try:
            resp = requests.post(
                f"{self.valves.BACKEND_URL}/engagements",
                json={"client_name": client_name, "engagement_name": engagement_name, "notes": notes},
                timeout=self.valves.REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            engagement = resp.json()

            activate_resp = requests.post(
                f"{self.valves.BACKEND_URL}/engagements/active",
                json={"user_id": user_id, "engagement_id": engagement["id"]},
                timeout=self.valves.REQUEST_TIMEOUT,
            )
            activate_resp.raise_for_status()
        except Exception as e:
            return f"⚠️ Could not create engagement: {e}"

        return (
            f"✅ Created and activated engagement **{client_name} — {engagement_name}**.\n\n"
            f"Upload documents now, or ask questions — everything you do in this chat (and "
            f"any other chat) will apply to this engagement until you `/use` a different one."
        )

    def _activate_engagement(self, user_id: str, engagement_id: str) -> str:
        try:
            resp = requests.post(
                f"{self.valves.BACKEND_URL}/engagements/active",
                json={"user_id": user_id, "engagement_id": engagement_id},
                timeout=self.valves.REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            engagement = resp.json()
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return f"⚠️ No engagement with ID `{engagement_id}`. See `/engagements` for valid IDs."
            return f"⚠️ Could not switch engagement: {e.response.text[:300]}"
        except Exception as e:
            return f"⚠️ Could not reach the backend: {e}"

        return f"✅ Switched to **{engagement['client_name']} — {engagement['name']}**."

    def _get_active_engagement(self, user_id: str) -> Optional[Dict]:
        try:
            resp = requests.get(
                f"{self.valves.BACKEND_URL}/engagements/active/{user_id}",
                timeout=self.valves.REQUEST_TIMEOUT,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    # ---- Table post-processing --------------------------------------------

    _TABLE_REQUEST_RE = re.compile(
        r"\b(table|matrix|compare|comparison|vs\.?|chart|list.*column|"
        r"create.*list|numbered|markdown.*format|bar\s*chart)\b",
        re.IGNORECASE,
    )

    def _is_table_request(self, text: str) -> bool:
        return bool(self._TABLE_REQUEST_RE.search(text))

    def _is_broken_table(self, reply: str) -> bool:
        stripped = reply.strip()
        if len(stripped) < 50:
            return True
        if stripped.startswith("|"):
            lines = [l for l in stripped.splitlines() if l.strip()]
            has_separator = any(re.match(r"^\|[\s\-:|]+\|$", l) for l in lines)
            if len(lines) < 3 or not has_separator:
                return True
        return False

    def _postprocess_reply(self, reply: str, user_text: str, engagement_id: str, history: List[Dict[str, str]]) -> str:
        if not self._is_table_request(user_text):
            return reply
        if not self._is_broken_table(reply):
            return reply
        enhanced = (
            "Format your ENTIRE response as a proper markdown table. "
            "Requirements: (1) first row is the header with column names, "
            "(2) second row is a separator like |---|---|---|, "
            "(3) remaining rows are data. "
            "Every row must start and end with |. "
            "Do NOT include any text outside the table. "
            "Original request: " + user_text
        )
        return self._chat(engagement_id, enhanced, history)

    def _is_offtopic_refusal(self, reply: str) -> bool:
        """True if the model fell back to the generic off-topic refusal message,
        even though the query already passed the cybersecurity guardrail."""
        lowered = reply.lower()
        return (
            "i can only help with security-related topics" in lowered
            or "please ask a cybersecurity question" in lowered
        )

    def _retry_refusal(self, reply: str, user_text: str, engagement_id: str, history: List[Dict[str, str]]) -> str:
        """If the model wrongly refuses a query that passed the guardrail, retry
        once with an explicit instruction to answer the security topic."""
        if not self._is_offtopic_refusal(reply):
            return reply
        forced = (
            "This is a cybersecurity topic. Answer the user's question directly "
            "and completely. Do not refuse, do not say you can only help with "
            "cybersecurity topics. Original request: " + user_text
        )
        return self._chat(engagement_id, forced, history)

    # ---- Chat / upload / report helpers ----------------------------------

    def _extract_text(self, message: Dict) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                part.get("text", "") for part in content if part.get("type") == "text"
            )
        return ""

    def _to_backend_history(self, messages: List[Dict]) -> List[Dict[str, str]]:
        out = []
        for m in messages:
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            out.append({"role": role, "content": self._extract_text(m)})
        return out

    def _ingest_files(self, files: Optional[List[Dict]], engagement_id: str) -> str:
        if not files:
            return ""
        ingested = []
        for f in files:
            file_path = f.get("file", {}).get("path") or f.get("path")
            filename = f.get("file", {}).get("filename") or f.get("name", "upload")
            if not file_path:
                continue
            try:
                with open(file_path, "rb") as fh:
                    resp = requests.post(
                        f"{self.valves.BACKEND_URL}/upload",
                        files={"file": (filename, fh)},
                        data={"engagement_id": engagement_id},
                        timeout=self.valves.REQUEST_TIMEOUT,
                    )
                resp.raise_for_status()
                data = resp.json()
                ingested.append(f"- **{filename}**: indexed {data['chunks_indexed']} chunks")
            except Exception as e:
                ingested.append(f"- **{filename}**: failed to ingest ({e})")
        if not ingested:
            return ""
        return "📄 Document ingestion:\n" + "\n".join(ingested)

    def _detect_report_request(self, text: str) -> Optional[str]:
        t = text.lower()
        action = r"\b(generate|create|give|make|produce|write|export|download|build)\b"
        deliverable = r"\b(report|summary|deck|presentation|docx|pptx|pdf|document|write-?up)\b"
        if not (re.search(action, t) and re.search(deliverable, t)):
            return None
        if "pptx" in t or "powerpoint" in t or "slide" in t or "deck" in t or "presentation" in t:
            return "pptx"
        if "pdf" in t:
            return "pdf"
        return self.valves.DEFAULT_REPORT_FORMAT

    def _chat(self, engagement_id: str, message: str, history: List[Dict[str, str]]) -> str:
        try:
            resp = requests.post(
                f"{self.valves.BACKEND_URL}/chat",
                json={"engagement_id": engagement_id, "message": message, "history": history},
                timeout=self.valves.REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["reply"]
        except requests.HTTPError as e:
            return f"⚠️ Backend error: {e.response.status_code} {e.response.text[:300]}"
        except Exception as e:
            return f"⚠️ Could not reach the cybersecurity advisor backend: {e}"

    def _generate_report(self, engagement_id: str, fmt: str, focus_text: str) -> str:
        try:
            resp = requests.post(
                f"{self.valves.BACKEND_URL}/report/generate",
                json={
                    "engagement_id": engagement_id,
                    "format": fmt,
                    "focus_instructions": focus_text,
                },
                timeout=self.valves.REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            download_url = f"{self.valves.BACKEND_URL}{data['download_url']}"
            eng = data.get("engagement", {})
            return (
                f"✅ **Security report generated** for {eng.get('client_name', '')} — "
                f"{eng.get('engagement_name', '')} ({fmt.upper()})\n\n"
                f"- Findings: {data['findings_count']}\n"
                f"- Overall risk rating: **{data['overall_risk_rating']}**\n"
                f"- [Download the report]({download_url})\n\n"
                f"_If the link doesn't open directly in your browser, the backend also "
                f"serves it at `{data['download_url']}` on port 8010._"
            )
        except requests.HTTPError as e:
            return f"⚠️ Report generation failed: {e.response.status_code} {e.response.text[:300]}"
        except Exception as e:
            return f"⚠️ Could not reach the cybersecurity advisor backend: {e}"
