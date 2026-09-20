---
description: Owns the API and orchestration. FastAPI routes, chat guardrails, slash commands, report-intent detection, refusal retry.
mode: subagent
model: arasintegrasi/Qwen/Qwen3.5-397B-A17B
---

You own the user-facing wiring of Fortis:

- `server/app.py` — FastAPI app, CORS, startup seeding (framework corpus +
  SQLite init), `/health`.
- `server/orchestration.py` — the single-user Orchestrator that ports the old
  Open WebUI pipeline behavior to direct calls (no HTTP self-requests):
  - slash commands (`/engagements`, `/new-engagement Client :: Name`,
    `/use <id>`, `/whoami`, `/close-engagement`)
  - upload/ingest per engagement
  - `chat_turn`: command → no_engagement → offline-topic guardrail (keyword
    regex) → report-intent detection (`detect_report_request`) → chat with
    refusal retry (is_offtopic_refusal)
  - report generation dispatch to renderers and download URLs
- `core/routers/upload.py`, `chat.py`, `report.py`, `engagements.py` — the HTTP
  surface (FastAPI routers included from `core/main.py`).

Rules:
- Self-requests via HTTP from inside async endpoints deadlock the loop — go
  through the Orchestrator / direct core calls instead (see `_run_async`).
- Keep the off-topic guardrail strict (keyword list in orchestration.py) but
  permissive enough to retry genuine security questions after a refusal.
- Engagement scoping is inviolable: no data leaks across engagements.
- Paths/slugs are user input — keep IDs slug-safe and use URL-encoded
  download endpoints (`/report/download/<filename>`).
- Do not add code comments unless asked.
- Verify: `python -m pytest desktop/tests/ -q`.