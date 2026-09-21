---
description: Print the Fortis architecture and agent ownership map.
agent: fortis
---

Print this project's architecture and agent ownership map:

- Repo layout (core/, engine/, server/, desktop/, openwebui/, training/)
- How the app runs (desktop, headless server, tests, LLM backend env vars)
- Which opencode subagent owns which area, and who the read-only security reviewer is
- The data model (engagement-scoped documents → RAG → SecurityReport)

Keep it a compact tree/diagram the user can paste into a chat.