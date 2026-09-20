---
description: Owns the legacy containerized stack and training material. backend/, openwebui_pipeline/, docker-compose.yml, training/.
mode: subagent
model: arasintegrasi/Qwen/Qwen3.5-397B-A17B
---

You own the "how this was originally built as a service" artifacts:

- `backend/` — the original containerized FastAPI backend (Ollama-based).
- `openwebui_pipeline/` — `fortis_pipe.py` (Open WebUI pipeline) +
  `fortis_skill.py`.
- `docker-compose.yml` — the original 4-container stack (Ollama + FastAPI
  backend + Open WebUI + pipelines). Older notes referenced qwen2.5:3b / phi;
  verify against current compose + `.env` + `.plan` state.
- `training/` — workshop guides (`00-workshop-guide.md`,
  `01-lab-exercises.md`, `02-cheat-sheet.md`, `04-desktop-guide.md`) and
  sample lab docs (`sample-lab-docs/`).

Rules:
- This is the historical/reference stack, not the shipped desktop product.
  `core/` was lifted from it and is the shared library — keep them in sync in
  behavior (same API/schema) without rewriting the legacy stack.
- These modules contain their own prompts/guardrails; cross-check them against
  `server/orchestration.py` when behavior drifts matters.
- Training docs are deliverables — keep commands copy/paste-correct for the
  stack they reference (docker vs desktop).
- Do not add code comments unless asked.