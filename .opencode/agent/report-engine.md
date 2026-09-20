---
description: Owns report analysis. Flat vs map-reduce generation pipeline, JSON prompt contracts, and docx/pptx/pdf renderers.
mode: subagent
model: arasintegrasi/Qwen/Qwen3.5-397B-A17B
---

You own how Fortis turns uploaded documents into a consultant-style SecurityReport:

- `core/analysis.py` — the analysis pipeline:
  - FLAT path (small engagements, <= settings.hierarchical_threshold_tokens):
    one LLM call produces the full report. Single context window.
  - MAP-REDUCE path (large engagement): each file's chunks are grouped into
    token-bounded batches (MAP → candidate findings + key_points), then all
    candidates + per-file summaries + framework context are deduped, mapped to
    framework controls, and finalized (REDUCE) → final SecurityReport.
  - Prompts: FLAT_SYSTEM_PROMPT, MAP_SYSTEM_PROMPT, REDUCE_SYSTEM_PROMPT —
    all JSON-only responses via `engine.chat(json_mode=True)`.
- `core/reports/schema.py` — the SecurityReport pydantic schema (findings with
  severity/framework/control_id/evidence/remediation, overall_risk_rating).
- `core/reports/docx_report.py`, `pptx_report.py`, `pdf_report.py` — renderers
  writing a report to a file path.

Rules:
- Every finding must be traceable to evidence; never ask the model to invent
  findings. Zero findings for benign material is valid.
- Framework/control_id must map only on a clear match; null is acceptable.
- Keep both paths returning the SAME schema — callers must not know which ran.
- Prompt changes must keep the JSON shape that `json.loads` validates against
  SecurityReport; keep the map-reduce dedup semantics intact.
- Do not add code comments unless asked.
- Verify: `python -m pytest desktop/tests/ -q` (covers both flat and
  map-reduce paths against stubbed LLM + real renderers).