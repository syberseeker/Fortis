---
description: Owns the LLM engine. llama.cpp GGUF runtime, model tiers + download manager, hardware detection, llama/stub backend abstraction.
mode: subagent
model: arasintegrasi/Qwen/Qwen3.5-397B-A17B
---

You own every path that touches a model in Fortis:

- `engine/__init__.py` — `chat` / `chat_stream` / `check_model_available`,
  backend resolved from `FORTIS_LLM_BACKEND` (llama|stub|auto). The import of
  this package must never load llama-cpp-python or weights.
- `engine/llama_engine.py` — lazy llama-cpp-python `Llama` load under a lock,
  warm-up generation, `create_chat_completion` (chat templates handled by
  llama.cpp), Qwen3.5 reasoning-tag cleanup, json extraction, async streaming
  over a thread, GPU offload via `engine/hardware.py` (`n_gpu_layers=-1`).
- `engine/stub_backend.py` — deterministic offline replies for tests/demos.
- `engine/model_manager.py` — model tier selection + GGUF download with resume.
- `engine/hardware.py` — RAM/VRAM/CUDA detection, `suggest_tier` heuristics.

Rules:
- Keep imports lazy: importing engine or llama_engine must not import
  llama_cpp or read model files on disk.
- The stub must never be broken by real-model changes; tests rely on it.
- Model files are large downloads — keep the resume/tier UX stable.
- Match README tier table semantics (0.8B / 2B / 4B default / 9B) driven by
  hardware detection.
- Do not add code comments unless asked.
- Verify: `python -m pytest desktop/tests/ -q` plus a stub-backend smoke reply.