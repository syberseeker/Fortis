# Fortis Desktop — Trainer & Participant Guide

Fortis is now a **desktop application**. No Docker, no terminals, no admin
rights: participants run one script and a window opens.

```
Participant setup:
  Windows:   double-click  setup.ps1        (or: powershell -ExecutionPolicy Bypass -File setup.ps1)
  macOS/Linux:  bash setup.sh
```

The script: checks Python → creates a virtual environment → installs
dependencies → installs the llama.cpp engine (GPU build if an NVIDIA card is
present, CPU otherwise) → runs 14 offline self-tests → opens Fortis.

## First run inside the app

1. Click **⚙ Model** (top right). The app detected the hardware and marks a
   suggested tier:

   | Tier | Download | Runs on |
   |---|---|---|
   | Qwen3.5 0.8B | ~1.0 GB | any machine |
   | Qwen3.5 2B | ~2.7 GB | 8 GB RAM |
   | Qwen3.5 4B *(default)* | ~3.4 GB | 4 GB+ VRAM or 16 GB RAM |
   | Qwen3.5 9B | ~6.6 GB | 8 GB+ VRAM only |

2. Click a tier → downloads once (~3.4 GB for 4B, resumable) → send any
   message; the model loads automatically.

3. Click **+ New engagement** (sidebar) — documents, chats, and reports are
   scoped to an engagement, so closing the app never loses work.

4. Attach files with **📎** (PDF, DOCX, PPTX, XLSX, CSV, configs, code),
   ask questions, then ask for "a docx report" — download links appear in
   chat; reports also land in the reports folder (sidebar button).

## What changed vs. the Docker stack

| | Docker stack (old) | Fortis Desktop (new) |
|---|---|---|
| Participant setup | Docker Desktop + compose + model pull + pipeline install | one script |
| Components | 4 containers, 4 ports | 1 process, 1 window |
| Models | Ollama, manual pull | llama.cpp + Qwen3.5 GGUF, in-app picker |
| Sign-up / roles | admin approval flow | none — single local user |
| Engagements | slash commands in chat | sidebar buttons *and* slash commands |
| Data location | Docker volumes | `%APPDATA%\Fortis` (Win) / `~/.local/share/Fortis` (Mac/Linux) |

Slash commands still work in the chat box: `/engagements`, `/new-engagement
Client :: Name`, `/use <id>`, `/whoami`, `/close-engagement`.

## For the instructor

- **Prep:** run the setup script on your demo machine before the session and
  pre-download the 4B tier — venue Wi-Fi then only matters for participants'
  own downloads.
- **Under-the-hood module:** the old container architecture
  (`docker-compose.yml`, backend/, `openwebui_pipeline/`) is still in this
  repo and still runs — use it for the "how would you build this as a
  service" module. The desktop app reuses the same RAG core (`core/`),
  report renderers, and engagement store, so concepts transfer 1:1.
- **Offline testing:** `EMBEDDING_BACKEND=hash-stub FORTIS_LLM_BACKEND=stub
  python -m pytest desktop/tests/` validates the whole pipeline with no
  model and no network.
- **Headless demo:** `python -m server.app --port 8757` serves the same UI
  at `http://127.0.0.1:8757` (useful when screen-sharing a browser).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `python` not found (Windows) | Reinstall Python, tick **Add to PATH** |
| Setup script blocked (Windows) | `powershell -ExecutionPolicy Bypass -File setup.ps1` |
| Very slow first answer | The model loads on first use (10–60 s). Later answers are faster. |
| 9B tier greyed out | GPU has < 8 GB VRAM — use 4B |
| Answers are stub/placeholder text | No model downloaded yet — open **⚙ Model** |
| Report link doesn't download | The browser blocked it — files are in the reports folder (sidebar button) |
| Apple Silicon (M1–M4) | Works CPU-only via llama.cpp; 4B is the practical ceiling |
