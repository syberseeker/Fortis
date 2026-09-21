# Fortis Setup Guide — Participant Edition

This guide gets Fortis — a fully local AI Cybersecurity Advisor desktop app —
installed and running on your machine. It is the companion to
`training/03-syllabus.md`: the workshop opens with a **guided setup (25 min)**
module, and this is the document participants follow during it. It also works
as self-service pre-workshop setup: run it the day before the workshop so
workshop time goes to the labs, not to downloads. Once setup is done, continue
with `training/04-desktop-guide.md` for the app walkthrough.

Everything runs locally. No Docker, no admin rights, no accounts, no cloud API.

## Before you start — checklist

Work through this list before running the setup script.

| # | Item | Requirement | How to check |
|---|------|-------------|--------------|
| 1 | Operating system | Windows 10 or 11 (64-bit) | Windows key + `winver` |
| 2 | Python | Python 3.10 – 3.14, **with "Add python.exe to PATH" ticked** during install. Download from https://www.python.org/downloads/ | Open PowerShell, run `python --version` — it should print `Python 3.x.x` with x.10 or higher |
| 3 | Free disk space | 20 GB minimum, 50 GB recommended | Windows key + `ms-settings:storagesense`, or check the drive in File Explorer |
| 4 | RAM | 16 GB minimum, 32 GB recommended | Task Manager → Performance → Memory |
| 5 | CPU | 4+ cores (6+ recommended) | Task Manager → Performance → CPU |
| 6 | GPU (optional) | NVIDIA with 4+ GB VRAM (e.g. RTX 3050) for fast answers; 8+ GB VRAM recommended. CPU-only works too — just slower | Open PowerShell, run `nvidia-smi` — if it lists your GPU, it is ready |
| 7 | NVIDIA driver | Up to date, from https://www.nvidia.com/drivers — only needed if you have an NVIDIA GPU | `nvidia-smi` shows a driver version |
| 8 | Internet | Needed once, for downloads (~2 GB of packages + ~3.4 GB model). The app itself runs fully offline afterwards | Any browser works |

Notes:

- If you are missing Python, install it from the link above and **tick
  "Add python.exe to PATH"** on the first installer screen — this is the single
  most common setup problem. Close and reopen PowerShell after installing.
- The GPU is optional. Without one, Fortis runs the AI engine on the CPU:
  answers take longer, but everything else is identical.

## Run the setup script

Open PowerShell in the Fortis folder (`C:\Fortis`), then run:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Use this exact command. A plain double-click "Run with PowerShell" can be
blocked by corporate policy; the `-ExecutionPolicy Bypass` form is not.

Total time: roughly **10–15 minutes** on the first run (most of it is the
dependency download). On later runs the script skips everything already done
and starts the app in seconds.

### What the script does, step by step

Each step prints a status line. Success looks like this:

**Step 1 — Python check** (~5 seconds)

```
  [ok] Python 3.12.4
```

If Python is missing or too old you will see:

```
  [X] Python 3.10+ not found.
      Install it from https://www.python.org/downloads/
      IMPORTANT: tick 'Add python.exe to PATH' in the installer.
```

Install Python with the PATH option ticked, reopen PowerShell, and run the
script again.

**Step 2 — Virtual environment** (~30 seconds, one-time)

```
  [..] Creating virtual environment (one-time)...
```

No error after this line means the `.venv` folder was created. On later runs
you will instead see `[ok] Virtual environment exists`.

**Step 3 — Dependencies** (~5 minutes, first run only)

```
  [..] Installing dependencies (first run only, ~5 min)...
  [ok] Dependencies installed
```

On later runs: `[ok] Dependencies present`. If this step fails it prints
`[X] Dependency install failed - check internet connection` — see
Troubleshooting below.

**Step 4 — Local AI engine (llama.cpp)** (~2 minutes, ~200 MB)

This is where the script decides between GPU and CPU:

- With an NVIDIA GPU:

  ```
  [..] Installing the local AI engine (llama.cpp, ~200 MB)...
       NVIDIA GPU detected - trying the CUDA build first...
  [ok] Engine installed (GPU-accelerated)
  ```

- Without an NVIDIA GPU (or if the CUDA build is unavailable):

  ```
  [ok] Engine installed (CPU)
  ```

  along with `CUDA build unavailable - falling back to CPU (slower but works)`
  in the fallback case. CPU mode is fully supported — inference is just
  slower. On later runs: `[ok] AI engine present`.

**Step 5 — Offline self-tests** (~1–2 minutes, no model download)

```
  [..] Running offline sanity tests...
  [ok] Self-tests passed
```

If you see `[X] Self-test failed - see output above`, note the text above it
and re-run the script once; if it fails again, see Troubleshooting.

**Step 6 — Launch**

```
  Setup complete. Starting Fortis...
  (First chat message downloads the AI model - about 3.4 GB, one time.)
```

The Fortis window (titled **Fortis — Cybersecurity Advisor**) opens. Setup is
done.

## First run: pick a model and send the first message

1. In the Fortis window, click **⚙ Model** (top right). The app has already
   detected your hardware and marks a suggested tier.
2. Pick the tier that matches your machine:

   | Your machine | Pick this tier | Download |
   |---|---|---|
   | Older or CPU-only laptop | Qwen3.5 0.8B | ~1.0 GB |
   | 8 GB RAM, no strong GPU | Qwen3.5 2B | ~2.7 GB |
   | 16 GB RAM or 4 GB+ VRAM (RTX 3050) | **Qwen3.5 4B (default)** | ~3.4 GB |
   | Low-VRAM laptop (2 GB), long policy documents | Llama 3.2 3B | ~1.9 GB |
   | RTX 4050-class or 6+ GB VRAM | Llama 3.1 8B | ~3.9 GB |

   If in doubt, take the default (4B). The remaining tiers are specialist
   models — see the catalog below.
3. **The model downloads on your first chat message**, not at setup. Type
   anything into the chat box and send it: a ~3.4 GB download starts (for the
   4B default). This is one-time, and it is **resumable** — if the network
   drops or the machine sleeps, restart the app and send a message again; it
   picks up where it stopped. A progress indicator shows the download; the
   first answer then takes 10–60 s while the model loads. Later answers are
   much faster.
4. Verify:

   | Check | Expected result |
   |---|---|
   | Send a chat message | A real answer arrives (not placeholder/stub text) |
   | Ask a controls question (e.g. "what does ISO 27001 require for access reviews?") | The answer references frameworks such as ISO 27001, NIST CSF, or CIS Controls |
   | Ask for "a docx report" | A download link appears in chat (details in `04-desktop-guide.md`) |

## Model catalog (reference)

All tiers from the in-app **⚙ Model** picker. "Min VRAM" is the NVIDIA VRAM
needed for GPU acceleration; tiers with "none" also run comfortably on CPU.

| Tier | Model | Download | Min VRAM | Best for |
|---|---|---|---|---|
| 0.8b | Qwen3.5 0.8B | ~1.0 GB | none | Fastest; any laptop |
| 2b | Qwen3.5 2B | ~2.7 GB | none | Light general use |
| 4b | Qwen3.5 4B *(default)* | ~3.4 GB | 3.5 GB | Recommended default; chat, GRC, OSINT |
| 3b-llama | Llama 3.2 3B | ~1.9 GB | 2.0 GB | Low-VRAM laptops; long policy context |
| 1.5b-coder | Qwen2.5-Coder 1.5B | ~1.0 GB | none | Code review on CPU-only machines |
| 3b-coder | Qwen2.5-Coder 3B | ~1.9 GB | 2.0 GB | Code review with a small GPU |
| 8b-llama | Llama 3.1 8B | ~3.9 GB | 5.5 GB | RTX 4050-class; 12k context; CTI |
| 7b-coder | Qwen2.5-Coder 7B | ~3.7 GB | 5.5 GB | Deep code analysis |
| 7b-r1 | DeepSeek-R1 Distill 7B | ~3.6 GB | 5.5 GB | OSINT / CTI reasoning |

## Verify your setup

| Symptom | Check |
|---|---|
| Chat answers look like placeholder/stub text | No model downloaded yet — open **⚙ Model**, pick a tier, send a message |
| Answers never mention ISO 27001 / NIST CSF / CIS Controls | Ask an explicit controls question; check an engagement is open (see `04-desktop-guide.md`) |
| No Fortis window, console shows `Port 8757 is already in use — another Fortis instance is likely running.` | Close the other Fortis window first, or start with a different port |
| Window did not open, console says `pywebview not installed — opening in the default browser instead.` | Normal fallback — Fortis is now open in your browser at `http://127.0.0.1:8757`; everything works the same |
| Very slow first answer | The model loads on first use (10–60 s). Later answers are faster |
| Model tier greyed out in the picker | Your GPU has less VRAM than that tier needs — pick a smaller tier |
| Model download is stuck | Check internet/proxy; restart the app and send a message — the download resumes |

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| PowerShell says scripts are disabled / the script is blocked | Windows ExecutionPolicy | Run the exact command: `powershell -ExecutionPolicy Bypass -File setup.ps1` |
| `[X] Python 3.10+ not found.` | Python missing, older than 3.10, or not on PATH | Install from https://www.python.org/downloads/ and tick **Add python.exe to PATH**; close and reopen PowerShell; re-run the script |
| `[X] Dependency install failed - check internet connection` | Network drop during the ~5 min dependency step | Reconnect and re-run the script; completed steps are skipped automatically |
| No NVIDIA GPU (or `nvidia-smi` is not recognized) | CPU-only machine or GPU driver missing | Nothing to do — the script installs the CPU engine automatically. Answers are slower; pick a smaller tier (2b or 0.8b) for snappier responses |
| Model download keeps failing | Corporate proxy/firewall blocks Hugging Face (huggingface.co and its CDN) | Ask IT to allow `huggingface.co`, `cdn-lfs.huggingface.co`, and `unsloth` model repos, or download the GGUF on a home network and copy it into `%APPDATA%\Fortis\models` |
| Model download interrupted | Wi-Fi drop, sleep, or laptop lid closed | Downloads are resumable — restart the app and send a chat message again; it continues from where it stopped |
| Low VRAM error, or 8B tiers greyed out | GPU has less VRAM than the tier requires (needs 3.5–5.5 GB depending on tier) | Open **⚙ Model** and pick a smaller tier (2b, 0.8b, or 3b-llama) |
| Fortis window never opens | pywebview failed or missing | This is handled: the app falls back to your default browser at `http://127.0.0.1:8757` (console prints `pywebview not installed — opening in the default browser instead.`). If nothing opens at all, re-run the setup script |
| `[X] Self-test failed - see output above` | A test failed — often a partial download or antivirus interference | Read the failing test name in the console, re-run the script once; if it persists, delete the `.venv` folder in `C:\Fortis` and re-run setup from scratch |

## Starting Fortis again later

The easiest way: re-run the setup command. It skips everything already
installed and opens the app.

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Equivalent direct start, from the `C:\Fortis` folder:

```powershell
.venv\Scripts\python.exe desktop\app.py
```

## Advanced options (optional)

These are not needed for the workshop. Skip them unless instructed.

### Headless mode (browser instead of app window)

Serves the same UI as a local web page — useful on machines where the native
window misbehaves, or when screen-sharing a browser:

```powershell
.venv\Scripts\python.exe -m server.app --port 8757
```

Then open `http://127.0.0.1:8757` in a browser. To use a different port,
change the `--port` value (or set the `FORTIS_PORT` environment variable;
8757 is the default).

### GPU-less demo machines (stub backend)

To rehearse the full app flow on a machine with no GPU and no room for a
model, run with the stub backend. It answers with deterministic offline
replies — no download, no model:

```powershell
$env:FORTIS_LLM_BACKEND = "stub"
.venv\Scripts\python.exe desktop\app.py
```

Valid values for `FORTIS_LLM_BACKEND` are `llama` (real model), `stub`
(offline replies), and `auto` (default: use the model if one is downloaded).
Environment variables only last for the current PowerShell window — set them
again in each new window.

### Move the data folder

All persistent data lives in one folder: `%APPDATA%\Fortis` (models, the
`chroma/` knowledge base, `uploads/`, `reports/`, and `engagements.sqlite3`).
To relocate it — e.g. to a bigger drive:

```powershell
$env:FORTIS_DATA_DIR = "D:\FortisData"
.venv\Scripts\python.exe desktop\app.py
```

Set `FORTIS_DATA_DIR` every time you start the app this way.

## Appendix: macOS and Linux

On macOS or Linux, run the equivalent script instead:

```bash
bash setup.sh
```

It performs the same steps as the Windows script (Python check, virtual
environment, dependencies, llama.cpp engine, the offline self-test suite) and
then starts Fortis.

- On Apple Silicon (M1–M4) the engine runs CPU-only via llama.cpp/Metal; the
  4B tier is the practical ceiling.
- With an NVIDIA GPU on Linux, the script tries the CUDA build first and falls
  back to CPU otherwise.
- Data lives in `~/.local/share/Fortis` (or `$XDG_DATA_HOME/Fortis` if set),
  with the same subfolders as Windows: `models/`, `chroma/`, `uploads/`,
  `reports/`, `engagements.sqlite3`.

After setup, continue with `training/04-desktop-guide.md` for the app
walkthrough.
