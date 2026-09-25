# Fortis Workshop Syllabus — Local AI Cybersecurity Advisor (Half Day)

**Duration:** ~2 hours (120 min)
**Level:** Beginner — no coding, no Docker, no command-line experience required
**Platform:** Fortis Desktop (fully local AI cybersecurity advisor, Windows-first)
**Companion labs:** `01-lab-exercises.md` (Labs 1–5)

---

## Course Overview

Fortis is a fully local AI cybersecurity advisor: users upload documents (configs, policies, code, logs), receive findings mapped to NIST CSF, OWASP Top 10, and CIS Controls, and generate consultant-grade DOCX, PPTX, or PDF reports. Everything runs on the local machine — no cloud APIs, no Docker, no data leaves the network — which is what makes it safe for confidential client material, air-gapped environments, and regulated data. This half-day session takes a non-technical audience from a plain Windows laptop to a finished, framework-mapped security report.

## Audience & Prerequisites

This workshop is built for **end users, not developers**: security analysts, GRC staff, IT security teams, auditors, and consultants who want AI assistance without sending documents to a cloud service.

Participants need:

- Comfort with the Windows file explorer (open, copy, browse to a folder)
- A Windows 10/11 laptop that meets the hardware minimums below
- Python installed (python.org installer, tick **Add to PATH**) — the setup script checks for it
- Internet access for the one-time model download (~3.4 GB, resumable if interrupted)

Participants do **not** need:

- Any programming or scripting experience
- Docker, virtual machines, or cloud accounts
- Prior exposure to AI tools or prompt engineering

> **Mac/Linux footnote:** `setup.sh` exists and the app works the same way
> (Apple Silicon runs CPU-only via llama.cpp). All instructions in this
> syllabus are Windows-first; Mac/Linux is a self-service variation.

## Learning Objectives

By the end of this workshop you can:

1. Set up Fortis on a Windows machine with a single script and open the desktop app
2. Choose the right model tier for your hardware and understand the one-time download
3. Create an engagement, attach documents, and get framework-mapped findings from chats
4. Generate consultant-grade DOCX, PPTX, and PDF reports from an engagement
5. Switch persona modes (general, grc, code, osint/cti) to change how findings are framed
6. Request structured summaries and Mermaid diagrams, and find all your saved work on disk

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 6+ cores |
| RAM | 16 GB | 32 GB |
| GPU | 4 GB VRAM (RTX 3050) | 8+ GB VRAM |
| Disk | 20 GB free | 50 GB free |

**CPU-only note:** machines without an NVIDIA GPU still work (CPU-only
fallback) — setup detects this and installs the CPU build of the engine. It
works, just slower; pick the 0.8B or 2B tier on those machines.

## Model Tiers (reference)

Models download lazily on first chat — one-time, resumable. Pick via the
**⚙ Model** button (top right); the app suggests a tier based on detected
hardware.

| Tier | Download | Role tags | Runs on |
|---|---|---|---|
| Qwen3.5 0.8B | ~1.0 GB | chat | any machine |
| Qwen3.5 2B | ~2.7 GB | chat, osint | 8 GB RAM |
| **Qwen3.5 4B (default)** | ~3.4 GB | chat, grc, osint | 4 GB+ VRAM or 16 GB RAM |
| Llama 3.2 3B | ~1.9 GB | chat, grc | low-VRAM laptops, long policy context |
| Qwen2.5-Coder 1.5B | ~1.0 GB | code | CPU-only machines |
| Qwen2.5-Coder 3B | ~1.9 GB | code | 2 GB+ VRAM |
| Llama 3.1 8B | ~3.9 GB | chat, grc, cti | RTX 4050-class, 12k context |
| Qwen2.5-Coder 7B | ~3.7 GB | code | 5.5 GB+ VRAM |
| DeepSeek-R1 Distill 7B | ~3.6 GB | osint, cti | 5.5 GB+ VRAM |

For this workshop, **everyone should use the default 4B tier** (or 0.8B/2B on
CPU-only laptops) so results are comparable across the room.

## Pre-Workshop Checklist

### Participants

- [ ] Windows 10/11 laptop meeting the minimums above, plugged in
- [ ] At least 20 GB free disk space (the default model is ~3.4 GB)
- [ ] Python installed with **Add to PATH** ticked
- [ ] NVIDIA drivers up to date (if the laptop has an NVIDIA GPU)
- [ ] Internet access for the one-time model download
- [ ] Copy of `sample-lab-docs/` on the Desktop or knowledge of the share location
- [ ] No admin rights, Docker, or cloud accounts needed — leave them at home

### Instructor prep

- [ ] Run `setup.ps1` on a clean, participant-like machine before the session; fix anything that breaks
- [ ] **Pre-download the 4B model tier on the demo machine before class** — venue Wi-Fi then only matters for participants' own downloads
- [ ] Verify GPU drivers on the demo machine and confirm the suggested tier is sensible
- [ ] Have `training/sample-lab-docs/` ready on a share or USB stick
- [ ] Optional bandwidth saver: pre-seed `%APPDATA%\Fortis\models` on participant machines so the first chat skips the download
- [ ] Optional: enlarge projector font / zoom the app window — you will demo every click

## Agenda

| # | Module | Time | Topic | Lab mapping |
|---|---|---|---|---|
| 1 | Welcome & concepts | 15 min | What Fortis is, why local AI matters for security work, product tour | — |
| 2 | Guided setup | 25 min | Run the setup script, verify the window opens, what the script does | Prep for Lab 1 |
| 3 | First engagement & chat | 20 min | Model tier picker, create engagement, first message triggers download | Lab 1 |
| 4 | Document analysis | 20 min | Upload configs and policies, review framework-mapped findings, follow-ups | Labs 1–2 (Lab 1 wrap-up, Lab 2) |
| 5 | Reports & personas | 25 min | Generate DOCX/PPTX/PDF; compare general vs grc personas | Labs 3–4 |
| 6 | Structured output, wrap-up & Q&A | 15 min | Mermaid diagrams, structured summaries, resources, questions | Lab 5 |
| | **Total** | **120 min** | | |

Hands-on lab time inside the agenda is ~55–60 minutes (Lab 1: 10, Lab 2: 15,
Lab 3: 10, Lab 4: 10, Lab 5: 10); the rest is instruction and demos.

---

## Module 1 — Welcome & Concepts (15 min)

- **Objective:** understand what Fortis does and why "fully local" matters for security work (confidentiality, compliance, air-gapped environments).
- **Instructor demos:** a 3-minute tour of the finished product — open an existing engagement, show a chat answer with framework citations, open a generated report. No setup, just the destination.
- **Participants do:** watch, ask questions, note one use case from their own job they want to try.
- **Lab:** none — this is the "why" before the "how".

## Module 2 — Guided Setup (25 min)

- **Objective:** every participant has the Fortis desktop window open.
- **Instructor demos:** run the setup command on the projector and narrate each step (checks Python, creates a virtual environment, installs dependencies, installs the llama.cpp engine — GPU build if an NVIDIA card is present, CPU otherwise — runs the offline self-test suite, opens Fortis).
- **Participants do:** run the setup script and wait for the app window:

  ```powershell
  powershell -ExecutionPolicy Bypass -File setup.ps1
  ```

  If the script is blocked, re-run the same command (the `-ExecutionPolicy Bypass` part handles it).
- **Lab:** prep for Lab 1 — nobody chats yet.

## Module 3 — First Engagement & Chat (20 min)

- **Objective:** understand engagements as the unit of work, pick a model tier, and get a first answer.
- **Instructor demos:** the **⚙ Model** picker (top right) and how hardware detection suggests a tier; creating an engagement with **+ New engagement** (sidebar); sending the first message and narrating the one-time model download (~3.4 GB for the default 4B tier, resumable); the 10–60 s first-answer load. Mention that every chat turn is saved per engagement (newest 500) and restored when they switch back or restart — switching engagements mid-lab no longer loses the conversation, and **Clear chat** (top bar) deliberately erases the active engagement's transcript.
- **Participants do:** Lab 1 — launch is already done, so they create an engagement and send their first message, watching the download progress.
- **Lab:** **Lab 1** (verify setup & first chat).

## Module 4 — Document Analysis (20 min)

- **Objective:** turn uploaded documents into framework-mapped findings and drill in with follow-up questions.
- **Instructor demos:** attach `firewall-config.txt` with the attach button (paperclip icon), ask for findings, show how results cite NIST CSF / OWASP Top 10 / CIS Controls; demonstrate one follow-up that narrows to a specific finding.
- **Participants do:** Lab 1 wrap-up (first answer received), then Lab 2 — upload `firewall-config.txt` and `iam-policy.json`, review findings, ask at least two follow-ups.
- **Lab:** **Labs 1–2** (Lab 1 wrap-up, Lab 2 document analysis).

## Module 5 — Reports & Personas (25 min)

- **Objective:** produce a deliverable report and control how findings are framed via persona modes.
- **Instructor demos (folded into the lab steps):** while participants generate their reports, ask for "a docx report" in the Lab 2 engagement on the projector — download links appear in chat and files land in the reports folder (sidebar button); repeat as pptx and pdf. Mention the `/report [docx|pptx|pdf]` slash command as the keyboard-only alternative. Then switch persona mode and re-ask the same question on `large-policy-document.txt` to contrast **general** (balanced consultant summary) vs **grc** (named control mappings, audit-evidence phrasing, gap/finding/observation distinction).
- **Participants do:** Lab 3 (generate all three report formats from their Lab 2 engagement), then Lab 4 (general vs grc comparison; optional: try the **code** persona on `vulnerable-app.py`).
- **Lab:** **Labs 3–4** (report generation; persona modes).

## Module 6 — Structured Output, Wrap-Up & Q&A (15 min)

- **Objective:** get machine-usable output, not just prose, and know where to go next.
- **Instructor demos:** request a Mermaid diagram of an attack path or control flow (rendered in chat) and a structured summary (tables/fields instead of paragraphs); recap resources and the data location.
- **Participants do:** Lab 5 (mermaid diagram + structured summary on their engagement), remaining questions, and fill out the exit checklist.
- **Lab:** **Lab 5** (structured output & diagrams).

---

## Post-Workshop Resources

| Resource | Use it for |
|---|---|
| `training/01-lab-exercises.md` | Re-run Labs 1–5 at your own pace with your own documents |
| `training/02-cheat-sheet.md` | **Builder track only:** quick reference — data flow, config, severity scale. End users: skip this one |
| `training/04-desktop-guide.md` | Desktop app guide: setup, model tiers, troubleshooting table |
| `training/sample-lab-docs/` | Practice documents (`firewall-config.txt`, `iam-policy.json`, `large-policy-document.txt`, `vulnerable-app.py`) before trusting it with real client material — keep using these after the workshop |

**Where your data lives:** everything is local under `%APPDATA%\Fortis`:

| Folder / file | Contents |
|---|---|
| `models/` | Downloaded model files (one-time, resumable downloads) |
| `chroma/` | Vector index of your uploaded documents |
| `uploads/` | Documents you attached to engagements |
| `reports/` | Generated DOCX/PPTX/PDF reports |
| `engagements.sqlite3` | All engagements, chats, and settings - closing the app never loses work. Chats are restored per engagement (newest 500 turns); **Clear chat** deletes an engagement's transcript |

Mac/Linux data location: `~/.local/share/Fortis`.

## Not Covered in This Workshop

This session stays at the UI level. Out of scope:

- **Docker / Ollama internals** — containerized deployment is not part of this repo
- **Code-level customization** — prompts, guardrails, RAG internals, extending
  frameworks

The code-level walkthrough is covered in `00-workshop-guide.md`, the builder-level workshop
(3–4 hours, intermediate: assumes Python and networking basics).
Point interested participants there instead of going down that path in class.
