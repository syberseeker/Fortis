# Training Materials

## Fortis Training Materials

### Contents

| File | Description |
|------|-------------|
| `03-syllabus.md` | **Half-day end-user workshop syllabus — start here for instructors** |
| `05-setup-guide.md` | Participant setup steps (Windows-first) — companion to the syllabus |
| `01-lab-exercises.md` | 5 hands-on desktop-app labs (~55–60 min) |
| `04-desktop-guide.md` | Desktop app walkthrough — first-run guide for participants |
| `02-cheat-sheet.md` | Quick reference — architecture, file map, API endpoints, config (builder track) |
| `00-workshop-guide.md` | Builder-level deep dive — modules, architecture, code walkthrough (contributors) |
| `sample-lab-docs/` | Documents for the labs |

### Workshop tracks

- **End-user track (recommended):** follow
  `03-syllabus.md` → participants set up with `05-setup-guide.md` → hands-on
  with `01-lab-exercises.md` Labs 1–5. No coding, no Docker.
- **Builder track (contributors / under-the-hood):** follow
  `00-workshop-guide.md` for the full architecture and code walkthrough,
  against the `core/`, `engine/`, and `server/` code in this repo.

### Quick start (desktop app — recommended for participants)

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File setup.ps1
```
```bash
# macOS / Linux
bash setup.sh
```

Then: pick a model tier in the app (⚙ Model) → create an engagement →
upload a lab document → chat. Details in `04-desktop-guide.md`.

### Sample Lab Documents

| File | Purpose |
|------|---------|
| `firewall-config.txt` | Intentionally misconfigured Cisco firewall — Lab 2 |
| `iam-policy.json` | Overly permissive AWS IAM policy — Lab 2 |
| `large-policy-document.txt` | 12-section enterprise security policy — Lab 4 (persona modes) |
| `vulnerable-app.py` | Python Flask app with multiple vulnerabilities — Lab 4 extension (code persona) |
