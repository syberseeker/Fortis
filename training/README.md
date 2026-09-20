# Training Materials

## Workshop: How to Build Your Own AI Cybersecurity Advisor

### Contents

| File | Description |
|------|-------------|
| `00-workshop-guide.md` | Main workshop guide — modules, architecture deep dive, code walkthrough |
| `01-lab-exercises.md` | 7 hands-on labs with step-by-step instructions |
| `02-cheat-sheet.md` | Quick reference — architecture, file map, API endpoints, config |
| `04-desktop-guide.md` | **Desktop app guide — start here for participant setup** |
| `sample-lab-docs/` | Documents for the labs |

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
| `firewall-config.txt` | Intentionally misconfigured Cisco firewall — Lab 1 & 2 |
| `iam-policy.json` | Overly permissive AWS IAM policy — Lab 1 |
| `large-policy-document.txt` | 12-section enterprise security policy — Lab 4 (map-reduce) |
| `vulnerable-app.py` | Python Flask app with multiple vulnerabilities — optional extension |

### Running the original containerized stack (instructor / under-the-hood module)

1. Start the stack: `docker compose up -d --build`
2. Pull the model: `docker exec -it cyber-advisor-ollama ollama pull qwen2.5:3b`
3. Follow `00-workshop-guide.md` for presentation flow
4. Use `01-lab-exercises.md` for hands-on segments
5. Reference `02-cheat-sheet.md` for quick lookups during Q&A
