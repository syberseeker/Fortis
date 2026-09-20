# Lab Exercises: Building Your AI Cybersecurity Advisor

These hands-on labs walk you through the system end-to-end. Each lab builds on the previous one.

**Prerequisites:** Docker Compose stack running (`docker compose up -d --build`), model pulled (`ollama pull qwen2.5:3b`), Open WebUI accessible at `http://localhost:3000`.

---

## Lab 1: Upload and Analyze a Firewall Configuration

**Time:** 15 minutes
**Goal:** Understand the full ingestion → retrieval → chat flow

### Step 1: Create an Engagement

In the Open WebUI chat, type:

```
/new-engagement Workshop Lab :: Firewall Config Review
```

You should see: "Created and activated engagement **Workshop Lab — Firewall Config Review**."

### Step 2: Upload the Sample Config

Use the file attachment button in Open WebUI to upload `sample-lab-docs/firewall-config.txt`. The pipeline will automatically ingest it.

You should see:
```
Document ingestion:
- firewall-config.txt: indexed 2 chunks
```

### Step 3: Ask Questions

Try these queries and observe how Fortis responds:

```
What are the security issues in this firewall configuration?
```

```
Which NIST CSF controls are relevant to the findings?
```

```
What would you recommend to improve this configuration?
```

### Step 4: Verify Grounding

Ask Fortis to cite its sources:
```
For each finding, tell me which file and section it came from.
```

**What to look for:**
- Findings reference the actual uploaded file (not hallucinated sources)
- Framework mappings are reasonable (not forced)
- Severity ratings have justification
- Remediation is specific to your config, not generic

### Step 5: Check Engagement State

```
/whoami
```

Note the files listed — they persist. Close the browser tab, open a new one, and type:

```
/use workshop-lab-firewall-config-review
what did we find last time?
```

The context persists across sessions.

---

## Lab 2: Generate a Security Report

**Time:** 15 minutes
**Goal:** Exercise the analysis engine and report generation

### Step 1: Request a Report (DOCX)

Still in the same engagement from Lab 1, type:

```
generate a docx report
```

You should see a response like:
```
Security report generated for Workshop Lab — Firewall Config Review (DOCX)

- Findings: [N]
- Overall risk rating: [High/Medium/etc]
- [Download the report]
```

### Step 2: Download and Review

Click the download link or navigate to:
```
http://localhost:8010/report/download/[filename].docx
```

Open the DOCX file. Check for:
- Title page with engagement name and date
- Client Context section
- Scope section
- Executive Summary with overall risk rating
- Findings section with severity colors, evidence, framework mappings
- Recommendations summary

### Step 3: Try Other Formats

```
generate a pptx presentation
```

```
generate a pdf report
```

Compare the three formats — same data, different presentation.

### Step 4: Add Focus Instructions

```
generate a docx report focusing on access control issues only
```

The `focus_instructions` parameter narrows the LLM's analysis.

---

## Lab 3: Explore the API Directly

**Time:** 10 minutes
**Goal:** Understand the backend API

### Step 1: Open API Docs

Navigate to `http://localhost:8010/docs` — this is the auto-generated FastAPI Swagger UI.

### Step 2: List Engagements

Find the `GET /engagements` endpoint and click "Try it out". You should see the engagement you created.

### Step 3: Upload via API

```bash
curl -X POST http://localhost:8010/upload \
  -F "file=@sample-lab-docs/iam-policy.json" \
  -F "engagement_id=workshop-lab-firewall-config-review"
```

### Step 4: Chat via API

```bash
curl -X POST http://localhost:8010/chat \
  -H "Content-Type: application/json" \
  -d '{
    "engagement_id": "workshop-lab-firewall-config-review",
    "message": "What are the top 3 risks in the uploaded documents?"
  }'
```

### Step 5: Generate Report via API

```bash
curl -X POST http://localhost:8010/report/generate \
  -H "Content-Type: application/json" \
  -d '{
    "engagement_id": "workshop-lab-firewall-config-review",
    "format": "pdf",
    "focus_instructions": "Focus on credential management"
  }'
```

---

## Lab 4: Large Document — Map-Reduce in Action

**Time:** 10 minutes
**Goal:** Trigger and understand the map-reduce analysis path

### Step 1: Create a New Engagement

```
/new-engagement Workshop Lab :: Large Document Test
```

### Step 2: Upload the Large Sample

Upload `sample-lab-docs/large-policy-document.txt`. This document is intentionally large enough to exceed the `HIERARCHICAL_THRESHOLD_TOKENS` (5000 tokens in production, 200 in tests).

### Step 3: Generate a Report

```
generate a docx report for this large document
```

**What to observe in the backend logs:**
```bash
docker logs cyber-advisor-backend 2>&1 | tail -20
```

You should see log lines indicating:
```
Engagement ...: N tokens > threshold, using map-reduce analysis pipeline
```

This means the system automatically:
1. Split the document into token-bounded batches
2. Analyzed each batch independently (map)
3. Combined and deduplicated findings (reduce)

### Step 4: Compare

Upload a small file to a separate engagement and generate a report. Check the logs — you should see the flat path:
```
Engagement ...: N tokens <= threshold, using flat analysis pass
```

---

## Lab 5: Extending the System — Custom Framework

**Time:** 15 minutes
**Goal:** Add your own security framework to the reference corpus

### Understanding Framework Storage

Framework data lives in two places:
1. **Built-in JSON files** (`backend/app/frameworks/*.json`) — loaded at startup
2. **ChromaDB** (`security_frameworks` collection) — where retrieval happens

### Step 1: Create a Custom Framework JSON

Create a file called `custom_framework.json` in `backend/app/frameworks/`:

```json
[
  {
    "id": "CUSTOM-1",
    "title": "Data Classification",
    "text": "All data must be classified according to sensitivity levels: Public, Internal, Confidential, Restricted. Classification labels must be applied to all documents and data stores.",
    "version": "Internal Policy v1.0",
    "source_url": null
  },
  {
    "id": "CUSTOM-2",
    "title": "Password Policy",
    "text": "Minimum 14 characters, must include uppercase, lowercase, numbers, and special characters. Passwords must be rotated every 90 days. No password reuse across 12 generations.",
    "version": "Internal Policy v1.0",
    "source_url": null
  },
  {
    "id": "CUSTOM-3",
    "title": "Multi-Factor Authentication",
    "text": "MFA is required for all remote access, administrative accounts, and access to Confidential or Restricted data. Acceptable MFA methods: hardware tokens, TOTP apps. SMS-based MFA is prohibited.",
    "version": "Internal Policy v1.0",
    "source_url": null
  }
]
```

### Step 2: Register the Framework

Edit `backend/app/frameworks/__init__.py` and add your framework:

```python
_FILES = {
    "NIST_CSF": "nist_csf.json",
    "OWASP_TOP10": "owasp_top10.json",
    "CIS_CONTROLS": "cis_controls.json",
    "MITRE_ATTACK": "mitre_attack.json",
    "CIS_BENCHMARKS": "cis_benchmarks.json",
    "CUSTOM_POLICY": "custom_framework.json",  # Add this line
}
```

### Step 3: Rebuild and Test

```bash
docker compose build backend
docker compose up -d backend
```

Verify it loaded:
```bash
curl http://localhost:8010/health
# Should show "CUSTOM_POLICY" in frameworks_loaded
```

### Step 4: Test Retrieval

Upload a document that violates your custom policy and ask:
```
Does this document comply with our internal password policy?
```

Fortis should now retrieve and reference your custom framework controls.

### Step 5: Load a Real PDF (Optional)

If you have an official NIST CSF PDF:

```bash
docker exec -it cyber-advisor-backend \
  python -m app.frameworks.loader NIST_CSF /path/to/NIST.CSWP.29.pdf
```

This replaces the paraphrased seed with verbatim text from the official document.

---

## Lab 6: Testing the System

**Time:** 10 minutes
**Goal:** Run the offline test suite and understand what it validates

### Step 1: Run Tests (No GPU Required)

```bash
cd backend
pip install -r requirements-dev.txt --break-system-packages
pytest tests/ -v
```

### Step 2: Understand the Test Architecture

Read `tests/conftest.py` — it sets environment variables before any imports:

```python
os.environ["EMBEDDING_BACKEND"] = "hash-stub"       # No model download
os.environ["HIERARCHICAL_THRESHOLD_TOKENS"] = "200"  # Low threshold
os.environ["MAP_BATCH_TOKENS"] = "80"                # Small batches
os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:1" # Deliberately unreachable
```

Read `tests/test_integration.py` — the LLM is monkeypatched:

```python
async def fake_chat(messages, temperature=0.2, json_mode=False):
    system_content = messages[0]["content"]
    if "ONE batch of excerpts" in system_content:
        return json.dumps(FAKE_MAP_RESULT)   # Map phase response
    if "already reviewed" in system_content:
        return json.dumps(FAKE_REPORT)        # Reduce phase response
    if json_mode:
        return json.dumps(FAKE_REPORT)        # Flat path response
    return "This is a stub conversational reply from Fortis."
```

### Step 3: What the Tests Validate

| Test | What it checks |
|------|----------------|
| `test_health` | Frameworks loaded correctly |
| `test_create_engagement_has_readable_slug` | ID generation (human-readable slugs) |
| `test_create_engagement_idempotent` | Duplicate creation returns same ID |
| `test_active_engagement_persists_per_user` | Per-user engagement scoping |
| `test_upload_and_list_files` | File ingestion and listing |
| `test_chat_returns_stub_reply` | Chat routing works |
| `test_report_generation_flat_path` | Small doc → flat analysis → valid DOCX/PPTX/PDF |
| `test_report_generation_triggers_map_reduce` | Large doc → map-reduce triggered |

### Step 4: What It Does NOT Validate

- Model output quality (needs real GPU + Ollama)
- Embedding retrieval quality (hash-stub is not semantically meaningful)
- UI integration (Open WebUI pipeline)

For quality validation, you need the live stack. This test suite validates the wiring.

---

## Lab 7: Customization Challenge

**Time:** 15 minutes (open-ended)
**Goal:** Modify the system and observe the effects

### Challenge 1: Change the Persona

Edit `backend/app/rag.py` and modify the `SYSTEM_PERSONA`. Change the name, the expertise description, or the scope restrictions. Rebuild the backend and observe how it changes responses.

### Challenge 2: Adjust Severity Criteria

Edit the `FLAT_SYSTEM_PROMPT` in `backend/app/analysis.py`. Change the severity rating criteria — for example, add a requirement that "any finding involving credentials must be rated Critical". Rebuild and test with the sample firewall config.

### Challenge 3: Add a New File Format

Edit `backend/app/ingestion.py`. Add support for `.log` files with timestamp parsing, or `.xml` files with element extraction. Test by uploading a sample file of that type.

### Challenge 4: Modify the Topic Guardrail

Edit `openwebui_pipeline/cyber_advisor_pipe.py`. Add a keyword pattern that currently blocks a query you think should be allowed, or remove one that's too permissive. Test the boundary.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Model not pulled yet" | `docker exec -it cyber-advisor-ollama ollama pull qwen2.5:3b` |
| "Could not reach the backend" | Check `docker compose ps` — all containers should be "Up" |
| "No active engagement" | Create one first: `/new-engagement Client :: Name` |
| Upload returns 413 | File exceeds 40MB limit — split it |
| Report generation is slow | First run downloads embedding model weights — subsequent runs are fast |
| GPU not detected | Verify NVIDIA drivers + `nvidia-smi` works outside Docker |
| Chroma errors after restart | Data persists in Docker volumes — don't use `docker compose down -v` |
