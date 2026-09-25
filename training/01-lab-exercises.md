# Lab Exercises: Working with Fortis Desktop

Five hands-on labs for the Fortis desktop app — a fully local AI cybersecurity advisor. Everything runs on your machine: no cloud, no accounts, no code. You launch the app, create engagements, attach documents, chat, and generate consultant reports.

**Total lab time:** ~55–60 minutes. Labs 1–5 are sequential; Labs 2, 3 and 5 reuse engagements from earlier labs; Lab 4 creates a new one.

**Prerequisites:** Completed setup per `05-setup-guide.md` — running `setup.ps1` once checks Python, creates a virtual environment, installs dependencies and the llama.cpp engine, runs the offline self-tests, and opens Fortis. All lab files are in `training\sample-lab-docs\` inside the Fortis folder.

**Syllabus:** Labs 1–5 map to the modules in `03-syllabus.md`.

All work is scoped to an engagement and stored locally under `%APPDATA%\Fortis` (`models\`, `chroma\`, `uploads\`, `reports\`, `engagements.sqlite3`), so closing the app never loses work.

---

## Lab 1: Verify setup & first chat (~10 min)

**Objective:** Launch the desktop app, complete the one-time model download, create an engagement, and confirm a framework-grounded reply.

**Duration:** ~10 minutes

### Steps

1. Run `setup.ps1` in the Fortis folder. When the offline self-tests pass, the Fortis window opens — dark sidebar on the left, chat area on the right.
2. Click **⚙ Model** (top right). The app detected your hardware and marks a suggested tier. Confirm **Qwen3.5 4B (recommended default)** is listed.
3. Click the 4B tier. The model downloads once (~3.4 GB, resumable) and the dialog reports **Download complete — model loads on next message.** Close the dialog.
4. Click **+ New engagement** (sidebar). In the dialog enter:
   - Client name: `Workshop`
   - Engagement name: `Lab 1 — first chat`
   - Notes: leave blank (optional)

   Click **Create**. The status line confirms the engagement was created and activated, and the top bar shows its name.
5. Click in the chat box, type `What are the OWASP Top 10 risks I should check first in a new web application?` and press Enter.
6. The first message loads the model (10–60 s) before the reply streams. The status dot turns green once the model is ready.

### Expected result

A consultant-style answer grounded in named frameworks rather than generic chat output. Fortis ships 12 framework seeds — NIST CSF, OWASP Top 10, CIS Controls, plus MITRE ATT&CK, CWE Top 25, NIST 800-53, ISO 27001, SOC 2, PCI DSS, GDPR, HIPAA, and CIS Benchmarks.

### Try also

Send `Compare NIST CSF 2.0 functions with CIS Controls v8.1 in a table.` — one of the starter suggestions on the empty chat screen — and note the structured comparison.

---

## Lab 2: Document analysis (~15 min)

**Objective:** Attach two sample documents to an engagement, review the security findings, and drill into the highest-risk one.

**Duration:** ~15 minutes

**Files used:** `training\sample-lab-docs\firewall-config.txt` (a misconfigured Cisco IOS-style perimeter firewall) and `training\sample-lab-docs\iam-policy.json` (an over-permissive AWS IAM policy).

### Steps

1. Reuse the Lab 1 engagement, or click **+ New engagement** and create `Lab 2 — document review`.
2. Click **📎** (Attach file for review) at the left of the chat box, browse to `training\sample-lab-docs\`, and select `firewall-config.txt`. The status line shows the ingest progress, then the chat confirms `<file> ingested: N chunks indexed`.
3. Attach `iam-policy.json` the same way.
4. Ask: `What are the security issues across the documents I uploaded?`
5. Read the findings. Expect the firewall analysis to surface items such as the `permit ip any any` ACL applied inbound on the WAN interface, default management credentials, Telnet and plain-HTTP management enabled, a DES/MD5 VPN with a plaintext pre-shared key, and the SNMP community `public`. Expect the IAM analysis to surface `"Principal": "*"` with `"Action": "*"` on all resources, a statement that denies users the ability to set up MFA, and permissions to stop or delete CloudTrail logging.
6. Ask: `Which finding is highest risk and why?`
7. Ground the answer: `For each finding, tell me which file and section it came from.`

### Expected result

Findings reference the actual uploaded files (not invented sources), each with a severity and a justification. The highest-risk callouts are typically the wide-open WAN ACL and the `"Action": "*"` IAM policy, followed by the MFA-denial and audit-logging statements.

### Try also

Ask `What remediation order would you recommend for the top three findings, and which NIST CSF functions do they map to?`

---

## Lab 3: Generate a consultant report (~10 min)

**Objective:** Turn the Lab 2 engagement into consultant deliverables in three formats and locate the downloads.

**Duration:** ~10 minutes

### Steps

1. Open the Lab 2 engagement (click it in the sidebar).
2. In the chat box type: `generate a docx report`. The reply summarizes the run — **Security report generated** for the engagement (DOCX), the findings count, the overall risk rating — with a **Download the report** link.
3. Click the download link and open the DOCX. Check the title page (engagement name and date), client context, scope, executive summary with the overall risk rating, findings with severity, evidence and framework mappings, and recommendations.
4. Type `generate a pptx presentation`, then `generate a pdf report`. Download both the same way.
5. Click **Open reports folder** (sidebar). File Explorer opens at `%APPDATA%\Fortis\reports` — all three files are there.

### Expected result

Three reports generated from the same engagement — identical findings, three presentations (document, slide deck, PDF). Reports persist in the reports folder even if a browser blocks a download link.

### Try also

Type `generate a docx report focusing on access control issues only` and compare it against the full report — focus instructions narrow the analysis.

---

## Lab 4: Persona modes (~10 min)

**Objective:** Compare how the same document is analyzed under different analyst personas.

**Duration:** ~10 minutes

**File used:** `training\sample-lab-docs\large-policy-document.txt` — a 12-section enterprise security policy (acceptable use, access control, authentication, data handling, network, endpoint, application security, incident response, vulnerability management, cloud, physical, training) with appendices mapping to NIST CSF 2.0, ISO 27001, CIS Controls, OWASP Top 10, and SOC 2.

### Steps

1. Click **+ New engagement** and create `Lab 4 — personas`.
2. Attach `training\sample-lab-docs\large-policy-document.txt`.
3. Leave the **Analyst mode** dropdown (left of the chat box) on **General**. Ask: `What are the biggest security gaps in this policy?` Note the consultant-memo style: material findings first, clear explanations.
4. Switch the dropdown to **GRC**. Ask the same question again (the persona applies from your next message).
5. Compare the two replies. In GRC mode Fortis maps each point to a named control with framework and control ID (for example `NIST CSF PR.PS`), frames issues as policy or control gaps in audit-evidence phrasing, and distinguishes gaps from findings from observations.
6. Optional: attach `training\sample-lab-docs\vulnerable-app.py` (a deliberately vulnerable Flask user-management API), switch the dropdown to **Code**, and ask `What vulnerabilities are in this application?` Expect vulnerability classes named by CWE, references to specific functions and lines, data-flow descriptions from source to sink, and minimal fix sketches.

### Expected result

Same document, three professional framings: a consultant memo (General), an audit workpaper with control references (GRC), and a code review (Code). Severity framing shifts with the persona — GRC speaks in control gaps, Code in vulnerability classes.

### Try also

Switch to **OSINT / CTI** on the same policy and ask how the incident-response section would be exercised — answers shift to structured analyst formats.

---

## Lab 5: Structured output & diagrams (~10 min)

**Objective:** Get structured summaries and a rendered architecture or attack-path diagram directly in the chat.

**Duration:** ~10 minutes

### Steps

1. Return to the Lab 2 engagement (firewall and IAM documents attached).
2. Ask: `Give me a structured summary of the findings as a table.` The chat renders a formatted table (severity, finding, source) instead of prose.
3. Ask: `Draw a network architecture diagram based on the firewall configuration.` Fortis replies with a rendered flowchart inside the chat — internet-facing zones, the perimeter, and internal segments.
4. Ask: `Show the attack path an outside attacker would take against this firewall as a diagram.` Compare the two diagrams.
5. Follow up: `Where should network segmentation break this attack path?` — note that the discussion stays anchored to the rendered chart.

### Expected result

Tables render as real tables and diagrams render as charts in the chat. If a diagram cannot be rendered, the raw source appears with a notice — ask Fortis to redraw it.

### Try also

In the Lab 4 engagement ask for `a sequence diagram of the incident response flow from the policy's incident response section` — flowcharts and sequence diagrams are both supported.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| App won't open | `python` is missing from PATH, or the setup script was blocked by execution policy | Reinstall Python ticking **Add to PATH**, then run `powershell -ExecutionPolicy Bypass -File setup.ps1` |
| Model download stalls | Interrupted Wi-Fi, VPN, or proxy during the one-time download | Click the tier again — downloads are resumable and pick up where they stopped; the instructor can also pre-download the tier for the room |
| Replies are very slow | The model loads on first use, and the 4B tier is heavy on CPU-only machines | Wait for the first reply (10–60 s), or open **⚙ Model** and switch to the 0.8B or 2B tier |
| No findings for a file type | The format was not recognized, or it ingested zero chunks | Check for the `ingested: N chunks indexed` confirmation — supported types include PDF, DOCX, PPTX, XLSX, CSV, configs, and code; if N is 0, convert the file or paste its content into the chat |
| Report request does nothing, or replies with a placeholder warning | No model loaded (stub mode), or the engagement has no attached documents yet | Confirm the status dot is green (open **⚙ Model** if not) and attach files first; stub mode never writes a file. Reports always land in `%APPDATA%\Fortis\reports` — use **Open reports folder** (sidebar). You can also trigger generation with `/report [docx\|pptx\|pdf]` |
| Chat history disappeared after switching engagements | Older build — current builds restore the saved transcript per engagement | Restart Fortis: every turn is stored in `engagements.sqlite3` (newest 500) and restored on app start and engagement switch. Use **Clear chat** (top bar) to deliberately erase an engagement's transcript |
| Browser opened instead of the native app window | The WebView2 runtime was unavailable, so the interface fell back to the default browser | These labs work identically in the browser view; to restore the native window, update Microsoft Edge or install the WebView2 Runtime and relaunch `setup.ps1` |

---

For builder-level labs (API, custom frameworks, offline tests), see `00-workshop-guide.md`.
