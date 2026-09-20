"""
Golden live-evaluation suite for the AI Cybersecurity Advisor.

Runs a fixed set of adversarial/advisory questions against the LIVE backend
/RAG pipeline (not mocks) and scores each answer against expected properties:
presence of a named framework/control, required structure (e.g. tables),
refusals for out-of-scope queries, and non-refusal for in-scope security
queries.

Usage:
    python -m scripts.golden_eval [--base-url http://localhost:8010] [--keep]

Creates a throwaway client + engagement, runs every case, prints a PASS/FAIL
table, and by default deletes the engagement (use --keep to retain it for
manual inspection).

Exit code 0 if all cases pass, 1 if any fail.

Expected runtime: a few seconds per question on qwen3:8b (total is ~15-25 min
for the full set). Re-run after any change to rag.py, llm.py, the persona,
the framework corpus, or after a model swap -- it is the regression gate.
"""
import argparse
import json
import sys
import time
import urllib.request
import urllib.error

DEFAULT_BASE_URL = "http://localhost:8010"

CID = "Golden-Eval"
EID = "golden-live-eval"


def _post(base, path, payload, timeout=600):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _delete(base, path):
    try:
        req = urllib.request.Request(base + path, method="DELETE")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError:
        return None


# Each case:
#   name       - short label (printed in results)
#   q          - the user message
#   must_any   - list of ("any" groups); EACH group must pass, where a group
#                passes if ANY of its candidate substrings is present (ci)
#   must_not   - list of substrings; ANY present fails the case
#   table      - if True, the answer must contain a markdown table
CASES = [
    {
        "name": "NIST CSF v2 asset inventory",
        "q": "Which NIST CSF 2.0 Identify function governs asset inventory and reducing cybersecurity risk?  Give me a table of the most relevant subcategories and how a small firm implements each.",
        "must_any": [["nist", "cSF"], ["id.", "GV.RM", "GV.AM"]],
        "table": True,
    },
    {
        "name": "CIS Controls v8 email/web safeguards",
        "q": "Map the CIS Controls v8 Safeguards relevant to email and web browsing protections for a mid-size company.",
        "must_any": [["CIS", "CIS Controls"], ["safeguard", "Safeguard"]],
        "table": True,
    },
    {
        "name": "ISO 27001 backup control",
        "q": "Which ISO/IEC 27001:2022 Annex A control covers information backup and how should we test restores?",
        "must_any": [["A.8.13"]],
    },
    {
        "name": "ISO 27001 asset inventory",
        "q": "Which Annex A control of ISO 27001:2022 requires an inventory of information assets?",
        "must_any": [["A.5.9"]],
    },
    {
        "name": "PCI DSS card data storage",
        "q": "Under PCI DSS 4.0, how much cardholder data may we store and how must PANs be protected at rest?",
        "must_any": [["req 3", "requirement 3", "req3"], ["pan", "sad"]],
    },
    {
        "name": "PCI DSS MFA",
        "q": "Does PCI DSS 4.0 require multi-factor authentication for remote access into the cardholder data environment?",
        "must_any": [["mfa", "multi-factor", "multi factor"], ["req 8", "requirement 8", "8.4.2"]],
    },
    {
        "name": "SOC 2 monitoring/incident response",
        "q": "Which SOC 2 trust services criteria point to system monitoring and incident response?",
        "must_any": [["CC7"]],
    },
    {
        "name": "SOC 2 availability",
        "q": "How does SOC 2 assess availability of a service provider's system?",
        "must_any": [["a1", "availability"]],
    },
    {
        "name": "GDPR breach notification",
        "q": "What are the GDPR notification deadlines and requirements when a personal data breach occurs?",
        "must_any": [["72"], ["supervisory authority", "supervisory authority"]],
    },
    {
        "name": "GDPR DPIA",
        "q": "When is a data protection impact assessment mandatory under GDPR and what should it include?",
        "must_any": [["dpia", "impact assessment"], ["art. 35", "article 35", "art. 35"]],
    },
    {
        "name": "HIPAA technical safeguards",
        "q": "What technical safeguards does the HIPAA Security Rule require to protect ePHI on remote access?",
        "must_any": [["164.312", "technical safeguard"], ["encryption", "encrypt"]],
    },
    {
        "name": "HIPAA risk analysis",
        "q": "What kind of risk analysis does HIPAA 164.308(a)(1) require and how is it documented?",
        "must_any": [["164.308", "risk analysis"]],
    },
    {
        "name": "NIST 800-53 least privilege",
        "q": "Which NIST SP 800-53 r5 control implements least privilege for accounts?",
        "must_any": [["AC-6"]],
    },
    {
        "name": "NIST 800-53 boundary protection",
        "q": "Which NIST SP 800-53 control is the primary control for network boundary protection and zoning?",
        "must_any": [["SC-7"]],
    },
    {
        "name": "CWE SQL injection",
        "q": "Which CWE covers SQL injection and what are the top mitigations for it?",
        "must_any": [["CWE-89"], ["parameter", "prepared statement", "parameterized"]],
    },
    {
        "name": "CWE XSS",
        "q": "Which CWE identifier is used for cross-site scripting and what category in the Top 25 does it belong to?",
        "must_any": [["CWE-79", "cross-site scripting"]],
    },
    {
        "name": "MITRE ATT&CK ransomware lifecycle",
        "q": "Describe the typical MITRE ATT&CK tactics and techniques used across a ransomware attack lifecycle.",
        "must_any": [["attack", "att&ck"], ["T1486", "T1490", "T1059", "initial access"]],
        "table": True,
    },
    {
        "name": "OWASP injection category",
        "q": "Which OWASP Top 10 (2021) category groups injection flaws and what do they have in common?",
        "must_any": [["a03", "injection"]],
    },
    {
        "name": "Risk assessment matrix",
        "q": "Assess the risk of an internet-facing RDP server with weak credentials and no patches. Present a likelihood/impact risk matrix table and recommend remediation priority.",
        "must_any": [["risk", "rdp", "high"]],
        "table": True,
    },
    {
        "name": "Ransomware mitigations",
        "q": "What are the most effective mitigations to defend against ransomware?",
        "must_any": [["backup", "3-2-1"], ["segment", "kill switch", "controls", "protection"]],
    },
    {
        "name": "NIST 800-63B passwords",
        "q": "What does NIST SP 800-63B say about password policy: length, complexity, and expiration?",
        "must_any": [["800-63b", "63B"], ["length", "15"]],
    },
    {
        "name": "Off-topic refused (recipe)",
        "q": "Tell me how to make a perfect chicken risotto at home.",
        "must_not": ["risotto", "arborio", "stock", "milanese"],
        "must_any": [["cybersecurity", "cyber security", "security", "not able", "can't", "cannot", "scope"]],
    },
    {
        "name": "Off-topic refused (sport)",
        "q": "Who is going to win the next World Cup and by how many goals?",
        "must_not": ["win the world cup", "by 2", "by 3 goals"],
        "must_any": [["cybersecurity", "security", "not able", "can't", "cannot", "scope"]],
    },
    {
        "name": "SQLi answered (in-scope defence)",
        "q": "Why is SQL injection dangerous and how do I eliminate it from my web application?",
        "must_not": ["cannot help", "can't help", "unable to help", "not able to help", "can't answer"],
        "must_any": [["sql"], ["injection"]],
    },
]


def _norm(s):
    return s.lower()


def _has_table(reply):
    lines = [ln for ln in reply.splitlines() if ln.strip().startswith("|")]
    return len(lines) >= 2 and ("---" in reply or "| ---" in reply)


def run_case(case, base, engagement_id):
    payload = {"engagement_id": engagement_id, "message": case["q"], "history": []}
    t0 = time.monotonic()
    try:
        resp = _post(base, "/chat", payload)
    except Exception as exc:  # network/server error -> FAIL, not crash
        return {"case": case["name"], "ok": False, "why": f"request failed: {exc.__class__.__name__}: {exc}", "secs": round(time.monotonic() - t0)}
    reply = resp.get("reply", "")
    low = _norm(reply)
    norm_c = lambda s: _norm(s)

    failures = []

    if case.get("table") and not _has_table(reply):
        failures.append("expected a markdown table")

    for group in case.get("must_any", []):
        if not any(norm_c(c) in low for c in group):
            failures.append("missing any of: " + " | ".join(group))

    for bad in case.get("must_not", []):
        if norm_c(bad) in low:
            failures.append("must not contain: " + bad)

    why = "; ".join(failures) if failures else ""
    return {"case": case["name"], "ok": not why, "why": why, "secs": round(time.monotonic() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--keep", action="store_true", help="do not delete the golden engagement afterward")
    ap.add_argument("--case", default=None, help="run only cases whose name contains this substring")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")

    print(f"[setup] creating golden engagement at {base}")
    _post(base, "/engagements/clients", {"name": CID})
    engagement = _post(base, "/engagements", {"client_name": CID, "engagement_name": "Golden Live Eval"})
    engagement_id = engagement["id"]
    print(f"[setup] engagement id {engagement_id}")

    results = []
    for case in CASES:
        if args.case and args.case.lower() not in case["name"].lower():
            continue
        print(f"[run] {case['name']} ...", flush=True)
        results.append(run_case(case, base, engagement_id))

    passed = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    print("\n" + "=" * 100)
    print(f"{'CASE':<42} {'RESULT':<7} {'TIME/s':<8} WHY")
    print("-" * 100)
    for r in results:
        print(f"{r['case']:<42} {'PASS' if r['ok'] else 'FAIL':<7} {r['secs']:<8.1f} {r['why']}")
    print("-" * 100)
    if results:
        print(f"TOTAL {len(results)}  PASS {len(passed)}  FAIL {len(failed)}  "
              f"score {len(passed)}/{len(results)} ({100.0 * len(passed) / len(results):.0f}%)")
    else:
        print("No cases matched the filter.")

    if not args.keep:
        _delete(base, f"/engagements/{engagement_id}")
        print(f"[cleanup] deleted engagement {engagement_id}")
    else:
        print(f"[skip] --keep: engagement {engagement_id} retained")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()