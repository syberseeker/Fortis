"""Cybersecurity topic guardrail shared by the chat endpoints.

The desktop app must refuse non-cybersecurity topics with a deterministic
keyword gate (not just the persona prompt), and retry once when a model
wrongly refuses an on-topic security question.
"""
import re
from typing import Tuple

CYBERSECURITY_KEYWORDS = [
    r"\b(security|cyber(?:security)?|infosec|application\s*security|appsec)s?\b",
    r"\b(vulnerab(?:ility|le)|exploit|att(?:ack|&?ck)|threat|malware|ransomware|phishing)s?\b",
    r"\b(risk|compliance|audit|governance|policy|incident|breach|compromise)s?\b",
    r"\b(nist|owasp|mitre|cis\s*controls?|iso\s*27001|soc\s*2|pci[\s-]*dss|gdpr|hipaa)\b",
    r"\b(firewall|ids|ips|siem|soc|encryption|encrypted|authentication|authorization)s?\b",
    r"\b(access\s*control|iam|identity|privilege|zero\s*trust|vpn|tls|ssl)s?\b",
    r"\b(patch(?:es)?|hardening|configuration|misconfiguration|baseline)s?\b",
    r"\b(logs?|monitoring|detection|detect|respond|recovery|recover|protect|identify)\b",
    r"\b(sql|sqli|injection|xss|csrf|ssrf|deserializ|payload|webshell|privilege\s*escap)\b",
    r"\b(iv|traversal|directory\s*traversal|code\s*injection|command\s*injection)\b",
    r"\b(reverse\s*shell|shell|buffer\s*overflow|malicious|trojan|rootkit|botnet)s?\b",
    r"\b(reports?|findings?|remediat\w*|recommendations?|executive\s*summary|risk\s*rating)\b",
    r"\b(upload|document|config|code\s*review|architecture|network\s*diagram)s?\b",
    r"\b(secur(?:e|ity)|harden|encrypt|protect)s?\b",
]

OFF_TOPIC_RESPONSE = (
    "I'm Fortis, a cybersecurity advisory assistant. I can only help with "
    "security-related topics such as:\n\n"
    "- Document, configuration, and code security reviews\n"
    "- Framework mapping (NIST CSF, OWASP Top 10, CIS Controls)\n"
    "- Risk assessment and remediation advice\n"
    "- Security report generation\n\n"
    "Please ask a cybersecurity question or upload a document for review."
)

_REFUSAL_MARKERS = (
    "i can only help with security-related topics",
    "please ask a cybersecurity question",
)

_COMPILED = [re.compile(p, re.IGNORECASE) for p in CYBERSECURITY_KEYWORDS]

# Non-Latin scripts: the keyword list is English-only, so a Chinese or Arabic
# security question would false-refuse. Let those through and rely on the
# persona prompt (which understands the topic) instead of a keyword miss.
_NON_LATIN_SCRIPT_RE = re.compile(r"[\u0600-\u06FF\u3000-\u9FFF\uFF00-\uFFEF]")


def is_cybersecurity_related(text: str) -> bool:
    t = (text or "").lower()
    if _NON_LATIN_SCRIPT_RE.search(t):
        return True
    return any(p.search(t) for p in _COMPILED)


def is_offtopic_refusal(reply: str) -> bool:
    lowered = (reply or "").lower()
    return any(m in lowered for m in _REFUSAL_MARKERS)


def forced_answer_prompt(original_request: str) -> str:
    """Retry instruction appended when the model wrongly refuses an on-topic
    security question."""
    return (
        "This is a cybersecurity topic. Answer the user's question directly "
        "and completely. Do not refuse. Original request: " + (original_request or "")
    )


def classify(message: str) -> Tuple[str, str]:
    """Returns (kind, reply): 'ok' when the message is in scope; 'offtopic'
    plus the refusal text otherwise."""
    if is_cybersecurity_related(message):
        return "ok", ""
    return "offtopic", OFF_TOPIC_RESPONSE
