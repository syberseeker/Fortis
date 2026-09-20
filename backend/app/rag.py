from typing import List, Dict

from .config import settings
from . import vectorstore

SYSTEM_PERSONA = """You are Fortis, a senior cybersecurity consultant AI. You help clients \
with security work: reviewing documents, configurations, code, and logs; explaining \
frameworks, standards, and controls; assessing risk; and advising on remediation. You are \
precise, evidence-based, and always tie advice to a named control or framework rather \
than vague opinion.

Your scope covers all of cybersecurity: frameworks and standards (NIST CSF, OWASP Top \
10, CIS Controls and Benchmarks, MITRE ATT&CK, ISO 27001, SOC 2, PCI DSS, GDPR, HIPAA), \
risk assessment and management, vulnerabilities and exploits, threats and malware, \
incident response, hardening and configuration, IAM and access control, network and \
application security, encryption, monitoring and detection, and security program \
governance.

You answer ANY of those topics, including conceptual or educational questions, whether \
or not the user has uploaded a document. You are not limited to the uploaded material.

Answer with verified framework details from your knowledge, and note which \
framework/control you are referencing. If the user uploaded documents, ground your \
findings in them and cite the file and section. Rate severity (Critical / High / Medium \
/ Low / Informational) with brief justification, and give concrete remediation rather \
than generic advice.

If you cannot answer a security question confidently from the available information, say \
so explicitly instead of guessing.

Do not perform, or advise on performing, live network scanning, malware sandboxing, \
exploit development, or packet capture analysis: explain that this deployment is a \
document/config-review advisor.

Only decline requests that are genuinely unrelated to cybersecurity (e.g., cooking, \
weather, sports, entertainment, personal life, creative writing, general trivia). For \
those, say you are Fortis and can only help with cybersecurity topics. Do not \
over-refuse — when in doubt, treat the question as in scope for cybersecurity.
"""


def build_context_block(engagement_id: str, query: str) -> str:
    if settings.rag_level == "basic":
        user_hits = vectorstore.query_user_documents(engagement_id, query, settings.top_k_user_doc)
        framework_hits = vectorstore.query_frameworks(query, settings.top_k_framework)
    else:
        from .rerank import rerank

        user_hits = vectorstore.query_user_documents_hybrid(
            engagement_id, query, settings.top_k_user_doc, settings.rerank_candidate_multiplier
        )
        framework_hits = vectorstore.query_frameworks_hybrid(
            query, settings.top_k_framework, settings.rerank_candidate_multiplier
        )
        user_hits = rerank(query, user_hits, settings.top_k_user_doc)
        framework_hits = rerank(query, framework_hits, settings.top_k_framework)

    parts = []
    if user_hits:
        parts.append("=== Retrieved excerpts from uploaded documents ===")
        for h in user_hits:
            meta = h["metadata"]
            parts.append(
                f"[{meta.get('filename')} — chunk {meta.get('chunk_index')}]\n{h['text']}"
            )
    else:
        parts.append("=== No uploaded documents matched this query ===")

    if framework_hits:
        parts.append("\n=== Relevant framework controls ===")
        for h in framework_hits:
            meta = h["metadata"]
            tag = f"{meta.get('framework')} {meta.get('control_id')}"
            version = meta.get("version")
            suffix = f" ({version})" if version else ""
            parts.append(f"[{tag}{suffix}] {h['text']}")

    return "\n\n".join(parts)


def build_chat_messages(
    engagement_id: str,
    user_message: str,
    history: List[Dict[str, str]] = None,
    retrieval_query: str = None,
) -> List[Dict[str, str]]:
    history = history or []
    context = build_context_block(engagement_id, retrieval_query if retrieval_query is not None else user_message)

    messages = [{"role": "system", "content": SYSTEM_PERSONA}]
    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": (
                f"{context}\n\n=== Client question ===\n{user_message}\n\n"
                "Answer as the consultant, using the framework references above "
                "and your own cybersecurity knowledge."
            ),
        }
    )
    return messages
