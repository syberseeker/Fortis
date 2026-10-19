from typing import Any, Dict, List, Optional

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

ROLE_PRESETS: Dict[str, str] = {
    "general": "Balance breadth and depth in your analysis. Lead with the most material findings, then surface secondary observations only if they affect risk. Default to clear, consultant-style explanations that a technical audience can act on immediately.",
    "grc": "Map every finding to a NAMED control from the retrieved framework context, citing framework and control_id (e.g., \"NIST CSF PR.PS\"). Frame gaps as policy or control gaps using audit-evidence phrasing. Distinguish clearly between a \"gap\" (control missing or partially implemented), a \"finding\" (evidence-backed deviation from expected control state), and an \"observation\" (neutral note without risk implication). Never invent control IDs; cite only controls present in the retrieved context or well-known framework controls.",
    "code": "Identify vulnerability classes by CWE name and ID where confident. Reference the specific function, line number, or symbol from the provided code excerpts. Describe the tainted data flow from source to sink. Provide minimal, concrete fix sketches that show the corrected pattern. Never claim to have executed or run the code.",
    "osint": "Structure answers as entity tables with columns for name, role, source, and confidence. Propose next pivots explicitly for the analyst to pursue. Clearly separate fact from inference from unverified claim. Remind the user that this deployment performs no live scanning or collection; all collection steps are descriptions for the analyst to execute manually.",
    "cti": "Use MITRE ATT&CK technique IDs in TXXXX[.XXX] format where confident. Structure analysis as TTPs: tactic to technique to procedure. Hedge attribution explicitly when evidence is incomplete. Map indicators to the ATT&CK enterprise matrix. Keep campaign or actor claims grounded in the provided excerpts or well-known public reporting.",
}


def _retrieve_hits(
    engagement_id: str,
    query: str,
    top_k_user: Optional[int] = None,
    top_k_framework: Optional[int] = None,
):
    """Runs retrieval at the configured rag_level (basic = dense only;
    standard = hybrid BM25/RRF; full = hybrid + cross-encoder rerank) and
    returns the raw (user_hits, framework_hits) lists in final order."""
    tk_user = top_k_user if top_k_user is not None else settings.top_k_user_doc
    tk_fw = top_k_framework if top_k_framework is not None else settings.top_k_framework
    if settings.rag_level == "basic":
        user_hits = vectorstore.query_user_documents(engagement_id, query, tk_user)
        framework_hits = vectorstore.query_frameworks(query, tk_fw)
    elif settings.rag_level == "full":
        from .rerank import rerank

        cand_user = tk_user * settings.rerank_candidate_multiplier
        cand_fw = tk_fw * settings.rerank_candidate_multiplier
        user_hits = vectorstore.query_user_documents_hybrid(engagement_id, query, cand_user, 1)
        framework_hits = vectorstore.query_frameworks_hybrid(query, cand_fw, 1)
        user_hits = rerank(query, user_hits, tk_user)
        framework_hits = rerank(query, framework_hits, tk_fw)
    else:
        user_hits = vectorstore.query_user_documents_hybrid(
            engagement_id, query, tk_user, settings.rerank_candidate_multiplier
        )
        framework_hits = vectorstore.query_frameworks_hybrid(
            query, tk_fw, settings.rerank_candidate_multiplier
        )
    return user_hits, framework_hits


def _format_context_block(user_hits, framework_hits) -> str:
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


def _distance_or_none(hit: Dict) -> Optional[float]:
    distance = hit.get("distance")
    if distance is None:
        return None
    try:
        return float(distance)
    except (TypeError, ValueError):
        return None


def _structured_hits(hits: List[Dict], kind: str) -> List[Dict[str, Any]]:
    """Normalizes raw retrieval hits into the public provenance shape: id,
    text, full metadata, distance (float|None), and a 1-based rank over the
    final ordered list. Dense/BM25 hits carry no stable chroma id, so one is
    derived from metadata; rank reflects fusion/rerank order, not distance."""
    out: List[Dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        meta = hit.get("metadata") or {}
        hit_id = hit.get("id")
        if hit_id is None:
            if kind == "user":
                hit_id = f"{meta.get('filename')}::{meta.get('chunk_index')}"
            else:
                hit_id = f"{meta.get('framework')}::{meta.get('control_id')}"
        out.append(
            {
                "id": hit_id,
                "text": hit.get("text", ""),
                "metadata": meta,
                "distance": _distance_or_none(hit),
                "rank": rank,
            }
        )
    return out


def retrieve_for_query(
    engagement_id: str,
    query: str,
    *,
    top_k_user: Optional[int] = None,
    top_k_framework: Optional[int] = None,
) -> Dict[str, Any]:
    """Single source of truth for retrieval. Returns the rendered
    context_block (byte-identical to build_context_block for the same
    inputs) plus structured user_hits/framework_hits carrying span-level
    provenance for the evaluation harness. build_context_block delegates
    here."""
    user_hits, framework_hits = _retrieve_hits(
        engagement_id, query, top_k_user=top_k_user, top_k_framework=top_k_framework
    )
    return {
        "context_block": _format_context_block(user_hits, framework_hits),
        "user_hits": _structured_hits(user_hits, "user"),
        "framework_hits": _structured_hits(framework_hits, "framework"),
    }


def build_context_block(engagement_id: str, query: str) -> str:
    return retrieve_for_query(engagement_id, query)["context_block"]


_LLM_HISTORY_WINDOW = 16  # messages sent to the model, regardless of archive size


def build_chat_messages(
    engagement_id: str,
    user_message: str,
    history: List[Dict[str, str]] = None,
    role: str = None,
    retrieval_query: str = None,
    retrieved: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    # Persisted history can be long; the model window stays bounded to the
    # most recent turns regardless of how much the UI sends.
    history = (history or [])[-_LLM_HISTORY_WINDOW:]
    if retrieved is not None:
        context = retrieved.get("context_block", "")
    else:
        context = build_context_block(
            engagement_id, retrieval_query if retrieval_query is not None else user_message
        )

    system_content = SYSTEM_PERSONA
    if role in ROLE_PRESETS:
        system_content = SYSTEM_PERSONA + "\n\n" + ROLE_PRESETS[role]

    messages = [{"role": "system", "content": system_content}]
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
