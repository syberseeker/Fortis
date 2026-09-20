"""
Turns an engagement's uploaded documents into a structured SecurityReport.

Two paths, chosen automatically by total content size:

- Flat path (small engagements, <= settings.hierarchical_threshold_tokens of
  uploaded content): every chunk fits comfortably in one context window
  alongside framework references, so a single LLM call produces the full
  report directly. Fast, cheap, and simplest to reason about.

- Map-reduce path (large engagements): flat retrieval would either truncate
  content or blow the context window, silently dropping findings. Instead:
    MAP    -- each file's chunks are grouped into token-bounded batches;
              each batch is analyzed independently for candidate findings
              plus a short summary, grounded only in that batch's text.
    REDUCE -- all candidate findings + per-file summaries + framework
              reference context are combined in one final pass that dedupes
              overlapping findings, assigns framework mappings, and writes
              the executive summary / scope / overall risk rating.

Both paths return the same SecurityReport schema, so callers (the report
router) don't need to know or care which path ran.
"""
import json
import logging
from typing import List, Dict, Tuple

from . import vectorstore, store
from .config import settings
from .llm import chat
from .reports.schema import SecurityReport
from .token_utils import count_tokens

logger = logging.getLogger(__name__)

FRAMEWORK_PROBE_QUERIES = [
    "security vulnerability weakness misconfiguration",
    "access control authentication authorization",
    "encryption data protection secrets credentials",
    "network exposure firewall port service",
    "logging monitoring incident response",
]

# ---- Prompts ----------------------------------------------------------

FLAT_SYSTEM_PROMPT = """You are Fortis, a senior cybersecurity consultant. You will be \
given excerpts from a client's uploaded documents/configs/code plus relevant security \
framework controls. Produce a structured security assessment as JSON ONLY -- no prose \
outside the JSON, no markdown fences.

Respond with a single JSON object matching exactly this shape:
{
  "title": string,
  "client_context": string,
  "scope": string,
  "executive_summary": string,
  "findings": [
    {
      "title": string,
      "severity": "Critical" | "High" | "Medium" | "Low" | "Informational",
      "description": string,
      "evidence": string,
      "source_file": string or null,
      "framework": "NIST_CSF" | "OWASP_TOP10" | "CIS_CONTROLS" | null,
      "control_id": string or null,
      "remediation": string
    }
  ],
  "overall_risk_rating": "Critical" | "High" | "Medium" | "Low" | "Informational",
  "recommendations_summary": [string]
}

Rules:
- Base every finding strictly on the retrieved excerpts provided. Do not fabricate findings \
not supported by the evidence.
- If the material is entirely benign, it is valid to return zero findings and note that \
in the executive summary.
- Map to a framework control only when there's a clear match; otherwise use null.
- evidence must quote or closely paraphrase the specific excerpt that triggered the finding.
"""

MAP_SYSTEM_PROMPT = """You are Fortis, a senior cybersecurity consultant, reviewing ONE \
batch of excerpts from a single client document. You do not have the whole document or \
the client's other documents -- only what is given below. Extract candidate findings from \
THIS batch only. Do not attempt to write an executive summary or map to frameworks yet; \
that happens in a later step once all batches are combined.

Respond with a single JSON object matching exactly this shape:
{
  "key_points": [string],
  "findings": [
    {
      "title": string,
      "severity": "Critical" | "High" | "Medium" | "Low" | "Informational",
      "description": string,
      "evidence": string,
      "source_file": string
    }
  ]
}

Rules:
- Base findings strictly on the excerpts in this batch. Do not fabricate.
- key_points: 1-3 short bullet-style strings capturing what this batch covers, for use in \
a later per-file summary.
- If this batch is benign, return an empty findings list -- that is a valid and expected result.
"""

REDUCE_SYSTEM_PROMPT = """You are Fortis, a senior cybersecurity consultant. You have already \
reviewed a client's full document set in batches and extracted candidate findings from each \
batch. Your job now is to produce the FINAL structured security assessment: merge duplicate \
or overlapping candidate findings, discard any that are too vague or unsupported to state \
confidently, map each surviving finding to the most relevant framework control from the \
framework reference context provided (or null if no clear match), and write the executive \
summary, scope, client context, overall risk rating, and recommendations.

Respond with a single JSON object matching exactly this shape:
{
  "title": string,
  "client_context": string,
  "scope": string,
  "executive_summary": string,
  "findings": [
    {
      "title": string,
      "severity": "Critical" | "High" | "Medium" | "Low" | "Informational",
      "description": string,
      "evidence": string,
      "source_file": string or null,
      "framework": "NIST_CSF" | "OWASP_TOP10" | "CIS_CONTROLS" | null,
      "control_id": string or null,
      "remediation": string
    }
  ],
  "overall_risk_rating": "Critical" | "High" | "Medium" | "Low" | "Informational",
  "recommendations_summary": [string]
}

Rules:
- Every finding in your output must be traceable to at least one candidate finding you were given.
- Merge candidate findings that describe the same underlying issue (even across files) into \
one finding rather than repeating it.
- Add remediation advice for every finding -- the map step did not include this.
- Map to a framework control only when there's a clear match; otherwise use null.
"""


def _token_count(text: str) -> int:
    return count_tokens(text)


def _batch_chunks(chunks: List[str], max_tokens: int) -> List[List[str]]:
    """Groups consecutive chunks into batches that each fit under max_tokens."""
    batches: List[List[str]] = []
    current: List[str] = []
    current_tokens = 0
    for chunk in chunks:
        chunk_tokens = _token_count(chunk)
        if current and current_tokens + chunk_tokens > max_tokens:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(chunk)
        current_tokens += chunk_tokens
    if current:
        batches.append(current)
    return batches


def _framework_context_block() -> str:
    seen: Dict[Tuple[str, str], Dict] = {}
    for q in FRAMEWORK_PROBE_QUERIES:
        for h in vectorstore.query_frameworks(q, top_k=2):
            meta = h["metadata"]
            seen[(meta["framework"], meta["control_id"])] = h

    parts = ["=== Framework reference controls ==="]
    for (_, _), h in sorted(seen.items()):
        meta = h["metadata"]
        tag = f"{meta.get('framework')} {meta.get('control_id')}"
        version = meta.get("version")
        suffix = f" ({version})" if version else ""
        parts.append(f"[{tag}{suffix}] {h['text']}")
    return "\n".join(parts)


def _engagement_header(engagement_id: str) -> str:
    engagement = store.get_engagement(engagement_id)
    if not engagement:
        return ""
    return (
        f"=== Engagement ===\nClient: {engagement['client_name']}\n"
        f"Engagement: {engagement['name']}\n"
        f"Notes: {engagement.get('notes') or 'none'}\n\n"
    )


async def _call_json(system_prompt: str, user_content: str) -> Dict:
    raw = await chat(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
        temperature=0.1,
        json_mode=True,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Model did not return valid JSON (length=%d)", len(raw))
        raise ValueError("Analysis model returned malformed output. Try again.")


# ---- Flat path (small engagements) --------------------------------------

async def _generate_flat(engagement_id: str, focus_instructions: str) -> SecurityReport:
    filenames = vectorstore.list_engagement_files(engagement_id)
    parts = ["=== Uploaded document excerpts ==="]
    for filename in filenames:
        for i, chunk in enumerate(vectorstore.get_document_chunks(engagement_id, filename)):
            parts.append(f"[{filename} — chunk {i}]\n{chunk}")

    user_content = (
        _engagement_header(engagement_id)
        + "\n\n".join(parts)
        + "\n\n"
        + _framework_context_block()
    )
    if focus_instructions:
        user_content += f"\n\n=== Additional client instructions ===\n{focus_instructions}"

    data = await _call_json(FLAT_SYSTEM_PROMPT, user_content)
    return SecurityReport.model_validate(data)


# ---- Map-reduce path (large engagements) ---------------------------------

async def _map_document(filename: str, chunks: List[str]) -> Tuple[List[Dict], List[str]]:
    batches = _batch_chunks(chunks, settings.map_batch_tokens)
    all_findings: List[Dict] = []
    all_key_points: List[str] = []

    for batch_num, batch in enumerate(batches, start=1):
        content = f"=== {filename} (batch {batch_num}/{len(batches)}) ===\n\n" + "\n\n".join(batch)
        data = await _call_json(MAP_SYSTEM_PROMPT, content)
        for finding in data.get("findings", []):
            finding.setdefault("source_file", filename)
            all_findings.append(finding)
        all_key_points.extend(data.get("key_points", []))

    return all_findings, all_key_points


async def _generate_map_reduce(engagement_id: str, focus_instructions: str) -> SecurityReport:
    filenames = vectorstore.list_engagement_files(engagement_id)

    all_candidate_findings: List[Dict] = []
    file_summaries: List[str] = []

    for filename in filenames:
        chunks = vectorstore.get_document_chunks(engagement_id, filename)
        findings, key_points = await _map_document(filename, chunks)
        all_candidate_findings.extend(findings)
        if key_points:
            file_summaries.append(f"{filename}: " + "; ".join(key_points))

    reduce_parts = [_engagement_header(engagement_id)]
    reduce_parts.append("=== Per-file summaries ===\n" + "\n".join(file_summaries))
    reduce_parts.append(
        "=== Candidate findings extracted from all batches ===\n"
        + json.dumps(all_candidate_findings, indent=2)
    )
    reduce_parts.append(_framework_context_block())
    if focus_instructions:
        reduce_parts.append(f"=== Additional client instructions ===\n{focus_instructions}")

    user_content = "\n\n".join(reduce_parts)
    data = await _call_json(REDUCE_SYSTEM_PROMPT, user_content)
    return SecurityReport.model_validate(data)


# ---- Entry point ----------------------------------------------------------

async def generate_security_report(engagement_id: str, focus_instructions: str = "") -> SecurityReport:
    filenames = vectorstore.list_engagement_files(engagement_id)
    if not filenames:
        raise ValueError("No documents have been uploaded for this engagement yet.")

    total_tokens = 0
    for filename in filenames:
        for chunk in vectorstore.get_document_chunks(engagement_id, filename):
            total_tokens += _token_count(chunk)

    if total_tokens <= settings.hierarchical_threshold_tokens:
        logger.info(
            "Engagement %s: %d tokens <= threshold, using flat analysis pass",
            engagement_id, total_tokens,
        )
        return await _generate_flat(engagement_id, focus_instructions)

    logger.info(
        "Engagement %s: %d tokens > threshold, using map-reduce analysis pipeline",
        engagement_id, total_tokens,
    )
    return await _generate_map_reduce(engagement_id, focus_instructions)
