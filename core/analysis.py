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

from . import vectorstore, store, progress
from .config import settings
from engine import chat
from .reports.schema import SecurityReport
from .token_utils import count_tokens
from .structured import is_valid_mermaid

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
  "recommendations_summary": [string],
  "diagram": string or null
}

Rules:
- Base every finding strictly on the retrieved excerpts provided. Do not fabricate findings \
not supported by the evidence.
- If the material is entirely benign, it is valid to return zero findings and note that \
in the executive summary.
- Map to a framework control only when there's a clear match; otherwise use null.
- evidence must quote or closely paraphrase the specific excerpt that triggered the finding.
- diagram: include ONLY if the reviewed documents clearly describe a system architecture or \
data flow. Must be a simple mermaid flowchart TD or flowchart LR block source (no fences \
inside the JSON string, no @{...} shape syntax, quoted labels, under ~15 nodes). Otherwise null.
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
  "recommendations_summary": [string],
  "diagram": string or null
}

Rules:
- Every finding in your output must be traceable to at least one candidate finding you were given.
- Merge candidate findings that describe the same underlying issue (even across files) into \
one finding rather than repeating it.
- Add remediation advice for every finding -- the map step did not include this.
- Map to a framework control only when there's a clear match; otherwise use null.
- diagram: include ONLY if the reviewed documents clearly describe a system architecture or \
data flow. Must be a simple mermaid flowchart TD or flowchart LR block source (no fences \
inside the JSON string, no @{...} shape syntax, quoted labels, under ~15 nodes). Otherwise null.
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
    try:
        return await _call_json_once(system_prompt, user_content)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Malformed analysis JSON; retrying once")
    try:
        return await _call_json_once(system_prompt, user_content)
    except (json.JSONDecodeError, ValueError):
        logger.error("Model returned invalid JSON twice")
        raise ValueError("Analysis model returned malformed output. Try again.")


async def _call_json_once(system_prompt: str, user_content: str) -> Dict:
    raw = await chat(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
        temperature=0.1,
        json_mode=True,
    )
    return _parse_json_response(raw)


def _parse_json_response(text: str) -> Dict:
    import re
    pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r",(\s*[}\]])", r"\1", text)
        data = json.loads(text)
    if "diagram" in data and data["diagram"] is not None:
        if not is_valid_mermaid(data["diagram"]):
            logger.warning("Invalid mermaid diagram in LLM response; setting to None")
            data["diagram"] = None
    return data


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

def _effective_batch_tokens() -> int:
    if settings.engine_slow_backend:
        return settings.map_batch_tokens * 2
    return settings.map_batch_tokens


def _plan_map_batches(engagement_id: str, filenames: List[str], max_tokens: int) -> List[Tuple[str, List[List[str]]]]:
    """Pre-plans every file's batches once so progress totals are exact and
    slow backends can use fewer, larger LLM calls."""
    plan: List[Tuple[str, List[List[str]]]] = []
    for filename in filenames:
        chunks = vectorstore.get_document_chunks(engagement_id, filename)
        plan.append((filename, _batch_chunks(chunks, max_tokens)))
    return plan

async def _map_document(filename: str, batches: List[List[str]]) -> Tuple[List[Dict], List[str]]:
    all_findings: List[Dict] = []
    all_key_points: List[str] = []

    for batch_num, batch in enumerate(batches, start=1):
        content = f"=== {filename} (batch {batch_num}/{len(batches)}) ===\n\n" + "\n\n".join(batch)
        data = await _call_json(MAP_SYSTEM_PROMPT, content)
        for finding in data.get("findings", []):
            finding.setdefault("source_file", filename)
            all_findings.append(finding)
        all_key_points.extend(data.get("key_points", []))
        progress.advance(detail=f"{filename} batch {batch_num}/{len(batches)}")

    return all_findings, all_key_points


async def _generate_map_reduce(engagement_id: str, focus_instructions: str) -> SecurityReport:
    filenames = vectorstore.list_engagement_files(engagement_id)
    plan = _plan_map_batches(engagement_id, filenames, _effective_batch_tokens())

    progress.start(
        total=sum(len(batches) for _, batches in plan) + 1,
        detail=f"{sum(len(b) for _, b in plan)} batches across {len(filenames)} file(s)",
    )

    all_candidate_findings: List[Dict] = []
    file_summaries: List[str] = []

    for filename, batches in plan:
        findings, key_points = await _map_document(filename, batches)
        all_candidate_findings.extend(findings)
        if key_points:
            file_summaries.append(f"{filename}: " + "; ".join(key_points))

    progress.update(done=progress.snapshot()["total"] - 1, stage="reduce", detail="Merging candidate findings")
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
    try:
        report = await _generate_security_report_inner(engagement_id, focus_instructions)
    except Exception:
        progress.fail()
        raise
    progress.finish()
    return report


async def _generate_security_report_inner(engagement_id: str, focus_instructions: str = "") -> SecurityReport:
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
        progress.start(total=1, detail="Single-pass analysis")
        return await _generate_flat(engagement_id, focus_instructions)

    logger.info(
        "Engagement %s: %d tokens > threshold, using map-reduce analysis pipeline",
        engagement_id, total_tokens,
    )
    return await _generate_map_reduce(engagement_id, focus_instructions)
