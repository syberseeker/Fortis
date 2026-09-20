"""
Shared contract for structured LLM output (diagrams and tables).

Detects when a user's chat message asks for a diagram or a table, supplies
the format instructions injected into the prompt for that intent, and
validates the model's reply so the caller can decide on a repair retry.
"""
import re
from typing import Dict, List, Optional

DiagramKind = str
TableKind = str

_MERMAID_ALLOWED = re.compile(
    r"^(flowchart|graph)\s{1,50}(TB|TD|BT|RL|LR)\b|^(flowchart|graph)\b|^(sequenceDiagram)\b",
    re.IGNORECASE,
)

_MERMAID_FORBIDDEN = (
    "@{", "```", "javascript:", "onerror", "onload", "onclick",
    "onmouseover", "onfocus", "<svg", "<img", "<script", "<iframe",
)

_DIAGRAM_WORDS = (
    r"\b(diagram|topology|flow\s*chart|flowchart|sequence\s+diagram|"
    r"architecture\s+(?:diagram|overview)|mind\s*map|draw|visuali[sz]e|"
    r"network\s+graph|attack\s+path|data\s*flow)\b"
)
_TABLE_WORDS = (
    r"\b(table|tabular|matrix|spreadsheet|grid)\b|"
    r"\b(compare|comparison|versus|vs\.?)\b.*\b(table|matrix|side\s+by\s+side)\b|"
    r"\b(list|break\s*down|summar\w*)\b.*\b(columns?|rows?)\b|"
    r"\bbar\s*chart\b"
)

_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")


def detect_structured_output_request(text: str) -> Optional[str]:
    t = (text or "").lower()
    if not t.strip():
        return None
    if re.search(_DIAGRAM_WORDS, t):
        return "diagram"
    if re.search(_TABLE_WORDS, t):
        return "table"
    return None


FORMAT_INSTRUCTIONS: Dict[str, str] = {
    "diagram": (
        "When a diagram helps answer the request, include exactly ONE mermaid code "
        "block in your markdown reply, fenced like:\n"
        "```mermaid\n...\n```\n"
        "Allowed diagram types ONLY: `flowchart TD`, `flowchart LR` (the keyword "
        "`graph` also works) and `sequenceDiagram`. Rules:\n"
        "- First line must be exactly `flowchart TD`, `flowchart LR` or "
        "`sequenceDiagram`.\n"
        "- Keep node labels short; wrap labels that contain spaces or special "
        "characters in double quotes, e.g. FW[\"Perimeter Firewall\"].\n"
        "- Edges: `-->` for flow, `-.->` for trust/data boundaries, and "
        "`-->|label|` for a short edge label.\n"
        "- Group zones with `subgraph Name[\"Zone label\"]` ... `end`.\n"
        "- Never use the newer shape syntax like A@{ shape: cyl }, icons, images, "
        "or backtick markdown inside labels.\n"
        "- Keep the diagram under about 15 nodes.\n"
        "Example network topology:\n"
        "```mermaid\n"
        "flowchart LR\n"
        "    Internet[\"Internet\"] --> FW[\"Perimeter Firewall\"]\n"
        "    FW --> Web[\"Web Server\"]\n"
        "    FW --> Admin[\"Admin Workstation\"]\n"
        "    Web --> DB[(\"Customer Database\")]\n"
        "    DB -.->|encrypted backups| Backup[\"Backup Storage\"]\n"
        "```\n"
        "If the request is not about structure, flow or architecture, answer "
        "normally without a diagram block."
    ),
    "table": (
        "When tabular data helps answer the request, format it as a GitHub-flavored "
        "markdown table:\n"
        "- First row: header with column names, every cell surrounded by `|`.\n"
        "- Second row: separator like `|---|---|---|` (colons may align columns, "
        "e.g. `|:---|---:|`).\n"
        "- One data row per item; every row starts and ends with `|`.\n"
        "- Escape any literal `|` inside a cell as `\\|`.\n"
        "- Keep cells short; add a one-line lead-in sentence before the table."
    ),
}

REPAIR_INSTRUCTIONS: Dict[str, str] = {
    "diagram": (
        "Your previous reply's mermaid block was not valid. Rewrite the full answer "
        "with ONE corrected mermaid block: first line exactly `flowchart TD` or "
        "`flowchart LR` (or `sequenceDiagram` if a sequence was requested); short "
        "node labels in double quotes; simple `-->` edges; optional `-->|label|` "
        "edge labels; no `@{...}` shapes. Everything else in your answer stays the same."
    ),
    "table": (
        "Your previous reply was not a well-formed markdown table. Reformat the ENTIRE "
        "response as a proper GitHub-flavored markdown table: a header row with column "
        "names between `|` characters, a separator row like `|---|---|---|`, then one "
        "data row per item, every row starting and ending with `|`. Do not include any "
        "text outside the table. Original request: "
    ),
}


def extract_mermaid_blocks(reply: str) -> List[str]:
    if not reply:
        return []
    blocks = re.findall(r"```mermaid\s*\n(.*?)```", reply, re.DOTALL)
    return [b.strip() for b in blocks]


def is_valid_mermaid(code: str) -> bool:
    if not code or not code.strip():
        return False
    lines = [l for l in code.strip().splitlines() if l.strip()]
    if not lines:
        return False
    if not _MERMAID_ALLOWED.match(lines[0].strip()):
        return False
    joined = code
    if any(tok in joined for tok in _MERMAID_FORBIDDEN):
        return False
    if joined.count("[") != joined.count("]"):
        return False
    if joined.count("(") != joined.count(")"):
        return False
    if joined.count('"') % 2 != 0:
        return False
    return True


def reply_has_valid_diagram(reply: str) -> bool:
    blocks = extract_mermaid_blocks(reply)
    if not blocks:
        return False
    return all(is_valid_mermaid(b) for b in blocks)


def is_valid_gfm_table(reply: str) -> bool:
    if not reply:
        return False
    lines = reply.splitlines()
    best = 0
    i = 0
    while i < len(lines):
        if _TABLE_ROW_RE.match(lines[i]):
            j = i
            while j < len(lines) and _TABLE_ROW_RE.match(lines[j]):
                j += 1
            block = lines[i:j]
            if len(block) >= 3 and _TABLE_SEP_RE.match(block[1]):
                cols = _split_row(block[0])
                if len(cols) >= 2 and all(len(_split_row(r)) == len(cols) for r in block[2:]):
                    best = max(best, len(block))
            i = j
        else:
            i += 1
    return best >= 3


def _split_row(line: str) -> List[str]:
    t = line.strip()
    if t.startswith("|"):
        t = t[1:]
    if t.endswith("|"):
        t = t[:-1]
    return [c.strip() for c in t.split("|")]
