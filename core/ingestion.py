"""
Parses arbitrary uploaded files into plain text, then chunks them for
embedding. Designed to preserve structure where it matters for security
review (e.g. keep config blocks / code functions together where possible).
"""
import os
import csv
import json
import re
from typing import List

import pymupdf
import docx
import pptx
import openpyxl

from .config import settings
from .token_utils import count_tokens, tokens_to_chars

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".go", ".java", ".c", ".cpp", ".rb", ".php",
    ".yaml", ".yml", ".json", ".conf", ".cfg", ".ini", ".sh", ".tf",
    ".env", ".xml", ".sql", ".log",
}


def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return _extract_pdf(file_path)
    if ext == ".docx":
        return _extract_docx(file_path)
    if ext == ".pptx":
        return _extract_pptx(file_path)
    if ext in (".xlsx", ".xlsm"):
        return _extract_xlsx(file_path)
    if ext == ".csv":
        return _extract_csv(file_path)
    if ext in CODE_EXTENSIONS or ext in (".txt", ".md"):
        return _extract_plaintext(file_path)

    # Fallback: try plaintext decode, otherwise refuse
    try:
        return _extract_plaintext(file_path)
    except UnicodeDecodeError:
        raise ValueError(f"Unsupported or binary file type: {ext}")


def _extract_pdf(path: str) -> str:
    text_parts = []
    with pymupdf.open(path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text_parts.append(f"\n--- Page {page_num} ---\n{page.get_text()}")
    return "".join(text_parts)


def _extract_docx(path: str) -> str:
    d = docx.Document(path)
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def _extract_pptx(path: str) -> str:
    prs = pptx.Presentation(path)
    parts = []
    for i, slide in enumerate(prs.slides, start=1):
        slide_text = [f"\n--- Slide {i} ---"]
        for shape in slide.shapes:
            if shape.has_text_frame:
                slide_text.append(shape.text_frame.text)
            if shape.has_table:
                for row in shape.table.rows:
                    slide_text.append(" | ".join(c.text for c in row.cells))
        parts.append("\n".join(slide_text))
    return "\n".join(parts)


def _extract_xlsx(path: str) -> str:
    wb = openpyxl.load_workbook(path, data_only=True)
    parts = []
    for sheet in wb.worksheets:
        parts.append(f"\n--- Sheet: {sheet.title} ---")
        for row in sheet.iter_rows(values_only=True):
            if any(cell is not None for cell in row):
                parts.append(" | ".join("" if c is None else str(c) for c in row))
    return "\n".join(parts)


def _extract_csv(path: str) -> str:
    parts = []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            parts.append(" | ".join(row))
    return "\n".join(parts)


def _extract_plaintext(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def chunk_text(
    text: str,
    chunk_size_tokens: int = None,
    overlap_tokens: int = None,
) -> List[str]:
    """Token-aware sliding-window chunking. Splits on paragraph boundaries
    where possible so config/code blocks aren't cut mid-statement, falling
    back to a hard token split for very long paragraphs."""
    chunk_size_tokens = chunk_size_tokens or settings.chunk_size_tokens
    overlap_tokens = overlap_tokens or settings.chunk_overlap_tokens

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    def flush():
        if current:
            chunks.append("\n\n".join(current).strip())

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if para_tokens > chunk_size_tokens:
            # hard-split oversized paragraph by character-width approximation
            flush()
            current, current_tokens = [], 0
            chunk_chars = tokens_to_chars(chunk_size_tokens)
            overlap_chars = tokens_to_chars(overlap_tokens)
            step = max(1, chunk_chars - overlap_chars)
            for i in range(0, len(para), step):
                chunks.append(para[i:i + chunk_chars])
            continue

        if current_tokens + para_tokens > chunk_size_tokens:
            flush()
            # start new chunk with overlap from tail of previous chunk
            overlap_text = ""
            if chunks:
                overlap_chars = tokens_to_chars(overlap_tokens)
                overlap_text = chunks[-1][-overlap_chars:]
            current = [overlap_text] if overlap_text else []
            current_tokens = count_tokens(overlap_text) if overlap_text else 0

        current.append(para)
        current_tokens += para_tokens

    flush()
    return [c for c in chunks if c.strip()]


_PAGE_MARKER_RE = re.compile(r"^--- (Page \d{1,4}|Slide \d{1,4}|Sheet: [^\n-]{0,200}?) ---\s*$", re.MULTILINE)

_SANITIZE_RE = re.compile(r"[\r\n\x00-\x1f]+")


def _sanitize_source_name(name: str) -> str:
    """Filenames end up verbatim inside prompt context blocks and reports;
    strip control characters and cap length so a crafted filename cannot
    forge provenance or break rendering."""
    cleaned = _SANITIZE_RE.sub(" ", name or "").strip()
    return cleaned[:120]


def _page_span_label(text: str, start: int, end: int, markers: list = None) -> str:
    """Label for the marker(s) a chunk overlaps, e.g. 'pages 3-4'. The
    markers list is precomputed once per document by the caller."""
    labels = []
    for match in (markers if markers is not None else _PAGE_MARKER_RE.finditer(text)):
        if match.end() <= start:
            continue
        if match.start() >= end:
            break
        labels.append(match.group(1))
    if not labels:
        return ""
    if len(labels) == 1:
        label = labels[0]
        if label.startswith("Page "):
            return f"page {label[5:]}"
        if label.startswith("Slide "):
            return f"slide {label[6:]}"
        return label.lower()
    kinds = {label.split(" ")[0] for label in labels}
    numbers = [label.split(" ")[1] for label in labels]
    if kinds == {"Page"}:
        return f"pages {numbers[0]}-{numbers[-1]}"
    if kinds == {"Slide"}:
        return f"slides {numbers[0]}-{numbers[-1]}"
    return ", ".join(label.lower() for label in labels)


def chunk_text_enriched(text: str, filename: str) -> List[str]:
    """chunk_text() plus a self-describing header per chunk, e.g.
    '[From nginx.conf — pages 3-4]'. PDF/slide/sheet markers survive text
    extraction but are lost by chunking today; the header keeps every chunk
    traceable to its source span, which also helps the map-reduce report
    path where batches are read without retrieval context. Markers are
    scanned once per document (not per chunk) and the filename is sanitized
    before it enters prompt context."""
    filename = _sanitize_source_name(filename)
    chunks = chunk_text(text)
    markers = list(_PAGE_MARKER_RE.finditer(text))
    if not markers:
        return [f"[From {filename}]\n{chunk}" for chunk in chunks]
    last_marker_end = markers[-1].end()
    enriched = []
    cursor = 0
    for chunk in chunks:
        start = text.find(chunk[:80], cursor)
        if start == -1:
            start = cursor
        end = start + len(chunk)
        cursor = end
        label = _page_span_label(text, start, end, markers)
        if not label and start < last_marker_end:
            # chunk sits between markers (leading intro) or trailing overlap;
            # attribute it to the nearest preceding marker
            label = _page_span_label(text, last_marker_end - 1, last_marker_end, markers)
        header = f"[From {filename} — {label}]" if label else f"[From {filename}]"
        enriched.append(f"{header}\n{chunk}")
    return enriched


def summarize_if_large(text: str, max_tokens: int) -> bool:
    """Returns True if the document exceeds max_tokens and should be
    processed via hierarchical summarization instead of flat retrieval."""
    return count_tokens(text) > max_tokens
