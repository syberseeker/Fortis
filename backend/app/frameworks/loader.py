"""
Loads a user-supplied official standard document (PDF or text) as the
authoritative, verbatim reference text for a framework, replacing the
built-in paraphrased seed corpus for that framework.

Why this exists: the built-in seed corpus (nist_csf.json, owasp_top10.json,
cis_controls.json) is deliberately paraphrased rather than verbatim, per
Anthropic's copyright guidance -- reproducing full official standards text
isn't something Claude does even when a license would technically permit it.
NIST CSF is U.S. government public domain and OWASP Top 10 is CC BY-SA, so
either could legally be reproduced verbatim; CIS Controls has a more
restrictive license. Rather than us shipping copied standards text of
uncertain provenance, you supply the document yourself -- you can freely
download the official NIST CSF and OWASP Top 10 PDFs, or your own licensed
copy of CIS Controls -- and this loader ingests it exactly as published.

Usage (from inside the backend container, or locally with the same Python
environment and CHROMA_PERSIST_DIR pointed at the right place):

    python -m app.frameworks.loader NIST_CSF /path/to/NIST.CSWP.29.pdf
    python -m app.frameworks.loader OWASP_TOP10 /path/to/OWASP-Top-10-2025.pdf
    python -m app.frameworks.loader CIS_CONTROLS /path/to/CIS_Controls_v8.1.pdf

The framework name must be one of NIST_CSF, OWASP_TOP10, CIS_CONTROLS, or a
new custom name -- custom names are also picked up automatically by the RAG
retrieval and report generation, no code changes needed.
"""
import argparse
import sys

from ..ingestion import extract_text, chunk_text
from .. import vectorstore


def load_framework_document(framework: str, file_path: str) -> int:
    text = extract_text(file_path)
    if not text.strip():
        raise ValueError(f"No extractable text found in {file_path}")
    chunks = chunk_text(text, chunk_size_tokens=400, overlap_tokens=50)
    n = vectorstore.load_verbatim_framework_document(framework, file_path, chunks)
    return n


def main():
    parser = argparse.ArgumentParser(
        description="Load an official standard document as verbatim framework reference text."
    )
    parser.add_argument("framework", help="Framework name, e.g. NIST_CSF, OWASP_TOP10, CIS_CONTROLS")
    parser.add_argument("file_path", help="Path to the official PDF or text document")
    args = parser.parse_args()

    n = load_framework_document(args.framework.upper(), args.file_path)
    print(f"Loaded {n} verbatim chunks for framework '{args.framework.upper()}' "
          f"from {args.file_path}. This replaces any built-in paraphrased seed "
          f"for that framework.")


if __name__ == "__main__":
    sys.exit(main())
