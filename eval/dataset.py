"""Golden QA dataset loading, schema validation, evidence-anchor
verification, and sha256 content manifests.

Dataset schema (schema_version "1.0") is authored separately; the loader
degrades to a clear FileNotFoundError when the packaged file is absent.
Pure stdlib: no core imports, everything offline."""
import hashlib
import json
import os
import re

_WS_RE = re.compile(r"\s+")

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_EVAL_DIR)
DATASETS_DIR = os.path.join(_EVAL_DIR, "datasets")
DEFAULT_GOLDEN_QA_PATH = os.path.join(DATASETS_DIR, "golden_qa.json")
MANIFEST_NAME = "manifest.json"
SAMPLE_DOCS_DIR = os.path.join(REPO_ROOT, "training", "sample-lab-docs")
FRAMEWORKS_DIR = os.path.join(REPO_ROOT, "core", "frameworks")

SCHEMA_VERSION = "1.0"
VALID_KINDS = ("doc_qa", "framework_qa", "abstention")
VALID_DIFFICULTIES = ("easy", "medium", "hard")
REQUIRED_FIELDS = (
    "id",
    "kind",
    "question",
    "source_docs",
    "evidence_anchors",
    "gold_claims",
    "gold_control_ids",
    "abstention",
    "expected_behavior",
    "difficulty",
)
_LIST_FIELDS = ("source_docs", "evidence_anchors", "gold_claims")


def _norm(text) -> str:
    return _WS_RE.sub(" ", str(text or "").casefold()).strip()


def validate_item(item):
    """Return a list of schema problems for one dataset item ([] = valid)."""
    if not isinstance(item, dict):
        return ["item is not a JSON object"]

    problems = []
    for field in REQUIRED_FIELDS:
        if field not in item:
            problems.append(f"missing field '{field}'")

    item_id = item.get("id")
    if "id" in item and (not isinstance(item_id, str) or not item_id.strip()):
        problems.append("id must be a non-empty string")

    if "kind" in item and item["kind"] not in VALID_KINDS:
        problems.append(
            f"invalid kind {item['kind']!r} (expected one of {VALID_KINDS})"
        )

    question = item.get("question")
    if "question" in item and (
        not isinstance(question, str) or not question.strip()
    ):
        problems.append("question must be a non-empty string")

    for field in _LIST_FIELDS:
        if field in item and not isinstance(item[field], list):
            problems.append(f"{field} must be a list")
            continue
        if field in item:
            for i, entry in enumerate(item[field]):
                if not isinstance(entry, str) or not entry.strip():
                    problems.append(f"{field}[{i}] must be a non-empty string")

    if "gold_control_ids" in item:
        if not isinstance(item["gold_control_ids"], list):
            problems.append("gold_control_ids must be a list")
        else:
            for i, pair in enumerate(item["gold_control_ids"]):
                if not isinstance(pair, dict):
                    problems.append(f"gold_control_ids[{i}] must be an object")
                    continue
                for key in ("framework", "control_id"):
                    value = pair.get(key)
                    if not isinstance(value, str) or not value.strip():
                        problems.append(
                            f"gold_control_ids[{i}] needs a non-empty '{key}' string"
                        )

    if "abstention" in item and not isinstance(item["abstention"], bool):
        problems.append("abstention must be a boolean")

    if (
        item.get("abstention") is True
        and isinstance(item.get("evidence_anchors"), list)
        and item["evidence_anchors"]
    ):
        problems.append("abstention items must have empty evidence_anchors")

    if (
        "expected_behavior" in item
        and item["expected_behavior"] is not None
        and not isinstance(item["expected_behavior"], str)
    ):
        problems.append("expected_behavior must be a string or null")

    if "difficulty" in item and item["difficulty"] not in VALID_DIFFICULTIES:
        problems.append(
            f"invalid difficulty {item['difficulty']!r} "
            f"(expected one of {VALID_DIFFICULTIES})"
        )

    return problems


def list_sample_docs():
    if not os.path.isdir(SAMPLE_DOCS_DIR):
        return []
    return sorted(
        name
        for name in os.listdir(SAMPLE_DOCS_DIR)
        if os.path.isfile(os.path.join(SAMPLE_DOCS_DIR, name))
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_text_file(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _framework_corpus_text():
    parts = []
    for name in sorted(os.listdir(FRAMEWORKS_DIR)):
        if name.endswith(".json"):
            parts.append(_load_text_file(os.path.join(FRAMEWORKS_DIR, name)))
    return _norm("\n".join(parts))


def _verify_evidence_anchors(items):
    offenders = []
    doc_texts = {}
    framework_text = None
    for item in items:
        item_id = item.get("id")
        kind = item.get("kind")
        anchors = item.get("evidence_anchors") or []
        if not anchors:
            continue
        if kind == "doc_qa":
            if not item.get("source_docs"):
                offenders.append(
                    f"{item_id}: doc_qa item has evidence_anchors but no source_docs"
                )
                continue
            for anchor in anchors:
                found = False
                for doc_name in item["source_docs"]:
                    if doc_name not in doc_texts:
                        path = os.path.join(SAMPLE_DOCS_DIR, doc_name)
                        doc_texts[doc_name] = (
                            _norm(_load_text_file(path))
                            if os.path.isfile(path)
                            else None
                        )
                    text = doc_texts[doc_name]
                    if text is None:
                        offenders.append(
                            f"{item_id}: source doc not found in training/sample-lab-docs: {doc_name}"
                        )
                    elif _norm(anchor) in text:
                        found = True
                if not found:
                    offenders.append(
                        f"{item_id}: evidence anchor not found in source docs: {anchor!r}"
                    )
        elif kind == "framework_qa":
            if framework_text is None:
                framework_text = _framework_corpus_text()
            for anchor in anchors:
                if _norm(anchor) not in framework_text:
                    offenders.append(
                        f"{item_id}: evidence anchor not found in any core/frameworks/*.json seed text: {anchor!r}"
                    )
    return offenders


def load_golden_dataset(path=None, verify_anchors=False):
    """Load and schema-validate the golden QA dataset. With
    verify_anchors=True every evidence anchor must occur (casefolded,
    whitespace-normalized) in its source doc (doc_qa) or in some
    core/frameworks/*.json seed text (framework_qa); offenders are listed
    in a ValueError."""
    path = path or DEFAULT_GOLDEN_QA_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Golden QA dataset not found at {path!r}. It is authored "
            "separately (eval/datasets/golden_qa.json); create it or pass an "
            "explicit path via load_golden_dataset(path=...)."
        )
    with open(path, "r", encoding="utf-8") as fh:
        dataset = json.load(fh)

    if not isinstance(dataset, dict):
        raise ValueError("golden dataset must be a JSON object")
    if dataset.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {dataset.get('schema_version')!r} "
            f"(this harness implements {SCHEMA_VERSION!r})"
        )
    items = dataset.get("items")
    if not isinstance(items, list):
        raise ValueError("golden dataset must carry an 'items' list")

    problems = []
    seen_ids = set()
    for index, item in enumerate(items):
        for problem in validate_item(item):
            problems.append(f"item #{index} ({item.get('id') if isinstance(item, dict) else '?'}): {problem}")
        if isinstance(item, dict):
            item_id = item.get("id")
            if isinstance(item_id, str) and item_id:
                if item_id in seen_ids:
                    problems.append(f"item #{index}: duplicate id '{item_id}'")
                seen_ids.add(item_id)
    if problems:
        raise ValueError(
            "golden dataset schema validation failed:\n" + "\n".join(problems)
        )

    if verify_anchors:
        offenders = _verify_evidence_anchors(items)
        if offenders:
            raise ValueError(
                "evidence anchor verification failed:\n" + "\n".join(offenders)
            )

    return dataset


def build_manifest(dataset, files):
    """files: dict of path -> sha256 hex (use sha256_file)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "description": (dataset or {}).get("description", ""),
            "n_items": len((dataset or {}).get("items") or []),
        },
        "files": {str(path): sha for path, sha in sorted((files or {}).items())},
    }


def _resolve_manifest_path(raw_path, dataset_path_dir):
    candidates = [raw_path]
    if not os.path.isabs(raw_path):
        candidates.append(os.path.join(dataset_path_dir, raw_path))
        candidates.append(os.path.join(REPO_ROOT, raw_path))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def verify_manifest(dataset_path_dir):
    """Compare stored manifest shas to current file contents. Returns
    {"ok": bool, "mismatched": [paths]}."""
    manifest_path = os.path.join(dataset_path_dir, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        return {"ok": False, "mismatched": [f"manifest not found: {manifest_path}"]}
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError) as exc:
        return {"ok": False, "mismatched": [f"manifest unreadable: {manifest_path}: {exc}"]}

    files = manifest.get("files")
    if not isinstance(files, dict):
        return {"ok": False, "mismatched": ["manifest has no 'files' mapping"]}

    mismatched = []
    for raw_path, expected_sha in files.items():
        resolved = _resolve_manifest_path(str(raw_path), dataset_path_dir)
        if resolved is None:
            mismatched.append(str(raw_path))
            continue
        try:
            if sha256_file(resolved) != expected_sha:
                mismatched.append(str(raw_path))
        except OSError:
            mismatched.append(str(raw_path))
    return {"ok": not mismatched, "mismatched": sorted(mismatched)}
