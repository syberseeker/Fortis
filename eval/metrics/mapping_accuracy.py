"""Framework/control mapping-accuracy metrics for the Stage-B claim-support runner.

Scores which framework+control mappings a stored reply asserts, at two levels:
the exact (framework, control_id) pair, and the bare control id (robust to
replies that cite an id without naming its framework). "Fabricated" pairs are
mappings the seeded corpus cannot back, and the demotion confusion matrix checks
the production guardrail (core.report_service.validate_findings_mappings) against
corpus truth.

core.framework_aliases is imported lazily inside the helpers and the report
service / schema only inside the default validator, so importing this module
costs nothing and never pulls chromadb or the vector store.

Framework recognition table below is keyed on the names a reply may use; each
name is canonicalized through core.framework_aliases.canonical_framework, then
through _CORPUS_FRAMEWORK_ALIASES, which folds the file-style spellings
(SOC2_TSC, HIPAA_SECURITY_RULE, NIST_800_53) onto the seeded corpus ids
(SOC_2, HIPAA, NIST_800-53) so emitted pairs are comparable to corpus pairs.
"""
import re

from .citations import _mask_tokens, _mean_non_none, _prf_from_sets, _round4, split_sentences
from .grounding import extract_citation_tokens

_FRAMEWORK_PATTERNS = (
    (r"NIST\s+CYBERSECURITY\s+FRAMEWORK", "NIST_CSF"),
    (r"NIST\s+CSF", "NIST_CSF"),
    (r"(?<![A-Z])CSF(?![A-Z])", "NIST_CSF"),
    (r"OWASP(?:\s+TOP\s*10)?", "OWASP_TOP10"),
    (r"CIS\s+CONTROLS", "CIS_CONTROLS"),
    (r"(?<![A-Z])CIS(?![A-Z])", "CIS_CONTROLS"),
    (r"ISO\s*/\s*IEC\s*27001", "ISO_27001"),
    (r"ISO\s*27001", "ISO_27001"),
    (r"MITRE\s+ATT\s*&\s*CK(?:\s*\(\s*ENTERPRISE\s*\))?", "MITRE_ATTACK"),
    (r"ATT\s*&\s*CK", "MITRE_ATTACK"),
    (r"PCI[\s\-]*DSS", "PCI_DSS"),
    (r"SOC\s*2(?:\s*TSC)?", "SOC2_TSC"),
    (r"(?<![A-Z])GDPR(?![A-Z])", "GDPR"),
    (r"HIPAA(?:\s+SECURITY\s+RULE)?", "HIPAA_SECURITY_RULE"),
    (r"NIST\s*SP\s*800[\s\-]?53", "NIST_800_53"),
    (r"CWE\s+TOP\s*25", "CWE_TOP25"),
    (r"(?<![A-Z])CWE(?![A-Z])", "CWE_TOP25"),
)

_CORPUS_FRAMEWORK_ALIASES = {
    "SOC2_TSC": "SOC_2",
    "SOC_2_TSC": "SOC_2",
    "TSC": "SOC_2",
    "HIPAA_SECURITY_RULE": "HIPAA",
    "NIST_800_53": "NIST_800-53",
}

_UNVERIFIED_SUFFIX = " (unverified)"
MAX_FABRICATED_PAIRS = 20


def _canonical_framework(name) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    try:
        from core.framework_aliases import canonical_framework

        canon = canonical_framework(text) or text
    except ImportError:
        canon = text
    return _CORPUS_FRAMEWORK_ALIASES.get(canon.strip().upper(), canon).strip()


def _frameworks_in(masked_sentence) -> list:
    names = []
    for pattern, spelling in _FRAMEWORK_PATTERNS:
        if re.search(pattern, masked_sentence, re.IGNORECASE):
            canon = _canonical_framework(spelling)
            if canon and canon not in names:
                names.append(canon)
    return names


def _normalize_pair(entry):
    if isinstance(entry, dict):
        framework, control_id = entry.get("framework"), entry.get("control_id")
    elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
        framework, control_id = entry[0], entry[1]
    else:
        framework, control_id = "", entry
    return _canonical_framework(framework), str(control_id or "").strip().upper()


def _known_pair_set(known_pairs) -> set:
    out = set()
    for entry in (known_pairs or []):
        framework, control_id = _normalize_pair(entry)
        if framework and control_id:
            out.add((framework.upper(), control_id.upper()))
    return out


def _dedup_pairs(pairs) -> list:
    out = []
    seen = set()
    for entry in (pairs or []):
        framework, control_id = _normalize_pair(entry)
        if not framework and not control_id:
            continue
        key = (framework.upper(), control_id.upper())
        if key in seen:
            continue
        seen.add(key)
        out.append({"framework": framework, "control_id": control_id})
    return out


def extract_framework_pairs(reply) -> list:
    """(framework, control_id) pairs a reply asserts: an id-shaped token inherits
    every framework named in its own sentence, and {"framework": ""} when the
    sentence names none. Deduped, first-seen order."""
    pairs = []
    seen = set()
    for sentence in split_sentences(reply):
        masked, _mapping = _mask_tokens(sentence)
        frameworks = _frameworks_in(masked) or [""]
        for token in extract_citation_tokens(sentence):
            token = str(token or "").strip()
            if not token:
                continue
            for framework in frameworks:
                key = (framework, token.upper())
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({"framework": framework, "control_id": token})
    return pairs


def mapping_prf(emitted_pairs, gold_pairs) -> dict:
    """{"pair": ..., "id": ...}, each with the citation P/R/F1 keys. The pair
    level scores exact (framework, control_id) matches and ignores frameworkless
    emissions; the id level ignores the framework entirely."""
    emitted = [_normalize_pair(p) for p in (emitted_pairs or [])]
    gold = [_normalize_pair(p) for p in (gold_pairs or [])]
    emitted_pairs_set = {(f, c) for f, c in emitted if f and c}
    gold_pairs_set = {(f, c) for f, c in gold if f and c}
    emitted_ids = {c for _f, c in emitted if c}
    gold_ids = {c for _f, c in gold if c}
    return {
        "pair": _prf_from_sets(emitted_pairs_set, gold_pairs_set),
        "id": _prf_from_sets(emitted_ids, gold_ids),
    }


def fabricated_pairs(emitted_pairs, known_pairs) -> list:
    """Framework-named mappings absent from the corpus, sorted, capped at 20."""
    known = _known_pair_set(known_pairs)
    rows = []
    seen = set()
    for entry in (emitted_pairs or []):
        framework, control_id = _normalize_pair(entry)
        if not framework or not control_id:
            continue
        key = (framework.upper(), control_id.upper())
        if key in known or key in seen:
            continue
        seen.add(key)
        rows.append({"framework": framework, "control_id": control_id})
    return sorted(rows, key=lambda p: (p["framework"], p["control_id"]))[:MAX_FABRICATED_PAIRS]


def _kept_by_validator(pairs, validate_report_fn) -> list:
    if validate_report_fn is not None:
        return list(validate_report_fn([dict(p) for p in pairs]) or [])
    return _default_kept_pairs(pairs)


def _default_kept_pairs(pairs) -> list:
    from core import report_service
    from core.reports.schema import SecurityReport

    findings = [
        {
            "title": "eval",
            "severity": "Low",
            "description": "eval",
            "evidence": "eval",
            "remediation": "eval",
            "framework": p["framework"],
            "control_id": p["control_id"],
        }
        for p in pairs
    ]
    report = SecurityReport.model_validate({
        "title": "eval",
        "client_context": "eval",
        "scope": "eval",
        "executive_summary": "eval",
        "findings": findings,
        "overall_risk_rating": "Low",
        "recommendations_summary": [],
        "diagram": None,
    })
    report_service.validate_findings_mappings(report)
    kept = []
    for finding, pair in zip(report.findings, pairs):
        framework = str(finding.framework or "")
        control_id = str(finding.control_id or "")
        if framework.endswith(_UNVERIFIED_SUFFIX) or control_id.endswith(_UNVERIFIED_SUFFIX):
            continue
        kept.append((pair["framework"], pair["control_id"]))
    return kept


def _valid_keys(kept) -> set:
    keys = set()
    for entry in (kept or []):
        if isinstance(entry, dict):
            framework, control_id = entry.get("framework"), entry.get("control_id")
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            framework, control_id = entry[0], entry[1]
        else:
            framework, control_id = "", entry
        keys.add((str(framework or ""), str(control_id or "").strip()))
        keys.add((str(framework or "").strip().upper(), str(control_id or "").strip().upper()))
    return keys


def demotion_confusion(emitted_pairs, known_pairs, validate_report_fn=None) -> dict:
    """Confusion matrix of the mapping guardrail against corpus truth. A pair is
    predicted-valid when the validator leaves it unmarked; truth is membership in
    known_pairs. TP/FP count kept mappings, TN/FN count demoted ones."""
    pairs = _dedup_pairs(emitted_pairs)
    known = _known_pair_set(known_pairs)
    valid = _valid_keys(_kept_by_validator(pairs, validate_report_fn))
    tp = fp = tn = fn = 0
    for pair in pairs:
        key = (pair["framework"].upper(), pair["control_id"].upper())
        predicted_valid = key in valid or (pair["framework"], pair["control_id"]) in valid
        in_corpus = key in known
        if predicted_valid and in_corpus:
            tp += 1
        elif predicted_valid and not in_corpus:
            fp += 1
        elif not predicted_valid and not in_corpus:
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": total,
        "accuracy": _round4((tp + tn) / total) if total else None,
        "demotion_precision": _round4(tp / (tp + fp)) if (tp + fp) else None,
        "demotion_recall": _round4(tp / (tp + fn)) if (tp + fn) else None,
    }


def _item_prf(item, level) -> dict:
    for key in ("mapping_prf", "mapping"):
        nested = item.get(key)
        if isinstance(nested, dict) and isinstance(nested.get(level), dict):
            return nested[level]
    for key in (level, f"{level}_prf"):
        direct = item.get(key)
        if isinstance(direct, dict) and "precision" in direct:
            return direct
    return {}


def _item_fabricated_count(item) -> int:
    for key in ("n_fabricated_pairs", "n_fabricated"):
        value = item.get(key)
        if value is not None:
            return int(value or 0)
    for key in ("fabricated_pairs", "fabricated_mappings"):
        value = item.get(key)
        if isinstance(value, (list, tuple, set)):
            return len(value)
    return 0


def _item_demotion(item) -> dict:
    for key in ("demotion_confusion", "confusion", "demotion"):
        value = item.get(key)
        if isinstance(value, dict) and "tp" in value:
            return value
    if "tp" in item:
        return item
    return {}


def _item_emitted_ids(item) -> int:
    for key in ("n_emitted_ids", "n_mapping_ids", "n_pairs", "n_emitted_pairs"):
        value = item.get(key)
        if value is not None:
            return int(value or 0)
    for key in ("emitted_pairs", "mappings"):
        value = item.get(key)
        if isinstance(value, (list, tuple)):
            return len(value)
    return int(_item_prf(item, "id").get("n_emitted") or 0)


def aggregate_items(per_item_summaries, known_pairs) -> dict:
    """Means of the two P/R/F1 levels across items (None-skipped), fabricated
    totals, and the summed demotion confusion matrix with recomputed rates.

    Per-item rows may carry "mapping_prf"/"mapping" (or flat "pair"/"id"),
    "fabricated_pairs"/"n_fabricated_pairs", and "demotion_confusion"."""
    items = [s for s in (per_item_summaries or []) if isinstance(s, dict)]
    sums = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    n_demotion = 0
    for item in items:
        confusion = _item_demotion(item)
        if confusion:
            n_demotion += 1
            for key in sums:
                sums[key] += int(confusion.get(key) or 0)
    tp, fp, tn, fn = sums["tp"], sums["fp"], sums["tn"], sums["fn"]
    total = tp + fp + tn + fn
    return {
        "n_items": len(items),
        "n_items_demotion_scored": n_demotion,
        "mean_pair_precision": _mean_non_none(_item_prf(s, "pair").get("precision") for s in items),
        "mean_pair_recall": _mean_non_none(_item_prf(s, "pair").get("recall") for s in items),
        "mean_pair_f1": _mean_non_none(_item_prf(s, "pair").get("f1") for s in items),
        "mean_id_precision": _mean_non_none(_item_prf(s, "id").get("precision") for s in items),
        "mean_id_recall": _mean_non_none(_item_prf(s, "id").get("recall") for s in items),
        "mean_id_f1": _mean_non_none(_item_prf(s, "id").get("f1") for s in items),
        "mean_mapping_ids_per_item": _mean_non_none(_item_emitted_ids(s) for s in items),
        "total_fabricated_pairs": sum(_item_fabricated_count(s) for s in items),
        "demotion_confusion": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "n": total,
            "accuracy": _round4((tp + tn) / total) if total else None,
            "demotion_precision": _round4(tp / (tp + fp)) if (tp + fp) else None,
            "demotion_recall": _round4(tp / (tp + fn)) if (tp + fn) else None,
        },
        "n_known_pairs": len(_known_pair_set(known_pairs)),
    }
