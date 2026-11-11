"""Stage-B claim-support metrics for the eval harness.

Pure functions: no I/O, no LLM calls, no network. The judge LLM lives in
the runner (eval/runners/claim_support.py); this module only normalizes
raw judge verdicts, scores one item's claim list, aggregates per-item
scores across the dataset, scores the judge's own calibration set, and
aggregates the judge-assisted abstention-behavior verdicts. Correlations
use the standard library statistics module only.
"""
import statistics
from typing import Optional

VERDICTS = ("supported", "unsupported", "not_in_evidence")

JUDGE_FAILED = "judge_failed"

_VERDICT_KEYS = VERDICTS + (JUDGE_FAILED,)

_SYNONYMS = {
    "supported": "supported",
    "entailed": "supported",
    "support": "supported",
    "fully supported": "supported",
    "supported by evidence": "supported",
    "unsupported": "unsupported",
    "contradicted": "unsupported",
    "contradictory": "unsupported",
    "not supported": "unsupported",
    "unsupported by evidence": "unsupported",
    "not in evidence": "not_in_evidence",
    "unverifiable": "not_in_evidence",
    "no evidence": "not_in_evidence",
    "not found in evidence": "not_in_evidence",
    "not present in evidence": "not_in_evidence",
    "insufficient evidence": "not_in_evidence",
}


def _verdict_key(text: str) -> str:
    text = text.strip().lower()
    text = text.rstrip(" .:;!?")
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def normalize_verdict(raw: str) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    return _SYNONYMS.get(_verdict_key(raw))


def score_item(claims: list) -> dict:
    claims = claims or []
    counts = {verdict: 0 for verdict in VERDICTS}
    n_unparseable = 0
    for claim in claims:
        raw = claim.get("verdict") if isinstance(claim, dict) else None
        verdict = normalize_verdict(raw)
        if verdict is None:
            n_unparseable += 1
        else:
            counts[verdict] += 1
    n_parseable = sum(counts.values())
    support_rate = (counts["supported"] / n_parseable) if n_parseable else None
    if n_parseable == 0:
        item_verdict = JUDGE_FAILED
    elif counts["supported"] == n_parseable:
        item_verdict = "supported"
    elif counts["unsupported"]:
        item_verdict = "unsupported"
    else:
        item_verdict = "not_in_evidence"
    return {
        "n_claims": len(claims),
        "n_parseable": n_parseable,
        "n_supported": counts["supported"],
        "n_unsupported": counts["unsupported"],
        "n_not_in_evidence": counts["not_in_evidence"],
        "n_unparseable": n_unparseable,
        "support_rate": support_rate,
        "item_verdict": item_verdict,
    }


def _mean(values: list):
    return (sum(values) / len(values)) if values else None


def _correlations(xs: list, ys: list):
    if len(xs) < 2:
        return None, None
    try:
        pearson = statistics.correlation(xs, ys)
    except statistics.StatisticsError:
        pearson = None
    try:
        spearman = statistics.correlation(xs, ys, method="ranked")
    except statistics.StatisticsError:
        spearman = None
    return pearson, spearman


def aggregate(item_scores: list, stage_a_by_id: Optional[dict] = None) -> dict:
    scores = item_scores or []
    verdict_counts = {key: 0 for key in _VERDICT_KEYS}
    kind_buckets = {}
    rates = []
    total_claims = 0
    total_unparseable = 0
    for score in scores:
        item_verdict = score.get("item_verdict")
        if item_verdict in verdict_counts:
            verdict_counts[item_verdict] += 1
        rate = score.get("support_rate")
        if rate is not None:
            rates.append(rate)
        total_claims += score.get("n_claims", 0)
        total_unparseable += score.get("n_unparseable", 0)
        kind = score.get("kind") or "unknown"
        bucket = kind_buckets.setdefault(kind, {"n_items": 0, "rates": []})
        bucket["n_items"] += 1
        if rate is not None:
            bucket["rates"].append(rate)
    result = {
        "n_items": len(scores),
        "n_judge_failed": verdict_counts[JUDGE_FAILED],
        "mean_claim_support_rate": _mean(rates),
        "items_fully_supported": verdict_counts["supported"],
        "verdict_counts": verdict_counts,
        "n_unparseable_claims": total_unparseable,
        "mean_claims_per_item": (total_claims / len(scores)) if scores else None,
        "by_kind": {
            kind: {
                "n_items": bucket["n_items"],
                "mean_claim_support_rate": _mean(bucket["rates"]),
            }
            for kind, bucket in kind_buckets.items()
        },
    }
    if stage_a_by_id is not None:
        score_ids = {score.get("id") for score in scores}
        common_ids = score_ids & set(stage_a_by_id)
        if len(common_ids) >= 3:
            xs = []
            ys = []
            for score in scores:
                score_id = score.get("id")
                rate = score.get("support_rate")
                stage_a = stage_a_by_id.get(score_id)
                if score_id not in common_ids or rate is None:
                    continue
                if isinstance(stage_a, bool) or not isinstance(stage_a, (int, float)):
                    continue
                if not isinstance(rate, (int, float)) or isinstance(rate, bool):
                    continue
                xs.append(float(stage_a))
                ys.append(float(rate))
            pearson, spearman = _correlations(xs, ys)
            result["stage_a_correlation"] = {
                "n_pairs": len(xs),
                "pearson_r": pearson,
                "spearman_rho": spearman,
            }
    return result


def calibration_agreement(pairs: list) -> dict:
    pairs = pairs or []
    per_verdict = {verdict: {"n": 0, "correct": 0} for verdict in VERDICTS}
    correct = 0
    for pair in pairs:
        expected = normalize_verdict(pair.get("expected") if isinstance(pair, dict) else None)
        got = normalize_verdict(pair.get("got") if isinstance(pair, dict) else None)
        is_correct = expected is not None and got == expected
        if expected is not None:
            per_verdict[expected]["n"] += 1
            if is_correct:
                per_verdict[expected]["correct"] += 1
        if is_correct:
            correct += 1
    return {
        "n": len(pairs),
        "accuracy": (correct / len(pairs)) if pairs else None,
        "per_verdict": per_verdict,
    }


ABSTENTION_VERDICTS = ("appropriate_abstention", "fabricated_answer", "attempted_answer")

_ABSTENTION_SYNONYMS = {
    "appropriate abstention": "appropriate_abstention",
    "abstained": "appropriate_abstention",
    "correct abstention": "appropriate_abstention",
    "fabricated answer": "fabricated_answer",
    "hallucinated": "fabricated_answer",
    "fabricated": "fabricated_answer",
    "made up": "fabricated_answer",
    "attempted answer": "attempted_answer",
    "answered": "attempted_answer",
}


def normalize_abstention_verdict(raw: str) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    return _ABSTENTION_SYNONYMS.get(_verdict_key(raw))


def abstention_aggregate(entries: list) -> dict:
    entries = entries or []
    counts = {verdict: 0 for verdict in ABSTENTION_VERDICTS}
    counts["unparseable"] = 0
    agreements = []
    for entry in entries:
        entry = entry if isinstance(entry, dict) else {}
        verdict = normalize_abstention_verdict(entry.get("abstention_verdict"))
        if verdict is None:
            counts["unparseable"] += 1
        else:
            counts[verdict] += 1
        marker = entry.get("stage_a_marker_abstained")
        if verdict is not None and isinstance(marker, bool):
            agreements.append(
                1.0 if (verdict == "appropriate_abstention") == marker else 0.0
            )
    n_traps = len(entries)
    return {
        "n_traps": n_traps,
        "verdict_counts": counts,
        "appropriate_rate": (
            counts["appropriate_abstention"] / n_traps if n_traps else None
        ),
        "stage_a_agreement_rate": _mean(agreements),
    }
