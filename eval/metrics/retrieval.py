"""Binary-relevance retrieval metrics for the golden QA harness.

Pure functions over plain data: a ranking is a list of hit dicts as
produced by core.rag.retrieve_for_query ({"id", "text", "metadata",
"distance", "rank"}; a bare string is also tolerated) and relevance is
lexical anchor containment after casefold + whitespace collapse. No I/O,
no core imports, fully deterministic.
"""
import math
import re

_WS_RE = re.compile(r"\s+")


def _norm(text) -> str:
    return _WS_RE.sub(" ", str(text or "").casefold()).strip()


def _hit_text(hit) -> str:
    if isinstance(hit, dict):
        return hit.get("text") or ""
    return str(hit or "")


def is_relevant(hit_text, anchors) -> bool:
    anchors = anchors or []
    text = _norm(hit_text)
    for anchor in anchors:
        needle = _norm(anchor)
        if needle and needle in text:
            return True
    return False


def _relevant_flags(rankings, anchors):
    return [is_relevant(_hit_text(hit), anchors) for hit in rankings]


def recall_at_k(rankings, anchors, k) -> float:
    anchors = anchors or []
    if not anchors:
        return 0.0
    top_norms = [_norm(_hit_text(hit)) for hit in list(rankings)[:k]]
    covered = 0
    for anchor in anchors:
        needle = _norm(anchor)
        if needle and any(needle in text for text in top_norms):
            covered += 1
    return covered / len(anchors)


def precision_at_k(rankings, anchors, k) -> float:
    rankings = list(rankings)
    denominator = min(k, len(rankings))
    if denominator <= 0:
        return 0.0
    flags = _relevant_flags(rankings[:k], anchors)
    return sum(1 for rel in flags if rel) / denominator


def mrr(rankings, anchors) -> float:
    flags = _relevant_flags(rankings, anchors)
    for rank, rel in enumerate(flags, start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(rankings, anchors, k) -> float:
    anchors = anchors or []
    if not anchors:
        return 0.0
    flags = _relevant_flags(list(rankings)[:k], anchors)
    dcg = 0.0
    for idx, rel in enumerate(flags):
        if rel:
            dcg += 1.0 / math.log2(idx + 2)
    ideal_hits = min(len(anchors), k)
    idcg = 0.0
    for idx in range(ideal_hits):
        idcg += 1.0 / math.log2(idx + 2)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def context_sufficiency(rankings, anchors, k) -> bool:
    anchors = anchors or []
    top = list(rankings)[:k]
    for anchor in anchors:
        needle = _norm(anchor)
        if not needle:
            continue
        if not any(needle in _norm(_hit_text(hit)) for hit in top):
            return False
    return True


def evaluate_item(item, hits, ks=(3, 5, 10)) -> dict:
    anchors = list(item.get("evidence_anchors") or [])
    rankings = list(hits or [])
    result = {
        "id": item.get("id"),
        "n_hits": len(rankings),
        "mrr": mrr(rankings, anchors),
    }
    first_relevant_rank = None
    if anchors:
        flags = _relevant_flags(rankings, anchors)
        for rank, rel in enumerate(flags, start=1):
            if rel:
                first_relevant_rank = rank
                break
    result["first_relevant_rank"] = first_relevant_rank
    for k in ks:
        result[f"recall@{k}"] = recall_at_k(rankings, anchors, k)
        result[f"precision@{k}"] = precision_at_k(rankings, anchors, k)
        result[f"ndcg@{k}"] = ndcg_at_k(rankings, anchors, k)
        result[f"sufficiency@{k}"] = context_sufficiency(rankings, anchors, k)
    return result


def aggregate(per_item_results, ks) -> dict:
    items = [r for r in (per_item_results or []) if "error" not in r]

    def _mean(values):
        return round(sum(values) / len(values), 4) if values else 0.0

    out = {"n_items": len(items)}
    for k in ks:
        out[f"recall@{k}"] = _mean(
            [item[f"recall@{k}"] for item in items if f"recall@{k}" in item]
        )
        out[f"precision@{k}"] = _mean(
            [item[f"precision@{k}"] for item in items if f"precision@{k}" in item]
        )
        out[f"ndcg@{k}"] = _mean(
            [item[f"ndcg@{k}"] for item in items if f"ndcg@{k}" in item]
        )
        out[f"sufficiency@{k}"] = _mean(
            [
                1.0 if item[f"sufficiency@{k}"] else 0.0
                for item in items
                if f"sufficiency@{k}" in item
            ]
        )
    out["mrr"] = _mean([item["mrr"] for item in items if "mrr" in item])
    ranks = [
        item["first_relevant_rank"]
        for item in items
        if item.get("first_relevant_rank") is not None
    ]
    out["first_relevant_rank"] = (
        round(sum(ranks) / len(ranks), 4) if ranks else None
    )
    return out
