"""Deterministic Stage-A grounding scorers for the golden QA dataset.

No model, no network: these functions score an LLM reply against the item's
gold annotations and the retrieved evidence. They are proxies by design
(lexical, not semantic) — see the runner's LIMITATIONS section.

- Citation checking: control-id-shaped tokens in the reply are matched
  against the item's gold control ids and the seeded corpus vocabulary.
- Abstention: marker-based decline/hedge detection, applied in both
  directions (traps should abstain; answerable items should not).
- Lexical grounding: per-sentence content-word overlap against the
  retrieved evidence texts, plus verbatim evidence-anchor hits.
"""
import re

_CITATION_TOKEN_RE = re.compile(
    r"\b(?:"
    r"[A-Z]{1,6}[.\-][A-Z]{1,6}(?:-\d+)?(?:\(\d+\))?"  # PR.PS, ID.AM-1, IA-2(1)
    r"|[A-Z]{1,6}-\d[\w./-]*(?:\(\d+\))?"              # CWE-79, IA-2(1), CVE-2024-1234
    r"|[A-Z]{1,6}\d{2,4}(?:[:.]\d+)+"                  # A01:2021
    r"|[A-Z]{1,6}\d{2,4}\b"                            # ISO27001-style run-ons
    r")"
)

_ABSTENTION_MARKERS = (
    "cannot", "can't", "do not have", "don't have", "no information",
    "not present in", "unable to", "no evidence", "does not appear",
    "please upload", "not able to", "no record of", "cannot verify",
)

_SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]+")
_WORD_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was",
    "were", "has", "have", "had", "not", "but", "all", "any", "can",
    "will", "its", "our", "their", "there", "than", "then", "into",
    "over", "under", "also", "may", "should", "would", "could", "been",
    "being", "when", "what", "which", "who", "how", "why", "where",
    "you", "your", "does", "did", "out", "per", "via", "use", "using",
    "used", "more", "most", "such", "each", "both", "between", "these",
    "those", "them", "they",
}

_GROUNDING_THRESHOLD = 0.5


def extract_citation_tokens(text: str) -> list:
    return [t.rstrip(".") for t in _CITATION_TOKEN_RE.findall(text or "")]


def _gold_control_id_set(gold_control_ids) -> set:
    """Gold entries are either plain strings or {"framework", "control_id"}
    dicts (the golden dataset's shape); normalize both to bare upper ids."""
    ids = set()
    for gold in (gold_control_ids or []):
        if isinstance(gold, dict):
            control_id = str(gold.get("control_id") or "").upper()
            if control_id:
                ids.add(control_id)
        elif gold:
            ids.add(str(gold).upper())
    return ids


def citation_metrics(reply_text: str, gold_control_ids, corpus_control_ids) -> dict:
    tokens = sorted(set(extract_citation_tokens(reply_text)))
    upper = {t.upper() for t in tokens}
    gold = _gold_control_id_set(gold_control_ids)
    corpus = {str(c).upper() for c in (corpus_control_ids or []) if c}
    hits = sorted(g for g in gold if g in upper)
    recall = (len(hits) / len(gold)) if gold else None
    unverifiable = [t for t in tokens if t.upper() not in corpus and t.upper() not in gold]
    return {
        "cited_tokens": tokens,
        "gold_citation_hits": hits,
        "gold_citation_recall": recall,
        "unverifiable_citation_tokens": unverifiable,
    }


def abstention_detected(reply_text: str) -> bool:
    lowered = (reply_text or "").lower()
    return any(marker in lowered for marker in _ABSTENTION_MARKERS)


def _content_words(text: str) -> set:
    tokens = _WORD_RE.findall((text or "").lower())
    return {t for t in tokens if len(t) >= 3 and t not in _STOPWORDS}


def _norm_ws(text: str) -> str:
    return " ".join((text or "").split()).casefold()


def grounding_metrics(reply_text: str, evidence_texts, anchors) -> dict:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(reply_text or "") if s.strip()]
    evidence_sets = [words for words in (_content_words(t) for t in (evidence_texts or [])) if words]
    grounded = 0
    for sentence in sentences:
        sent_words = _content_words(sentence)
        if not sent_words:
            continue
        for ev_words in evidence_sets:
            overlap = len(sent_words & ev_words) / min(len(sent_words), len(ev_words))
            if overlap >= _GROUNDING_THRESHOLD:
                grounded += 1
                break
    normalized_reply = _norm_ws(reply_text)
    found = [a for a in (anchors or []) if a and _norm_ws(a) in normalized_reply]
    return {
        "sentences": len(sentences),
        "grounded_sentences": grounded,
        "sentence_grounded_fraction": (grounded / len(sentences)) if sentences else 0.0,
        "anchors_in_reply": found,
        "anchor_hit": bool(found),
    }


def score_item(item: dict, reply_text: str, evidence_texts, corpus_control_ids=None) -> dict:
    anchors = item.get("evidence_anchors") or []
    score = grounding_metrics(reply_text, evidence_texts, anchors)
    if item.get("abstention") is True or item.get("kind") == "abstention":
        score["abstained"] = abstention_detected(reply_text)
        return score
    citations = citation_metrics(
        reply_text, item.get("gold_control_ids") or [], corpus_control_ids
    )
    score.update(citations)
    score["false_abstention"] = abstention_detected(reply_text)
    return score


def aggregate(item_scores) -> dict:
    if not item_scores:
        return {
            "n_items": 0,
            "abstention_rate_on_traps": None,
            "false_abstention_rate_on_answerable": None,
            "mean_gold_citation_recall": None,
            "mean_sentence_grounded_fraction": None,
            "anchor_hit_rate": None,
            "items_with_unverifiable_citations": 0,
        }
    traps = [s for s in item_scores if "abstained" in s]
    answerable = [s for s in item_scores if "false_abstention" in s]
    recalls = [
        s["gold_citation_recall"]
        for s in item_scores
        if s.get("gold_citation_recall") is not None
    ]
    fractions = [s.get("sentence_grounded_fraction", 0.0) for s in item_scores]
    anchor_hits = sum(1 for s in item_scores if s.get("anchor_hit"))
    unverifiable = sum(1 for s in item_scores if s.get("unverifiable_citation_tokens"))
    return {
        "n_items": len(item_scores),
        "abstention_rate_on_traps": (
            sum(1 for s in traps if s["abstained"]) / len(traps)
        ) if traps else None,
        "false_abstention_rate_on_answerable": (
            sum(1 for s in answerable if s["false_abstention"]) / len(answerable)
        ) if answerable else None,
        "mean_gold_citation_recall": (
            sum(recalls) / len(recalls)
        ) if recalls else None,
        "mean_sentence_grounded_fraction": (
            sum(fractions) / len(fractions)
        ) if fractions else None,
        "anchor_hit_rate": anchor_hits / len(item_scores),
        "items_with_unverifiable_citations": unverifiable,
    }
