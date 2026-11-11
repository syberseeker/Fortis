"""Deterministic citation-attribution metrics for the Stage-B claim-support runner.

Pure functions over plain data: a citation is a {"token", "sentence"} pair pulled
out of a stored reply, a framework hit is the retrieval shape {"id", "text",
"metadata": {"framework", "control_id"}, "distance", "rank"}, and a corpus
vocabulary is any iterable of control-id strings. No model, no I/O, no core
imports; token extraction is delegated to eval.metrics.grounding so every module
in the harness recognises the same id-shaped tokens.

Sentence boundaries are runs of [ . ! ? newline ], with id-shaped tokens masked
out first so "PR.PS" is never torn into "PR" / "PS" fragments. All floats are
rounded to 4 decimals and every mean is None-safe.
"""
import re
from collections import Counter

from .grounding import extract_citation_tokens

_WS_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]+")
_WORD_RE = re.compile(r"[a-z0-9]+")

DEFAULT_OVERLAP_THRESHOLD = 0.35
MAX_FABRICATED_TOKENS_PER_ITEM = 10
MAX_FABRICATED_TOKEN_NAMES = 20

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "has", "have", "had", "not", "but", "all", "any", "can", "will", "its",
    "our", "their", "there", "than", "then", "into", "over", "under", "also",
    "may", "should", "would", "could", "been", "being", "when", "what",
    "which", "who", "how", "why", "where", "you", "your", "does", "did",
    "out", "per", "via", "use", "using", "used", "more", "most", "such",
    "each", "both", "between", "these", "those", "them", "they", "must",
    "need", "needs", "shall", "while", "because", "about", "after", "before",
    "during", "given", "across", "within", "without",
}


def _mean_non_none(values):
    usable = [float(v) for v in values if v is not None]
    return _round4(sum(usable) / len(usable)) if usable else None


def _norm(text) -> str:
    return _WS_RE.sub(" ", str(text or "").casefold()).strip()


def _norm_space(text) -> str:
    return " ".join(str(text or "").split())


def _round4(value):
    return None if value is None else round(float(value), 4)


def _content_words(text) -> set:
    tokens = _WORD_RE.findall(str(text or "").lower())
    return {t for t in tokens if len(t) >= 3 and t not in _STOPWORDS}


def _id_set(values) -> set:
    """Uppercased, stripped id set. Accepts plain id strings as well as
    gold_control_ids-style {"framework", "control_id"} dicts."""
    out = set()
    for value in (values or []):
        if isinstance(value, dict):
            value = value.get("control_id")
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            value = value[1]
        text = str(value or "").strip().upper()
        if text:
            out.add(text)
    return out


def _mask_tokens(text):
    """Replace every id-shaped token with a dot-free placeholder so sentence
    splitting cannot break it apart. Returns (masked_text, placeholder->token)."""
    raw = str(text or "")
    unique = sorted({t for t in extract_citation_tokens(raw) if t}, key=lambda t: (-len(t), t))
    mapping = {}
    masked = raw
    for index, token in enumerate(unique):
        placeholder = "\ue000%d\ue0ff" % index
        mapping[placeholder] = token
        masked = masked.replace(token, placeholder)
    return masked, mapping


def _unmask(text, mapping) -> str:
    for placeholder, token in mapping.items():
        text = text.replace(placeholder, token)
    return text


def split_sentences(text) -> list:
    """Split a reply into sentences on [ . ! ? newline ] runs, keeping
    id-shaped tokens intact, whitespace-normalized, empty fragments dropped."""
    masked, mapping = _mask_tokens(text)
    sentences = []
    for fragment in _SENTENCE_SPLIT_RE.split(masked):
        sentence = _norm_space(_unmask(fragment, mapping))
        if sentence:
            sentences.append(sentence)
    return sentences


def _prf_from_sets(emitted: set, gold: set) -> dict:
    matched = len(emitted & gold)
    precision = (matched / len(emitted)) if emitted else None
    recall = (matched / len(gold)) if gold else None
    if precision is None and recall is None:
        f1 = None
    else:
        left = precision or 0.0
        right = recall or 0.0
        f1 = 0.0 if (left + right) <= 0 else (2 * left * right) / (left + right)
    return {
        "precision": _round4(precision),
        "recall": _round4(recall),
        "f1": _round4(f1),
        "n_emitted": len(emitted),
        "n_gold": len(gold),
        "n_matched": matched,
    }


def _prf_from_lists(emitted_ids, gold_ids) -> dict:
    return _prf_from_sets(_id_set(emitted_ids), _id_set(gold_ids))


def extract_citations(reply) -> list:
    """Citations in reply order as [{"token", "sentence"}]; identical
    (token, sentence) pairs are deduped and empty tokens dropped."""
    citations = []
    seen = set()
    for sentence in split_sentences(reply):
        for token in extract_citation_tokens(sentence):
            token = str(token or "").strip()
            if not token:
                continue
            key = (token, sentence)
            if key in seen:
                continue
            seen.add(key)
            citations.append({"token": token, "sentence": sentence})
    return citations


def citation_precision_recall_f1(emitted_ids, gold_ids) -> dict:
    """Set-based precision/recall/F1 over uppercased control-id tokens."""
    return _prf_from_lists(emitted_ids, gold_ids)


def fabricated_tokens(emitted_ids, corpus_vocab) -> list:
    """Emitted ids the corpus vocabulary cannot back, uppercased, sorted, deduped."""
    vocab = _id_set(corpus_vocab)
    return sorted(t for t in _id_set(emitted_ids) if t not in vocab)


def sentence_overlap(sentence, hit_text) -> float:
    """Containment: fraction of the sentence's content words present in the hit.
    0.0 when the sentence carries no content words."""
    words = _content_words(sentence)
    if not words:
        return 0.0
    shared = words & _content_words(hit_text)
    return len(shared) / len(words)


def _hit_text(hit) -> str:
    return str(hit.get("text") or "") if isinstance(hit, dict) else str(hit or "")


def _hit_id(hit):
    return hit.get("id") if isinstance(hit, dict) else None


def _hit_control_id(hit) -> str:
    if not isinstance(hit, dict):
        return ""
    meta = hit.get("metadata")
    if not isinstance(meta, dict):
        return ""
    return str(meta.get("control_id") or "").strip().upper()


def attribute_citations(citations, framework_hits, overlap_threshold=DEFAULT_OVERLAP_THRESHOLD) -> list:
    """Resolve each citation against the retrieved framework hits by control id,
    then attribute it to the best-overlapping hit when the overlap clears the bar."""
    hits = list(framework_hits or [])
    attributed = []
    for citation in (citations or []):
        citation = citation if isinstance(citation, dict) else {}
        token = str(citation.get("token") or "").strip()
        sentence = str(citation.get("sentence") or "")
        want = token.upper()
        candidates = [h for h in hits if want and _hit_control_id(h) == want]
        if not candidates:
            attributed.append({
                "token": token,
                "sentence": sentence,
                "resolved": False,
                "match_hit_id": None,
                "overlap": 0.0,
                "attributed": False,
            })
            continue
        best_overlap = -1.0
        best_hit_id = None
        for hit in candidates:
            value = sentence_overlap(sentence, _hit_text(hit))
            if value > best_overlap:
                best_overlap = value
                best_hit_id = _hit_id(hit)
        attributed.append({
            "token": token,
            "sentence": sentence,
            "resolved": True,
            "match_hit_id": best_hit_id,
            "overlap": _round4(max(best_overlap, 0.0)),
            "attributed": best_overlap >= overlap_threshold,
        })
    return attributed


def citation_summary(citations_attributed, emitted_ids, gold_ids, corpus_vocab) -> dict:
    """Everything the runner needs for one item: attribution counts plus the
    gold-citation P/R/F1 and the corpus-unsupported tokens."""
    rows = [c for c in (citations_attributed or []) if isinstance(c, dict)]
    prf = citation_precision_recall_f1(emitted_ids, gold_ids)
    fabricated = fabricated_tokens(emitted_ids, corpus_vocab)
    n_resolved = sum(1 for c in rows if c.get("resolved"))
    n_attributed = sum(1 for c in rows if c.get("attributed"))
    return {
        "n_citations": len(rows),
        "n_resolved": n_resolved,
        "n_attributed": n_attributed,
        "attribution_rate": _round4(n_attributed / n_resolved) if n_resolved else None,
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "n_fabricated": len(fabricated),
        "fabricated_tokens": fabricated[:MAX_FABRICATED_TOKENS_PER_ITEM],
    }


def aggregate_items(per_item_summaries) -> dict:
    """Means over citation_summary rows; rate-style means skip None and report
    None when nothing usable is present."""
    items = [s for s in (per_item_summaries or []) if isinstance(s, dict)]
    counts = [int(s.get("n_citations") or 0) for s in items]
    rates = [s["attribution_rate"] for s in items if s.get("attribution_rate") is not None]
    counter = Counter()
    for summary in items:
        for token in (summary.get("fabricated_tokens") or []):
            counter[str(token)] += 1
    top = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_FABRICATED_TOKEN_NAMES]
    return {
        "n_items": len(items),
        "n_items_with_citations": sum(1 for c in counts if c > 0),
        "mean_citations_per_item": _round4(sum(counts) / len(counts)) if counts else 0.0,
        "mean_attribution_rate": _round4(sum(rates) / len(rates)) if rates else None,
        "mean_precision": _mean_non_none(s.get("precision") for s in items),
        "mean_recall": _mean_non_none(s.get("recall") for s in items),
        "mean_f1": _mean_non_none(s.get("f1") for s in items),
        "total_fabricated": sum(int(s.get("n_fabricated") or 0) for s in items),
        "fabricated_token_counts": dict(top),
    }


def _mean_non_none(values):
    usable = [float(v) for v in values if v is not None]
    return _round4(sum(usable) / len(usable)) if usable else None
