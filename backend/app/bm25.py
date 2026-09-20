"""
Lazy in-memory BM25 keyword index over the Chroma collections.

Dense embeddings are weak on exact identifiers (control IDs, CWE numbers,
config keys) that dominate security text. This module keeps a tokenized
BM25Okapi corpus per collection, marked dirty on any Chroma mutation and
rebuilt on the next search. Pure Python, no model, no network — cheap on
low-end hardware and fully meaningful even with the hash-stub embedding
backend used in offline tests.
"""
import re
import threading
from collections import OrderedDict

from rank_bm25 import BM25Okapi

_LOCK = threading.Lock()
_INDEXES = OrderedDict()
_MAX_SCOPES = 8  # per collection; evicts stale engagement scopes


def tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


def mark_dirty(collection: str) -> None:
    with _LOCK:
        for key, state in _INDEXES.items():
            if key[0] == collection:
                state["dirty"] = True


def _ensure_index(collection: str, scope: str, getter) -> dict:
    key = (collection, scope)
    with _LOCK:
        state = _INDEXES.get(key)
        if state is not None and not state["dirty"]:
            _INDEXES.move_to_end(key)
            return state
    # rebuild outside the lock so a slow chroma .get / BM25Okapi build never
    # stalls concurrent searches or mark_dirty callers
    raw = getter()
    ids, metadatas, documents = [], [], []
    for cid, meta, doc in zip(
        raw.get("ids", []),
        raw.get("metadatas", []),
        raw.get("documents", []),
    ):
        if doc is None:
            continue
        ids.append(cid)
        metadatas.append(meta)
        documents.append(doc)
    corpus = [tokenize(t) for t in documents]
    bm25 = None
    if corpus and any(corpus):
        bm25 = BM25Okapi(corpus)
    state = {
        "ids": ids,
        "metadatas": metadatas,
        "documents": documents,
        "bm25": bm25,
        "dirty": False,
    }
    with _LOCK:
        _INDEXES[key] = state
        _INDEXES.move_to_end(key)
        while len(_INDEXES) > _MAX_SCOPES:
            _INDEXES.popitem(last=False)
    return state


def search(collection: str, scope: str, getter, query: str, top_k: int, where_pred=None) -> list:
    """Scores the indexed corpus for query and returns hits shaped like
    vectorstore._flatten output: {"text", "metadata", "distance"}. The cache
    is keyed by (collection, scope) so indices never bleed across engagements;
    where_pred filters on metadata before truncation."""
    if top_k <= 0:
        return []
    state = _ensure_index(collection, scope, getter)
    if state["bm25"] is None:
        return []
    scores = state["bm25"].get_scores(tokenize(query))
    scored = [
        (scores[i], i)
        for i in range(len(state["documents"]))
        if scores[i] > 0 and (where_pred is None or where_pred(state["metadatas"][i]))
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {
            "text": state["documents"][i],
            "metadata": state["metadatas"][i],
            "distance": None,
        }
        for _, i in scored[:top_k]
    ]
