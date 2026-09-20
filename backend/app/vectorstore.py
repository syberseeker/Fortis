import uuid
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import settings
from . import bm25 as bm25_index
from .embeddings import embed_texts, embed_query

_client = chromadb.PersistentClient(
    path=settings.chroma_persist_dir,
    settings=ChromaSettings(anonymized_telemetry=False),
)

USER_DOCS_COLLECTION = "user_documents"
FRAMEWORKS_COLLECTION = "security_frameworks"

_user_docs = _client.get_or_create_collection(USER_DOCS_COLLECTION)
_frameworks = _client.get_or_create_collection(FRAMEWORKS_COLLECTION)


def add_user_document_chunks(
    engagement_id: str,
    filename: str,
    chunks: List[str],
) -> int:
    if not chunks:
        return 0
    ids = [f"{engagement_id}::{filename}::{i}::{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
    metadatas = [
        {"engagement_id": engagement_id, "filename": filename, "chunk_index": i}
        for i in range(len(chunks))
    ]
    embeddings = embed_texts(chunks)
    _user_docs.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    bm25_index.mark_dirty(USER_DOCS_COLLECTION)
    return len(chunks)


def query_user_documents(engagement_id: str, query: str, top_k: int) -> List[Dict]:
    result = _user_docs.query(
        query_embeddings=[embed_query(query)],
        n_results=top_k,
        where={"engagement_id": engagement_id},
    )
    return _flatten(result)


def query_frameworks(query: str, top_k: int, framework: Optional[str] = None) -> List[Dict]:
    where = {"framework": framework} if framework else None
    result = _frameworks.query(
        query_embeddings=[embed_query(query)],
        n_results=top_k,
        where=where,
    )
    return _flatten(result)


def seed_framework_chunks(framework: str, chunks: List[Dict[str, str]], force: bool = False) -> None:
    """chunks: list of dicts with at least {"id": control_id, "text": control_text}.
    Optional keys (title, version, source_url, function, category) are carried
    through into Chroma metadata for citation/transparency in reports.
    If force=True, replaces any existing entries for this framework (used when
    loading verbatim official text over the built-in paraphrased seed)."""
    if not chunks:
        return
    existing = _frameworks.get(where={"framework": framework})
    if existing["ids"]:
        if not force:
            return  # already seeded, leave as-is
        _frameworks.delete(ids=existing["ids"])

    ids = [f"{framework}::{c['id']}" for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = []
    for c in chunks:
        meta = {"framework": framework, "control_id": c["id"], "source_type": "built_in"}
        for optional_key in ("title", "version", "source_url", "function", "category"):
            if c.get(optional_key):
                meta[optional_key] = c[optional_key]
        metadatas.append(meta)
    embeddings = embed_texts(texts)
    _frameworks.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    bm25_index.mark_dirty(FRAMEWORKS_COLLECTION)


def load_verbatim_framework_document(framework: str, source_name: str, chunks: List[str]) -> int:
    """Ingests a user-supplied official standard document (e.g. the literal
    NIST/OWASP/CIS PDF, parsed and chunked the same way as an uploaded client
    document) as the authoritative text for `framework`, replacing any
    built-in paraphrased seed for that framework. See app/frameworks/loader.py."""
    if not chunks:
        return 0
    existing = _frameworks.get(where={"framework": framework})
    if existing["ids"]:
        _frameworks.delete(ids=existing["ids"])

    ids = [f"{framework}::verbatim::{i}::{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
    metadatas = [
        {
            "framework": framework,
            "control_id": f"{source_name} chunk {i}",
            "source_type": "verbatim_official",
            "source_name": source_name,
        }
        for i in range(len(chunks))
    ]
    embeddings = embed_texts(chunks)
    _frameworks.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    bm25_index.mark_dirty(FRAMEWORKS_COLLECTION)
    return len(chunks)


def get_document_chunks(engagement_id: str, filename: str) -> List[str]:
    """Returns all chunks for one uploaded file, in original order -- used by
    the map-reduce analysis pass to process a full document rather than a
    similarity-search subset."""
    result = _user_docs.get(where={"$and": [
        {"engagement_id": engagement_id}, {"filename": filename}
    ]})
    pairs = sorted(
        zip(result["metadatas"], result["documents"]),
        key=lambda p: p[0]["chunk_index"],
    )
    return [text for _, text in pairs]


def list_engagement_files(engagement_id: str) -> List[str]:
    result = _user_docs.get(where={"engagement_id": engagement_id})
    return sorted({m["filename"] for m in result["metadatas"]})


def clear_engagement(engagement_id: str) -> None:
    ids = _user_docs.get(where={"engagement_id": engagement_id})["ids"]
    if ids:
        _user_docs.delete(ids=ids)
        bm25_index.mark_dirty(USER_DOCS_COLLECTION)


def _rrf_fuse(dense_hits: List[Dict], bm25_hits: List[Dict], top_k: int, k: int) -> List[Dict]:
    """Reciprocal Rank Fusion of dense and BM25 result lists. Hits are keyed
    by (text, metadata) identity since dense hits carry no chroma id."""

    def _key(hit: Dict):
        return (hit["text"], tuple(sorted((hit["metadata"] or {}).items())))

    fused = {}
    for rank, hit in enumerate(dense_hits, start=1):
        fused.setdefault(_key(hit), {"hit": hit, "score": 0.0})
        fused[_key(hit)]["score"] += 1.0 / (k + rank)
    for rank, hit in enumerate(bm25_hits, start=1):
        fused.setdefault(_key(hit), {"hit": hit, "score": 0.0})
        fused[_key(hit)]["score"] += 1.0 / (k + rank)
    ranked = sorted(fused.values(), key=lambda entry: entry["score"], reverse=True)
    return [entry["hit"] for entry in ranked[:top_k]]


def query_user_documents_hybrid(
    engagement_id: str, query: str, top_k: int, candidate_multiplier: int = 4
) -> List[Dict]:
    """Dense retrieval fused with BM25 keyword scoring (RRF). The BM25 index
    is scoped to the engagement via a metadata getter, mirroring the dense
    where filter. Returns the same shape as query_user_documents."""
    dense = query_user_documents(engagement_id, query, top_k * candidate_multiplier)

    def _getter():
        return _user_docs.get(where={"engagement_id": engagement_id})

    bm25_hits = bm25_index.search(
        USER_DOCS_COLLECTION,
        engagement_id,
        _getter,
        query,
        top_k * candidate_multiplier,
        where_pred=lambda meta: (meta or {}).get("engagement_id") == engagement_id,
    )
    return _rrf_fuse(dense, bm25_hits, top_k, settings.bm25_rrf_k)


def query_frameworks_hybrid(
    query: str, top_k: int, candidate_multiplier: int = 4, framework: Optional[str] = None
) -> List[Dict]:
    """Hybrid retrieval against the framework collection; optional framework
    filter applies to both the dense where clause and the BM25 predicate."""
    dense = query_frameworks(query, top_k * candidate_multiplier, framework=framework)

    def _pred(meta) -> bool:
        return framework is None or (meta or {}).get("framework") == framework

    def _getter():
        return _frameworks.get()

    bm25_hits = bm25_index.search(
        FRAMEWORKS_COLLECTION,
        framework or "_all",
        _getter,
        query,
        top_k * candidate_multiplier,
        where_pred=_pred,
    )
    return _rrf_fuse(dense, bm25_hits, top_k, settings.bm25_rrf_k)


def _flatten(result: Dict) -> List[Dict]:
    out = []
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0] if result.get("distances") else [None] * len(docs)
    for doc, meta, dist in zip(docs, metas, dists):
        out.append({"text": doc, "metadata": meta, "distance": dist})
    return out
