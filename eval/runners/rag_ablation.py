"""RAG-level ablation runner.

Replays the golden QA dataset through core.rag.retrieve_for_query under
each condition (rag level / rerank backend), scores the merged
user+framework ranking with eval.metrics.retrieval, and reports per
condition. Ingestion mirrors the upload path (chunk_text_enriched ->
add_user_document_chunks) into a dedicated eval engagement.

Refuses to run against the hash-stub embedding backend unless
allow_stub=True: stub vectors are non-semantic and such runs exercise
wiring only.

CLI: python -m eval.runners.rag_ablation --help
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime

from core import embeddings, rerank, store, vectorstore
from core.config import settings
from core.ingestion import chunk_text_enriched, extract_text
from core.rag import retrieve_for_query

from eval import dataset as dataset_io
from eval.metrics.retrieval import aggregate, evaluate_item

CONDITIONS = [
    {"id": "basic", "rag_level": "basic"},
    {"id": "standard", "rag_level": "standard"},
    # The "full" condition pins the rerank backend so the ablation always
    # compares with the cross-encoder engaged (app default is opt-in).
    {
        "id": "full",
        "rag_level": "full",
        "rerank_backend": "sentence-transformers",
    },
    {"id": "full-no-rerank", "rag_level": "full", "rerank_backend": "none"},
]

DEFAULT_CLIENT_NAME = "Fortis"
DEFAULT_ENGAGEMENT_NAME = "Eval Ablation"

created_engagement_ids = []


def embedding_backend() -> str:
    return getattr(
        embeddings, "_BACKEND", os.getenv("EMBEDDING_BACKEND", "sentence-transformers")
    )


def _assert_embedding_backend(allow_stub: bool) -> None:
    if embedding_backend() == "hash-stub" and not allow_stub:
        raise RuntimeError(
            "Embedding backend is 'hash-stub': dense retrieval scores would "
            "be meaningless. Unset EMBEDDING_BACKEND=hash-stub for a real "
            "run, or pass allow_stub=True (CLI: --allow-stub) to proceed "
            "for wiring checks only."
        )


def _required_sample_docs():
    return dataset_io.list_sample_docs()


def _create_tracked(client_name: str, engagement_name: str):
    before = {eng["id"] for eng in store.list_engagements()}
    engagement = store.create_engagement(client_name, engagement_name)
    created = [] if engagement["id"] in before else [engagement["id"]]
    return engagement["id"], created


def _ensure_engagement(engagement_id):
    store.init_db()
    if engagement_id:
        if store.get_engagement(engagement_id):
            return engagement_id, []
        actual_id, created = _create_tracked("Fortis Eval", engagement_id)
        return actual_id, created
    return _create_tracked(DEFAULT_CLIENT_NAME, DEFAULT_ENGAGEMENT_NAME)


def _ensure_documents(engagement_id, doc_names, log) -> None:
    present = set(vectorstore.list_engagement_files(engagement_id))
    for name in doc_names:
        if name in present:
            continue
        path = os.path.join(dataset_io.SAMPLE_DOCS_DIR, name)
        text = extract_text(path)
        if not text.strip():
            log(f"warning: {name} yielded no text; skipped")
            continue
        chunks, chunk_meta = chunk_text_enriched(text, name, return_meta=True)
        added = vectorstore.add_user_document_chunks(
            engagement_id, name, chunks, chunk_meta
        )
        log(f"ingested {name}: {added} chunks")


def _percentile(sorted_values, q) -> float:
    if not sorted_values:
        return 0.0
    index = max(0, math.ceil(q * len(sorted_values)) - 1)
    return round(float(sorted_values[index]), 3)


def _condition_rerank_active() -> bool:
    rerank.reset_retry_state()
    rerank._RERANK_CHECKED = False
    return bool(rerank.is_available())


def run_ablation(
    dataset,
    conditions=None,
    ks=(3, 5, 10),
    engagement_id=None,
    allow_stub=False,
    log=None,
):
    _assert_embedding_backend(allow_stub)
    conditions = list(conditions) if conditions else list(CONDITIONS)
    items = list((dataset or {}).get("items") or [])
    if not items:
        raise ValueError("dataset has no items to evaluate")
    noop = lambda message: None
    log = log or noop

    engagement_id, created = _ensure_engagement(engagement_id)
    created_engagement_ids.extend(created)
    log(f"eval engagement: {engagement_id}")
    _ensure_documents(engagement_id, _required_sample_docs(), log)

    saved_rag_level = settings.rag_level
    saved_rerank_backend = settings.rerank_backend
    condition_results = []
    try:
        for condition in conditions:
            settings.rag_level = condition["rag_level"]
            settings.rerank_backend = condition.get(
                "rerank_backend", saved_rerank_backend
            )
            rerank_active = _condition_rerank_active()
            log(
                f"condition {condition['id']}: rag_level={settings.rag_level} "
                f"rerank_backend={settings.rerank_backend} "
                f"rerank_active={rerank_active}"
            )

            per_item = []
            for item in items:
                try:
                    started = time.perf_counter()
                    retrieved = retrieve_for_query(engagement_id, item["question"])
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    merged = list(retrieved["user_hits"]) + list(
                        retrieved["framework_hits"]
                    )
                    entry = evaluate_item(item, merged, ks)
                    entry["latency_ms"] = round(latency_ms, 3)
                    entry["rerank_active"] = rerank_active
                    per_item.append(entry)
                except Exception as exc:
                    per_item.append({"id": item.get("id"), "error": str(exc)})
                    log(f"item {item.get('id')} failed under "
                        f"{condition['id']}: {exc}")

            latencies = sorted(
                entry["latency_ms"] for entry in per_item if "latency_ms" in entry
            )
            condition_results.append(
                {
                    "id": condition["id"],
                    "aggregate": aggregate(per_item, ks),
                    "per_item": per_item,
                    "latency": {
                        "p50_ms": _percentile(latencies, 0.5),
                        "p95_ms": _percentile(latencies, 0.95),
                    },
                    "backend": {
                        "embedding": embedding_backend(),
                        "rag_level": settings.rag_level,
                        "rerank_active": rerank_active,
                    },
                }
            )
    finally:
        settings.rag_level = saved_rag_level
        settings.rerank_backend = saved_rerank_backend

    return {
        "conditions": condition_results,
        "ks": list(ks),
        "engagement_id": engagement_id,
        "dataset_items": len(items),
    }


def _write_results_json(out_dir, result):
    path = os.path.join(out_dir, "results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, sort_keys=True, ensure_ascii=False, indent=2)
    return path


def _write_results_csv(out_dir, result):
    path = os.path.join(out_dir, "results.csv")
    metric_columns = []
    for k in result["ks"]:
        metric_columns += [
            f"recall@{k}",
            f"precision@{k}",
            f"ndcg@{k}",
            f"sufficiency@{k}",
        ]
    header = (
        ["condition", "id", "error", "n_hits", "mrr", "first_relevant_rank",
         "latency_ms", "rerank_active"]
        + metric_columns
    )
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for condition in result["conditions"]:
            for entry in condition["per_item"]:
                row = [
                    condition["id"],
                    entry.get("id"),
                    entry.get("error", ""),
                    entry.get("n_hits", ""),
                    entry.get("mrr", ""),
                    entry.get("first_relevant_rank", ""),
                    entry.get("latency_ms", ""),
                    entry.get("rerank_active", ""),
                ]
                row += [entry.get(column, "") for column in metric_columns]
                writer.writerow(row)
    return path


_LIMITATIONS = (
    "This harness issues single-turn queries, so query condensation is "
    "nearly a no-op and cannot differentiate conditions here. Runs made "
    "with the hash-stub embedding backend exercise wiring only: the "
    "pseudo-embeddings are non-semantic, so dense-only tiers score on "
    "noise and only the lexical/BM25 path carries signal. Evidence anchors "
    "are matched lexically (casefolded, whitespace-normalized substring "
    "containment), so correctly paraphrased evidence counts as missed. "
    "Real embeddings and multi-turn tasks are required before drawing "
    "quality conclusions."
)


def _write_summary_md(out_dir, result, manifest_report):
    path = os.path.join(out_dir, "summary.md")
    lines = [
        "# Fortis RAG ablation",
        "",
        f"- embedding backend: {result['conditions'][0]['backend']['embedding'] if result['conditions'] else 'n/a'}",
        f"- dataset items: {result['dataset_items']}",
        f"- engagement: {result['engagement_id']}",
        f"- manifest verification: ok={manifest_report.get('ok')} "
        f"mismatched={manifest_report.get('mismatched', [])}",
        "",
        "## Conditions",
        "",
        "| condition | rag_level | rerank_active | p50_ms | p95_ms |",
        "| --- | --- | --- | --- | --- |",
    ]
    for condition in result["conditions"]:
        backend = condition["backend"]
        lines.append(
            f"| {condition['id']} | {backend['rag_level']} | "
            f"{backend['rerank_active']} | {condition['latency']['p50_ms']} | "
            f"{condition['latency']['p95_ms']} |"
        )
    for k in result["ks"]:
        lines += [
            "",
            f"## Retrieval quality @ K={k}",
            "",
            f"| condition | recall@{k} | precision@{k} | ndcg@{k} | mrr | "
            f"sufficiency@{k} | p50_ms | p95_ms |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for condition in result["conditions"]:
            agg = condition["aggregate"]
            lines.append(
                f"| {condition['id']} | {agg.get(f'recall@{k}', '')} | "
                f"{agg.get(f'precision@{k}', '')} | {agg.get(f'ndcg@{k}', '')} | "
                f"{agg.get('mrr', '')} | {agg.get(f'sufficiency@{k}', '')} | "
                f"{condition['latency']['p50_ms']} | "
                f"{condition['latency']['p95_ms']} |"
            )
    lines += ["", "## LIMITATIONS", "", _LIMITATIONS, ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def _write_outputs(out_dir, result, manifest_report, log):
    os.makedirs(out_dir, exist_ok=True)
    writers = (
        ("results.json", lambda: _write_results_json(out_dir, result)),
        ("results.csv", lambda: _write_results_csv(out_dir, result)),
        ("summary.md", lambda: _write_summary_md(out_dir, result, manifest_report)),
    )
    for label, writer in writers:
        try:
            log(f"wrote {writer()}")
        except OSError as exc:
            log(f"warning: could not write {label}: {exc}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m eval.runners.rag_ablation",
        description="Run the golden QA dataset through RAG-level ablation "
        "conditions and write results.json / results.csv / summary.md.",
    )
    parser.add_argument(
        "--conditions",
        default=",".join(condition["id"] for condition in CONDITIONS),
        help="comma-separated condition ids (default: all)",
    )
    parser.add_argument(
        "--k", default="3,5,10", help="comma-separated K values (default: 3,5,10)"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output directory (default: eval/runs/<YYYYmmdd-HHMMSS>)",
    )
    parser.add_argument(
        "--allow-stub",
        action="store_true",
        help="proceed even with the hash-stub embedding backend (wiring check only)",
    )
    parser.add_argument(
        "--engagement",
        default=None,
        help="engagement id to reuse/create for ingestion (default: the eval engagement)",
    )
    args = parser.parse_args(argv)

    requested = [part.strip() for part in args.conditions.split(",") if part.strip()]
    known = {condition["id"]: condition for condition in CONDITIONS}
    if not requested:
        parser.error("--conditions must list at least one id")
    unknown = [name for name in requested if name not in known]
    if unknown:
        parser.error(f"unknown condition(s) {unknown}; valid ids: {sorted(known)}")
    conditions = [known[name] for name in requested]

    try:
        ks = tuple(int(part.strip()) for part in args.k.split(",") if part.strip())
        if not ks or any(value <= 0 for value in ks):
            raise ValueError
    except ValueError:
        parser.error(f"--k must be positive integers, got: {args.k!r}")

    try:
        dataset = dataset_io.load_golden_dataset()
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        manifest_report = dataset_io.verify_manifest(dataset_io.DATASETS_DIR)
    except Exception as exc:
        manifest_report = {"ok": False, "mismatched": [f"verify failed: {exc}"]}

    out_dir = args.out or os.path.join(
        "eval", "runs", datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    try:
        result = run_ablation(
            dataset,
            conditions=conditions,
            ks=ks,
            engagement_id=args.engagement,
            allow_stub=args.allow_stub,
            log=print,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _write_outputs(out_dir, result, manifest_report, log=print)
    return 0


if __name__ == "__main__":
    sys.exit(main())
