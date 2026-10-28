"""Grounding evaluation runner (Phase 2, Stage A).

Replays the golden QA dataset through retrieval + the local LLM and scores
each reply with eval.metrics.grounding: citation correctness against the
seeded corpus vocabulary, abstention behavior on the dataset's traps (and
false abstention on answerable items), and lexical grounding against the
retrieved evidence plus verbatim anchor hits.

Ingestion and engagement handling mirror eval.runners.rag_ablation (the
shared eval engagement is reused when it already exists). Scoring is
deterministic; no judge model is used at this stage.

Stub guard: refuses to run when the LLM backend resolves to 'stub' (canned
replies would make every metric meaningless) unless allow_stub=True.

CLI: python -m eval.runners.grounding_eval --help
"""
import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime

from core.config import settings
from core.rag import build_chat_messages, retrieve_for_query
from core import vectorstore
from eval import dataset as dataset_io
from eval.metrics.grounding import aggregate, score_item
from eval.runners.rag_ablation import _ensure_documents, _ensure_engagement

import engine

created_engagement_ids = []

_LIMITATIONS = (
    "Stage-A scoring is lexical and judge-free: abstention detection is "
    "marker-based (a reply that legitimately says 'cannot' inside evidence "
    "counts as abstaining), grounding is content-word overlap against "
    "retrieved excerpts (paraphrases that share no wording count as "
    "ungrounded), and unverifiable citation tokens are id-SHAPED tokens the "
    "corpus cannot back (algorithm names like SHA-256 land there too; they "
    "are reported, not labeled fabrication). Single-turn only, so query "
    "condensation is a no-op. Stage B (LLM-judged claim support) is required "
    "before drawing semantic conclusions."
)


def _assert_real_llm(allow_stub: bool) -> None:
    try:
        backend = engine._resolve()
    except Exception:
        backend = "stub"
    if backend != "llama" and not allow_stub:
        raise RuntimeError(
            "LLM backend resolves to '%s': grounding scores over stub "
            "replies would be meaningless. Point FORTIS_LLM_BACKEND at a "
            "real model (or unset it with a downloaded GGUF), or pass "
            "allow_stub=True (CLI: --allow-stub) to proceed for wiring "
            "checks only." % backend
        )
    if backend == "llama":
        from engine.llama_engine import model_status

        warning = model_status()
        if warning:
            raise RuntimeError("no usable model: %s" % warning)


def _corpus_control_ids() -> set:
    """Vocabulary of every control_id in the seeded framework corpus, for
    citation verification."""
    ids = set()
    try:
        data = vectorstore.framework_collection().get(include=["metadatas"])
        for meta in data.get("metadatas") or []:
            control_id = meta.get("control_id")
            if control_id:
                ids.add(str(control_id))
    except Exception:
        pass
    return ids


async def _answer(item: dict, engagement_id: str):
    retrieved = retrieve_for_query(engagement_id, item["question"])
    messages = build_chat_messages(
        engagement_id, item["question"], history=[], retrieved=retrieved
    )
    reply = await asyncio.wait_for(
        engine.chat(messages, temperature=0.2),
        timeout=settings.llm_call_timeout_seconds,
    )
    return reply, retrieved


async def _run_items(items, engagement_id, corpus_ids, log):
    per_item = []
    for item in items:
        started = time.perf_counter()
        try:
            reply, retrieved = await _answer(item, engagement_id)
            hits = list(retrieved["user_hits"]) + list(retrieved["framework_hits"])
            evidence = [h.get("text", "") for h in hits if h.get("text")]
            provenance = [str(h.get("id")) for h in hits]
            score = score_item(
                item, reply, evidence, corpus_control_ids=corpus_ids
            )
            entry = {
                "id": item.get("id"),
                "kind": item.get("kind"),
                "reply": reply,
                "provenance": provenance,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }
            entry.update(score)
            log("scored %s (%s)" % (entry["id"], entry["kind"]))
        except Exception as exc:
            entry = {"id": item.get("id"), "kind": item.get("kind"), "error": str(exc)}
            log("item %s failed: %s" % (entry.get("id"), exc))
        per_item.append(entry)
    return per_item


def run_grounding_eval(
    dataset,
    engagement_id=None,
    rag_level=None,
    allow_stub=False,
    limit=None,
    log=None,
):
    _assert_real_llm(allow_stub)
    items = list((dataset or {}).get("items") or [])
    if not items:
        raise ValueError("dataset has no items to evaluate")
    if limit is not None:
        items = items[: max(1, int(limit))]
    noop = lambda message: None
    log = log or noop

    saved_rag_level = settings.rag_level
    if rag_level:
        settings.rag_level = rag_level
    try:
        engagement_id, created = _ensure_engagement(engagement_id)
        created_engagement_ids.extend(created)
        log("eval engagement: %s" % engagement_id)
        _ensure_documents(engagement_id, dataset_io.list_sample_docs(), log)
        corpus_ids = _corpus_control_ids()
        per_item = asyncio.run(_run_items(items, engagement_id, corpus_ids, log))
    finally:
        settings.rag_level = saved_rag_level

    scored = [entry for entry in per_item if "error" not in entry]
    return {
        "per_item": per_item,
        "aggregate": aggregate(scored),
        "engagement_id": engagement_id,
        "dataset_items": len(items),
        "backend": {
            "rag_level": saved_rag_level,
            "llm": engine.get_backend(),
        },
    }


def _write_results_json(out_dir, result):
    path = os.path.join(out_dir, "results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, sort_keys=True, ensure_ascii=False, indent=2)
    return path


_CSV_COLUMNS = [
    "id", "kind", "error", "latency_ms", "abstained", "false_abstention",
    "sentences", "grounded_sentences", "sentence_grounded_fraction",
    "gold_citation_recall", "anchor_hit", "n_unverifiable_citations",
    "provenance",
]


def _write_results_csv(out_dir, result):
    path = os.path.join(out_dir, "results.csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_COLUMNS)
        for entry in result["per_item"]:
            row = [
                entry.get("id", ""),
                entry.get("kind", ""),
                entry.get("error", ""),
                entry.get("latency_ms", ""),
                entry.get("abstained", ""),
                entry.get("false_abstention", ""),
                entry.get("sentences", ""),
                entry.get("grounded_sentences", ""),
                entry.get("sentence_grounded_fraction", ""),
                entry.get("gold_citation_recall", ""),
                entry.get("anchor_hit", ""),
                len(entry.get("unverifiable_citation_tokens") or []),
                ";".join(entry.get("provenance") or []),
            ]
            writer.writerow(row)
    return path


def _write_summary_md(out_dir, result, manifest_report):
    path = os.path.join(out_dir, "summary.md")
    agg = result["aggregate"]
    lines = [
        "# Fortis grounding evaluation (Stage A)",
        "",
        "- dataset items: %d" % result["dataset_items"],
        "- engagement: %s" % result["engagement_id"],
        "- rag_level: %s" % result["backend"]["rag_level"],
        "- manifest verification: ok=%s mismatched=%s"
        % (manifest_report.get("ok"), manifest_report.get("mismatched", [])),
        "",
        "## Aggregates",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key, value in agg.items():
        lines.append("| %s | %s |" % (key, value))
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
            log("wrote %s" % writer())
        except OSError as exc:
            log("warning: could not write %s: %s" % (label, exc))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m eval.runners.grounding_eval",
        description="Run the golden QA dataset through retrieval + the local "
        "LLM and score grounding (citations, abstention, lexical overlap).",
    )
    parser.add_argument(
        "--allow-stub",
        action="store_true",
        help="proceed even with the stub LLM backend (wiring check only)",
    )
    parser.add_argument(
        "--limit", default=None, help="evaluate only the first N items"
    )
    parser.add_argument(
        "--rag-level", default=None, help="override RAG_LEVEL for this run"
    )
    parser.add_argument(
        "--engagement", default=None,
        help="engagement id to reuse/create (default: the eval engagement)",
    )
    parser.add_argument(
        "--out", default=None,
        help="output directory (default: eval/runs/<YYYYmmdd-HHMMSS>)",
    )
    args = parser.parse_args(argv)

    try:
        dataset = dataset_io.load_golden_dataset()
    except (FileNotFoundError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    try:
        manifest_report = dataset_io.verify_manifest(dataset_io.DATASETS_DIR)
    except Exception as exc:
        manifest_report = {"ok": False, "mismatched": ["verify failed: %s" % exc]}

    out_dir = args.out or os.path.join(
        "eval", "runs", datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    try:
        result = run_grounding_eval(
            dataset,
            engagement_id=args.engagement,
            rag_level=args.rag_level,
            allow_stub=args.allow_stub,
            limit=int(args.limit) if args.limit else None,
            log=print,
        )
    except RuntimeError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    _write_outputs(out_dir, result, manifest_report, log=print)
    return 0


if __name__ == "__main__":
    sys.exit(main())
