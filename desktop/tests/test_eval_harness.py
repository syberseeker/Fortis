"""Offline tests for the eval harness (Phase 2): pure metric math with
exact expected values, dataset schema validation, evidence-anchor
verification, manifest round-trips, and end-to-end runner wiring plus
run-to-run determinism under the hash-stub embedding backend. No GPU, no
model, no network — see conftest.py for the environment (hash-stub
embeddings, RERANK_BACKEND=none). The real eval/datasets/golden_qa.json is
authored separately and is never read here: these tests drive every API
with tiny synthetic datasets."""
import json
import math

import pytest

from core import config, ingestion, store, vectorstore
from core.frameworks import seed_all_frameworks
from eval import dataset as eval_ds
from eval.metrics import retrieval as metrics
from eval.runners import rag_ablation


def _hit(hit_id, text, rank):
    return {
        "id": hit_id,
        "text": text,
        "metadata": {},
        "distance": None,
        "rank": rank,
    }


HITS = [
    _hit("h1", "Alpha zzanchor_one appears here", 1),
    _hit("h2", "Beta zzanchor_two and more filler", 2),
    _hit("h3", "Gamma irrelevant filler", 3),
]
HITS_LATE = [HITS[2], HITS[0], HITS[1]]
ANCHORS = ["zzanchor_one", "zzanchor_two"]


# ---- metrics: exact values ---------------------------------------------------

def test_norm_casefolds_and_collapses_whitespace():
    assert metrics._norm("  Foo\tBAR\n\nbaz  ") == "foo bar baz"


def test_is_relevant_matches_normalized_substrings():
    assert metrics.is_relevant("Doc says ZZANCHOR_ONE loudly", ["zzanchor_one"])
    assert not metrics.is_relevant("nothing here", ["zzanchor_one"])
    assert not metrics.is_relevant("blank anchors", [])


def test_recall_at_k_exact_fractions():
    assert metrics.recall_at_k(HITS, ANCHORS, 1) == 0.5
    assert metrics.recall_at_k(HITS, ANCHORS, 2) == 1.0
    assert metrics.recall_at_k(HITS, ANCHORS, 3) == 1.0
    assert metrics.recall_at_k(HITS, ANCHORS, 5) == 1.0
    assert metrics.recall_at_k(HITS, [], 3) == 0.0


def test_precision_at_k_exact_fractions():
    assert metrics.precision_at_k(HITS, ANCHORS, 1) == 1.0
    assert metrics.precision_at_k(HITS, ANCHORS, 2) == 1.0
    assert metrics.precision_at_k(HITS, ANCHORS, 3) == pytest.approx(2.0 / 3.0)
    assert metrics.precision_at_k([], ANCHORS, 3) == 0.0


def test_mrr_uses_first_relevant_rank():
    assert metrics.mrr(HITS, ANCHORS) == 1.0
    assert metrics.mrr(HITS_LATE, ANCHORS) == 0.5
    assert metrics.mrr([HITS[2]], ANCHORS) == 0.0
    assert metrics.mrr(HITS, []) == 0.0


def test_ndcg_at_k_exact_binary_relevance():
    gain_rank2 = 1.0 / math.log2(3)
    idcg2 = 1.0 + gain_rank2
    assert metrics.ndcg_at_k(HITS, ANCHORS, 2) == pytest.approx(1.0)
    assert metrics.ndcg_at_k(HITS, ANCHORS, 1) == pytest.approx(1.0)
    assert metrics.ndcg_at_k(HITS_LATE, ANCHORS, 2) == pytest.approx(
        gain_rank2 / idcg2
    )
    assert metrics.ndcg_at_k(HITS_LATE, ANCHORS, 3) == pytest.approx(
        (gain_rank2 + 1.0 / math.log2(4)) / idcg2
    )
    assert metrics.ndcg_at_k(HITS, [], 3) == 0.0


def test_context_sufficiency_requires_every_anchor_in_top_k():
    assert metrics.context_sufficiency(HITS, ANCHORS, 2) is True
    assert metrics.context_sufficiency(HITS, ANCHORS, 1) is False
    assert metrics.context_sufficiency(HITS_LATE, ANCHORS, 3) is True
    assert metrics.context_sufficiency(HITS, [], 1) is True


def test_evaluate_item_exact_values_and_keys():
    item = {"id": "unit-a", "question": "q", "evidence_anchors": ANCHORS}
    result = metrics.evaluate_item(item, HITS, ks=(1, 2, 3))
    assert result["id"] == "unit-a"
    assert result["n_hits"] == 3
    assert result["mrr"] == 1.0
    assert result["first_relevant_rank"] == 1
    assert result["recall@1"] == 0.5
    assert result["recall@2"] == 1.0
    assert result["precision@2"] == 1.0
    assert result["ndcg@2"] == pytest.approx(1.0)
    assert result["sufficiency@2"] is True
    assert result["sufficiency@1"] is False

    late = metrics.evaluate_item(item, HITS_LATE, ks=(2,))
    assert late["mrr"] == 0.5
    assert late["first_relevant_rank"] == 2

    empty = metrics.evaluate_item(
        {"id": "unit-b", "evidence_anchors": []}, HITS, ks=(3,)
    )
    assert empty["recall@3"] == 0.0
    assert empty["mrr"] == 0.0
    assert empty["first_relevant_rank"] is None
    assert empty["sufficiency@3"] is True


def test_evaluate_item_is_deterministic():
    item = {"id": "unit-a", "question": "q", "evidence_anchors": ANCHORS}
    assert metrics.evaluate_item(item, HITS, ks=(2, 3)) == metrics.evaluate_item(
        item, HITS, ks=(2, 3)
    )


def test_aggregate_means_bools_and_skips_error_rows():
    item_a = {"id": "a", "question": "q", "evidence_anchors": ANCHORS}
    item_b = {"id": "b", "question": "q", "evidence_anchors": ["zzanchor_one"]}
    results = [
        metrics.evaluate_item(item_a, HITS, ks=(2, 3)),
        metrics.evaluate_item(item_b, HITS_LATE, ks=(2, 3)),
        {"id": "c", "error": "boom"},
    ]
    agg = metrics.aggregate(results, ks=(2, 3))
    assert agg["n_items"] == 2
    assert agg["recall@2"] == pytest.approx(1.0)
    assert agg["recall@3"] == pytest.approx(1.0)
    assert agg["mrr"] == pytest.approx(0.75)
    assert agg["sufficiency@2"] == pytest.approx(1.0)
    assert agg["first_relevant_rank"] == pytest.approx(1.5)
    assert agg == metrics.aggregate(results, ks=(2, 3))


# ---- dataset schema validation -----------------------------------------------

def _valid_item(**overrides):
    item = {
        "id": "ok-item",
        "kind": "framework_qa",
        "question": "What does NIST CSF DE.CM cover?",
        "source_docs": [],
        "evidence_anchors": [],
        "gold_claims": ["DE.CM covers continuous monitoring"],
        "gold_control_ids": [{"framework": "NIST_CSF", "control_id": "DE.CM"}],
        "abstention": False,
        "expected_behavior": None,
        "difficulty": "easy",
    }
    item.update(overrides)
    return item


def test_validate_item_accepts_minimal_valid_item():
    assert eval_ds.validate_item(_valid_item()) == []


def test_validate_item_catches_bad_kind():
    problems = eval_ds.validate_item(_valid_item(kind="trivia"))
    assert any("kind" in problem for problem in problems)


def test_validate_item_catches_missing_question():
    item = _valid_item()
    del item["question"]
    problems = eval_ds.validate_item(item)
    assert any("question" in problem for problem in problems)


def test_validate_item_catches_abstention_with_anchor():
    problems = eval_ds.validate_item(
        _valid_item(
            kind="abstention", abstention=True, evidence_anchors=["zzanchor_one"]
        )
    )
    assert any(
        "abstention items must have empty evidence_anchors" in problem
        for problem in problems
    )


def _write_dataset(tmp_path, dataset, name="golden_qa.json"):
    path = tmp_path / name
    path.write_text(
        json.dumps(dataset, sort_keys=True, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def test_load_golden_dataset_missing_file_raises_filenotfound(tmp_path):
    missing = str(tmp_path / "absent" / "golden_qa.json")
    with pytest.raises(FileNotFoundError, match="[Gg]olden QA dataset not found"):
        eval_ds.load_golden_dataset(path=missing)


def test_load_golden_dataset_rejects_bad_schema_and_duplicate_ids(tmp_path):
    bad = {
        "schema_version": "1.0",
        "description": "t",
        "items": [_valid_item(kind="trivia"), _valid_item(), _valid_item()],
    }
    with pytest.raises(ValueError, match="schema validation failed") as excinfo:
        eval_ds.load_golden_dataset(path=_write_dataset(tmp_path, bad))
    assert "duplicate id" in str(excinfo.value)


def test_load_golden_dataset_verify_anchors_round_trip(tmp_path):
    nist_anchor = (
        "indicators of compromise, and other potentially adverse events"
    )
    doc_item = _valid_item(
        id="doc-item",
        kind="doc_qa",
        question="What does the firewall ACL permit?",
        source_docs=["firewall-config.txt"],
        evidence_anchors=["permit ip any any log"],
    )
    dataset = {
        "schema_version": "1.0",
        "description": "synthetic anchor verification",
        "items": [_valid_item(evidence_anchors=[nist_anchor]), doc_item],
    }
    loaded = eval_ds.load_golden_dataset(
        path=_write_dataset(tmp_path, dataset), verify_anchors=True
    )
    assert loaded["items"][1]["id"] == "doc-item"

    bad_doc = _valid_item(
        id="bad-doc",
        kind="doc_qa",
        question="unanswerable anchor",
        source_docs=["firewall-config.txt"],
        evidence_anchors=["zzanchor_that_is_not_in_any_doc"],
    )
    bad = {"schema_version": "1.0", "description": "t", "items": [bad_doc]}
    with pytest.raises(ValueError, match="evidence anchor verification failed"):
        eval_ds.load_golden_dataset(
            path=_write_dataset(tmp_path, bad, name="bad_golden_qa.json"),
            verify_anchors=True,
        )


# ---- manifests -----------------------------------------------------------------

def test_build_and_verify_manifest_ok_then_mismatch(tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("hello eval manifest", encoding="utf-8")
    dataset = {"schema_version": "1.0", "description": "d", "items": []}
    dataset_file = tmp_path / "golden_qa.json"
    dataset_file.write_text(
        json.dumps(dataset, sort_keys=True, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    files = {
        str(dataset_file): eval_ds.sha256_file(str(dataset_file)),
        str(doc): eval_ds.sha256_file(str(doc)),
    }
    manifest = eval_ds.build_manifest(dataset, files)
    assert manifest["files"] == files
    (tmp_path / eval_ds.MANIFEST_NAME).write_text(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = eval_ds.verify_manifest(str(tmp_path))
    assert report == {"ok": True, "mismatched": []}

    doc.write_text("tampered contents", encoding="utf-8")
    report = eval_ds.verify_manifest(str(tmp_path))
    assert report["ok"] is False
    assert str(doc) in report["mismatched"]


def test_verify_manifest_missing_manifest(tmp_path):
    report = eval_ds.verify_manifest(str(tmp_path / "absent"))
    assert report["ok"] is False
    assert report["mismatched"]


# ---- runner wiring under hash-stub ----------------------------------------------

SYNTH_DOC_TEXT = (
    "Credential rotation for zzalphaevalmarker accounts is reviewed every "
    "quarter under the Fortis rotation policy.\n\n"
    "The zzbetamarker service account must never be shared between humans "
    "and automation."
)

NIST_ANCHOR = "indicators of compromise, and other potentially adverse events"


def _synthetic_dataset():
    return {
        "schema_version": "1.0",
        "description": "synthetic wiring fixture",
        "items": [
            {
                "id": "doc-alpha",
                "kind": "doc_qa",
                "question": "zzalphaevalmarker",
                "source_docs": ["eval_synth_doc.txt"],
                "evidence_anchors": ["zzalphaevalmarker"],
                "gold_claims": ["zzalphaevalmarker credentials rotate quarterly"],
                "gold_control_ids": [
                    {"framework": "NIST_CSF", "control_id": "PR.AA"}
                ],
                "abstention": False,
                "expected_behavior": None,
                "difficulty": "easy",
            },
            {
                "id": "doc-beta",
                "kind": "doc_qa",
                "question": "zzbetamarker",
                "source_docs": ["eval_synth_doc.txt"],
                "evidence_anchors": ["zzbetamarker"],
                "gold_claims": ["zzbetamarker is a dedicated service account"],
                "gold_control_ids": [
                    {"framework": "NIST_CSF", "control_id": "PR.AA"}
                ],
                "abstention": False,
                "expected_behavior": None,
                "difficulty": "medium",
            },
            {
                "id": "fw-decm",
                "kind": "framework_qa",
                "question": "monitoring assets for anomalies and adverse events",
                "source_docs": [],
                "evidence_anchors": [NIST_ANCHOR],
                "gold_claims": ["NIST CSF DE.CM covers continuous monitoring"],
                "gold_control_ids": [
                    {"framework": "NIST_CSF", "control_id": "DE.CM"}
                ],
                "abstention": False,
                "expected_behavior": None,
                "difficulty": "medium",
            },
            {
                "id": "abst-1",
                "kind": "abstention",
                "question": "What pasta sauce pairs best with a firewall audit?",
                "source_docs": [],
                "evidence_anchors": [],
                "gold_claims": [],
                "gold_control_ids": [],
                "abstention": True,
                "expected_behavior": "Decline: unrelated to cybersecurity.",
                "difficulty": "easy",
            },
        ],
    }


def test_run_ablation_refuses_hash_stub_without_flag():
    with pytest.raises(RuntimeError, match="allow_stub"):
        rag_ablation.run_ablation(
            _synthetic_dataset(), conditions=[rag_ablation.CONDITIONS[0]]
        )


def _strip_latency(per_item):
    return [
        {k: v for k, v in entry.items() if k != "latency_ms"} for entry in per_item
    ]


def test_run_ablation_wiring_and_determinism():
    store.init_db()
    engagement = store.create_engagement("Fortis Eval Harness", "Wiring Fixture")
    engagement_id = engagement["id"]
    try:
        seed_all_frameworks()
        chunks, chunk_meta = ingestion.chunk_text_enriched(
            SYNTH_DOC_TEXT, "eval_synth_doc.txt", return_meta=True
        )
        vectorstore.add_user_document_chunks(
            engagement_id, "eval_synth_doc.txt", chunks, chunk_meta
        )

        dataset = _synthetic_dataset()
        for item in dataset["items"]:
            assert eval_ds.validate_item(item) == []

        conditions = [
            rag_ablation.CONDITIONS[0],
            rag_ablation.CONDITIONS[1],
        ]
        first = rag_ablation.run_ablation(
            dataset,
            conditions=conditions,
            ks=(2, 3),
            allow_stub=True,
            engagement_id=engagement_id,
        )
        assert set(first) >= {"conditions", "ks", "engagement_id", "dataset_items"}
        assert first["engagement_id"] == engagement_id
        assert first["dataset_items"] == 4
        assert first["ks"] == [2, 3]
        assert [c["id"] for c in first["conditions"]] == ["basic", "standard"]

        for condition in first["conditions"]:
            assert set(condition) == {
                "id", "aggregate", "per_item", "latency", "backend"
            }
            assert "recall@2" in condition["aggregate"]
            assert "recall@3" in condition["aggregate"]
            assert "mrr" in condition["aggregate"]
            assert "sufficiency@3" in condition["aggregate"]
            assert set(condition["latency"]) == {"p50_ms", "p95_ms"}
            assert condition["backend"]["embedding"] == "hash-stub"
            assert condition["backend"]["rerank_active"] is False
            assert len(condition["per_item"]) == 4
            for entry in condition["per_item"]:
                assert "error" not in entry, entry.get("error")
                assert "latency_ms" in entry
                assert "rerank_active" in entry

        standard = first["conditions"][1]
        by_id = {entry["id"]: entry for entry in standard["per_item"]}
        assert by_id["doc-alpha"]["recall@3"] > 0.0
        assert by_id["doc-beta"]["recall@3"] > 0.0
        assert by_id["fw-decm"]["n_hits"] > 0

        assert config.settings.rag_level == "standard"
        assert config.settings.rerank_backend == "none"

        second = rag_ablation.run_ablation(
            dataset,
            conditions=conditions,
            ks=(2, 3),
            allow_stub=True,
            engagement_id=engagement_id,
        )
        for cond_a, cond_b in zip(first["conditions"], second["conditions"]):
            assert cond_a["aggregate"] == cond_b["aggregate"]
            assert cond_a["backend"] == cond_b["backend"]
            assert _strip_latency(cond_a["per_item"]) == _strip_latency(
                cond_b["per_item"]
            )
    finally:
        vectorstore.clear_engagement(engagement_id)
        store.delete_engagement(engagement_id)
    assert store.get_engagement(engagement_id) is None
