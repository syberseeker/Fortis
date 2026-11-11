"""Offline tests for Stage-B claim-support metrics (eval/metrics/claim_support.py)
and the stub-backend runner wiring (eval/runners/claim_support.py).

Pure Python: no GPU, no model, no network — the metric module is driven
entirely with hand-built synthetic claim lists, item scores and abstention
entries (verdict normalization, per-item scoring, aggregation, Stage-A
correlation, judge calibration, abstention behavior verdicts). The runner
is exercised on the stub LLM backend with hash-stub embeddings (see
conftest.py): the canned judge cannot produce parseable claim or abstention
verdicts, so the wiring tests assert the structure of the per-item blocks —
claims, citation attribution, framework mapping, abstention verdict — and
the presence of the aggregate keys, not semantic verdicts. The citation and
mapping blocks are scored by the real metric modules
(eval.metrics.citations, eval.metrics.mapping_accuracy) over a monkeypatched
retrieval result. The real golden dataset is never read here."""
import csv
import inspect
import sys
import types

import pytest

import eval.metrics as metrics_package
from core import config, ingestion, store, vectorstore
from core.frameworks import seed_all_frameworks
from engine import stub_backend
from eval.metrics import claim_support
from eval.runners import claim_support as claim_support_runner


@pytest.fixture(autouse=True)
def _restore_stub_usage():
    saved_usage = stub_backend._LAST_USAGE
    yield
    stub_backend._LAST_USAGE = saved_usage

SCORE_KEYS = {
    "n_claims",
    "n_parseable",
    "n_supported",
    "n_unsupported",
    "n_not_in_evidence",
    "n_unparseable",
    "support_rate",
    "item_verdict",
}

AGGREGATE_KEYS = {
    "n_items",
    "n_judge_failed",
    "mean_claim_support_rate",
    "items_fully_supported",
    "verdict_counts",
    "n_unparseable_claims",
    "mean_claims_per_item",
    "by_kind",
}

VERDICT_COUNT_KEYS = {
    "supported",
    "unsupported",
    "not_in_evidence",
    "judge_failed",
}

ABSTENTION_AGG_KEYS = {
    "n_traps",
    "verdict_counts",
    "appropriate_rate",
    "stage_a_agreement_rate",
}

ABSTENTION_VERDICT_COUNT_KEYS = {
    "appropriate_abstention",
    "fabricated_answer",
    "attempted_answer",
    "unparseable",
}


def _claims(*verdicts):
    return [{"verdict": v} for v in verdicts]


def _item_score(
    score_id,
    kind,
    item_verdict,
    support_rate,
    n_claims=1,
    n_unparseable=0,
):
    return {
        "id": score_id,
        "kind": kind,
        "n_claims": n_claims,
        "n_unparseable": n_unparseable,
        "support_rate": support_rate,
        "item_verdict": item_verdict,
    }


# ---- normalize_verdict ---------------------------------------------------------

def test_normalize_verdict_accepts_canonical_forms():
    assert claim_support.normalize_verdict("supported") == "supported"
    assert claim_support.normalize_verdict("unsupported") == "unsupported"
    assert claim_support.normalize_verdict("not_in_evidence") == "not_in_evidence"
    assert claim_support.VERDICTS == ("supported", "unsupported", "not_in_evidence")


def test_normalize_verdict_handles_case_whitespace_and_punctuation():
    assert claim_support.normalize_verdict("  SUPPORTED  ") == "supported"
    assert claim_support.normalize_verdict("Not_In_Evidence\n") == "not_in_evidence"
    assert claim_support.normalize_verdict("Supported.") == "supported"
    assert claim_support.normalize_verdict("UNSupported:") == "unsupported"


def test_normalize_verdict_maps_documented_synonyms():
    assert claim_support.normalize_verdict("entailed") == "supported"
    assert claim_support.normalize_verdict("contradicted") == "unsupported"
    assert claim_support.normalize_verdict("not in evidence") == "not_in_evidence"
    assert claim_support.normalize_verdict("not-in-evidence") == "not_in_evidence"
    assert claim_support.normalize_verdict("Unverifiable") == "not_in_evidence"


def test_normalize_verdict_returns_none_for_unknown_or_empty():
    assert claim_support.normalize_verdict("maybe") is None
    assert claim_support.normalize_verdict("partially supported") is None
    assert claim_support.normalize_verdict("") is None
    assert claim_support.normalize_verdict("   ") is None
    assert claim_support.normalize_verdict("n/a") is None
    assert claim_support.normalize_verdict(None) is None
    assert claim_support.normalize_verdict(7) is None


# ---- score_item ------------------------------------------------------------------

def test_score_item_all_supported_gives_full_rate():
    score = claim_support.score_item(_claims("supported", "Supported", "entailed"))
    assert set(score) == SCORE_KEYS
    assert score["n_claims"] == 3
    assert score["n_parseable"] == 3
    assert score["n_supported"] == 3
    assert score["n_unsupported"] == 0
    assert score["n_not_in_evidence"] == 0
    assert score["n_unparseable"] == 0
    assert score["support_rate"] == 1.0
    assert score["item_verdict"] == "supported"


def test_score_item_mixed_supported_and_unsupported_is_unsupported():
    score = claim_support.score_item(_claims("supported", "unsupported"))
    assert score["n_supported"] == 1
    assert score["n_unsupported"] == 1
    assert score["support_rate"] == pytest.approx(0.5)
    assert score["item_verdict"] == "unsupported"


def test_score_item_all_not_in_evidence():
    score = claim_support.score_item(
        _claims("not_in_evidence", "unverifiable", "not-in-evidence")
    )
    assert score["n_not_in_evidence"] == 3
    assert score["support_rate"] == 0.0
    assert score["item_verdict"] == "not_in_evidence"


def test_score_item_supported_mixed_with_not_in_evidence_is_not_in_evidence():
    score = claim_support.score_item(_claims("supported", "not_in_evidence"))
    assert score["n_supported"] == 1
    assert score["n_not_in_evidence"] == 1
    assert score["support_rate"] == pytest.approx(0.5)
    assert score["item_verdict"] == "not_in_evidence"


def test_score_item_zero_parseable_is_judge_failed():
    score = claim_support.score_item(_claims("maybe", "gibberish verdict"))
    assert score["n_claims"] == 2
    assert score["n_parseable"] == 0
    assert score["n_unparseable"] == 2
    assert score["support_rate"] is None
    assert score["item_verdict"] == "judge_failed"


def test_score_item_empty_claims_is_judge_failed():
    score = claim_support.score_item([])
    assert score["n_claims"] == 0
    assert score["n_parseable"] == 0
    assert score["support_rate"] is None
    assert score["item_verdict"] == "judge_failed"


def test_score_item_excludes_unparseable_from_support_rate_denominator():
    claims = [
        {"verdict": "supported", "reason": "quoted verbatim"},
        {"verdict": "entailed", "reason": "paraphrase holds"},
        {"verdict": "the judge rambled", "reason": "unparseable row"},
    ]
    score = claim_support.score_item(claims)
    assert score["n_claims"] == 3
    assert score["n_parseable"] == 2
    assert score["n_unparseable"] == 1
    assert score["support_rate"] == 1.0
    assert score["item_verdict"] == "supported"


# ---- aggregate ---------------------------------------------------------------------

def test_aggregate_rates_over_handbuilt_scores():
    item_scores = [
        _item_score("i1", "doc_qa", "supported", 1.0, n_claims=3),
        _item_score("i2", "doc_qa", "unsupported", 0.0, n_claims=2, n_unparseable=1),
        _item_score("i3", "report", "not_in_evidence", 0.5, n_claims=2),
        _item_score("i4", "report", "judge_failed", None, n_claims=2, n_unparseable=2),
    ]
    agg = claim_support.aggregate(item_scores)
    assert set(agg) == AGGREGATE_KEYS
    assert agg["n_items"] == 4
    assert agg["n_judge_failed"] == 1
    assert agg["mean_claim_support_rate"] == pytest.approx((1.0 + 0.0 + 0.5) / 3)
    assert agg["items_fully_supported"] == 1
    assert agg["verdict_counts"] == {
        "supported": 1,
        "unsupported": 1,
        "not_in_evidence": 1,
        "judge_failed": 1,
    }
    assert agg["n_unparseable_claims"] == 3
    assert agg["mean_claims_per_item"] == pytest.approx(9 / 4)
    assert agg["by_kind"] == {
        "doc_qa": {"n_items": 2, "mean_claim_support_rate": pytest.approx(0.5)},
        "report": {"n_items": 2, "mean_claim_support_rate": pytest.approx(0.5)},
    }


def test_aggregate_includes_zero_support_rate_items_in_mean():
    item_scores = [
        _item_score("a1", "doc_qa", "supported", 1.0),
        _item_score("a2", "doc_qa", "unsupported", 0.0),
    ]
    agg = claim_support.aggregate(item_scores)
    assert agg["n_items"] == 2
    assert agg["mean_claim_support_rate"] == pytest.approx(0.5)


def test_aggregate_by_kind_excludes_judge_failed_from_mean():
    item_scores = [
        _item_score("r1", "report", "judge_failed", None, n_claims=2),
        _item_score("r2", "report", "judge_failed", None, n_claims=2),
    ]
    agg = claim_support.aggregate(item_scores)
    assert agg["mean_claim_support_rate"] is None
    assert agg["n_judge_failed"] == 2
    assert agg["items_fully_supported"] == 0
    assert agg["by_kind"] == {
        "report": {"n_items": 2, "mean_claim_support_rate": None}
    }


def test_aggregate_is_none_safe_when_empty():
    agg = claim_support.aggregate([])
    assert set(agg) == AGGREGATE_KEYS
    assert agg["n_items"] == 0
    assert agg["n_judge_failed"] == 0
    assert agg["mean_claim_support_rate"] is None
    assert agg["items_fully_supported"] == 0
    assert agg["verdict_counts"] == {key: 0 for key in VERDICT_COUNT_KEYS}
    assert agg["n_unparseable_claims"] == 0
    assert agg["mean_claims_per_item"] is None
    assert agg["by_kind"] == {}
    assert "stage_a_correlation" not in agg


# ---- aggregate stage_a correlation -----------------------------------------------

def _correlated_scores(rates_by_id):
    return [
        _item_score(score_id, "doc_qa", "supported", rate)
        for score_id, rate in rates_by_id.items()
    ]


def test_aggregate_stage_a_correlation_monotonic_is_plus_one():
    item_scores = _correlated_scores({"i1": 0.2, "i2": 0.4, "i3": 0.6})
    stage_a = {"i1": 0.1, "i2": 0.3, "i3": 0.5, "i9": 0.9}
    agg = claim_support.aggregate(item_scores, stage_a_by_id=stage_a)
    assert set(agg) == AGGREGATE_KEYS | {"stage_a_correlation"}
    correlation = agg["stage_a_correlation"]
    assert set(correlation) == {"n_pairs", "pearson_r", "spearman_rho"}
    assert correlation["n_pairs"] == 3
    assert correlation["pearson_r"] == pytest.approx(1.0)
    assert correlation["spearman_rho"] == 1.0


def test_aggregate_stage_a_correlation_inverted_is_minus_one():
    item_scores = _correlated_scores({"i1": 0.2, "i2": 0.4, "i3": 0.6})
    stage_a = {"i1": 0.5, "i2": 0.3, "i3": 0.1}
    agg = claim_support.aggregate(item_scores, stage_a_by_id=stage_a)
    correlation = agg["stage_a_correlation"]
    assert correlation["n_pairs"] == 3
    assert correlation["pearson_r"] == pytest.approx(-1.0)
    assert correlation["spearman_rho"] == -1.0


def test_aggregate_omits_stage_a_correlation_below_three_common_ids():
    item_scores = _correlated_scores({"i1": 0.2, "i2": 0.4, "i3": 0.6})

    two_common = claim_support.aggregate(
        item_scores, stage_a_by_id={"i1": 0.1, "i2": 0.3}
    )
    assert "stage_a_correlation" not in two_common

    disjoint = claim_support.aggregate(item_scores, stage_a_by_id={"z1": 0.1})
    assert "stage_a_correlation" not in disjoint

    empty = claim_support.aggregate(item_scores, stage_a_by_id={})
    assert "stage_a_correlation" not in empty

    none_map = claim_support.aggregate(item_scores, stage_a_by_id=None)
    assert "stage_a_correlation" not in none_map


# ---- calibration_agreement ---------------------------------------------------------

def test_calibration_agreement_counts_accuracy_and_per_verdict():
    pairs = [
        {"expected": "supported", "got": "supported"},
        {"expected": "unsupported", "got": None},
        {"expected": "not_in_evidence", "got": "unsupported"},
        {"expected": "unsupported", "got": "Contradicted"},
    ]
    result = claim_support.calibration_agreement(pairs)
    assert result["n"] == 4
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["per_verdict"] == {
        "supported": {"n": 1, "correct": 1},
        "unsupported": {"n": 2, "correct": 1},
        "not_in_evidence": {"n": 1, "correct": 0},
    }


def test_calibration_agreement_empty_input():
    result = claim_support.calibration_agreement([])
    assert result["n"] == 0
    assert result["accuracy"] is None
    assert result["per_verdict"] == {
        "supported": {"n": 0, "correct": 0},
        "unsupported": {"n": 0, "correct": 0},
        "not_in_evidence": {"n": 0, "correct": 0},
    }


# ---- normalize_abstention_verdict -------------------------------------------------

def test_normalize_abstention_verdict_accepts_canonical_forms():
    assert claim_support.ABSTENTION_VERDICTS == (
        "appropriate_abstention",
        "fabricated_answer",
        "attempted_answer",
    )
    assert (
        claim_support.normalize_abstention_verdict("appropriate_abstention")
        == "appropriate_abstention"
    )
    assert (
        claim_support.normalize_abstention_verdict("fabricated_answer")
        == "fabricated_answer"
    )
    assert (
        claim_support.normalize_abstention_verdict("attempted_answer")
        == "attempted_answer"
    )


def test_normalize_abstention_verdict_maps_documented_synonyms():
    assert (
        claim_support.normalize_abstention_verdict("abstained")
        == "appropriate_abstention"
    )
    assert (
        claim_support.normalize_abstention_verdict("correct abstention")
        == "appropriate_abstention"
    )
    assert (
        claim_support.normalize_abstention_verdict("Correct_Abstention.")
        == "appropriate_abstention"
    )
    assert (
        claim_support.normalize_abstention_verdict("Hallucinated")
        == "fabricated_answer"
    )
    assert (
        claim_support.normalize_abstention_verdict(" fabricated ")
        == "fabricated_answer"
    )
    assert (
        claim_support.normalize_abstention_verdict("made up")
        == "fabricated_answer"
    )
    assert (
        claim_support.normalize_abstention_verdict("Made_Up!")
        == "fabricated_answer"
    )
    assert (
        claim_support.normalize_abstention_verdict("Answered")
        == "attempted_answer"
    )


def test_normalize_abstention_verdict_returns_none_for_unknown_or_empty():
    assert claim_support.normalize_abstention_verdict("maybe") is None
    assert claim_support.normalize_abstention_verdict("partial abstention") is None
    assert claim_support.normalize_abstention_verdict("") is None
    assert claim_support.normalize_abstention_verdict("   ") is None
    assert claim_support.normalize_abstention_verdict(None) is None
    assert claim_support.normalize_abstention_verdict(7) is None


# ---- abstention_aggregate ----------------------------------------------------------

def _abstention_entry(entry_id, verdict, marker):
    return {
        "id": entry_id,
        "kind": "abstention",
        "abstention_verdict": verdict,
        "stage_a_marker_abstained": marker,
    }


def test_abstention_aggregate_rates_over_handbuilt_entries():
    entries = [
        _abstention_entry("t1", "appropriate_abstention", True),
        _abstention_entry("t2", "fabricated", False),
        _abstention_entry("t3", "attempted_answer", True),
        _abstention_entry("t4", "appropriate_abstention", None),
    ]
    agg = claim_support.abstention_aggregate(entries)
    assert set(agg) == ABSTENTION_AGG_KEYS
    assert agg["n_traps"] == 4
    assert agg["verdict_counts"] == {
        "appropriate_abstention": 2,
        "fabricated_answer": 1,
        "attempted_answer": 1,
        "unparseable": 0,
    }
    assert agg["appropriate_rate"] == pytest.approx(0.5)
    assert agg["stage_a_agreement_rate"] == pytest.approx(2 / 3)


def test_abstention_aggregate_buckets_unparseable_and_skips_unknown_markers():
    entries = [
        _abstention_entry("t1", "gibberish verdict", True),
        _abstention_entry("t2", "unparseable", False),
    ]
    agg = claim_support.abstention_aggregate(entries)
    assert agg["n_traps"] == 2
    assert agg["verdict_counts"]["unparseable"] == 2
    assert agg["appropriate_rate"] == 0.0
    assert agg["stage_a_agreement_rate"] is None


def test_abstention_aggregate_empty_is_none_safe():
    agg = claim_support.abstention_aggregate([])
    assert set(agg) == ABSTENTION_AGG_KEYS
    assert agg["n_traps"] == 0
    assert agg["verdict_counts"] == {key: 0 for key in ABSTENTION_VERDICT_COUNT_KEYS}
    assert agg["appropriate_rate"] is None
    assert agg["stage_a_agreement_rate"] is None


def test_abstention_aggregate_none_argument_is_none_safe():
    agg = claim_support.abstention_aggregate(None)
    assert agg["n_traps"] == 0
    assert agg["appropriate_rate"] is None


# ---- runner-contract shim ---------------------------------------------------------

def test_public_contract_names_and_signatures():
    assert callable(claim_support.normalize_verdict)
    assert callable(claim_support.score_item)
    assert callable(claim_support.aggregate)
    assert callable(claim_support.calibration_agreement)

    signature = inspect.signature(claim_support.normalize_verdict)
    assert list(signature.parameters) == ["raw"]

    signature = inspect.signature(claim_support.score_item)
    assert list(signature.parameters) == ["claims"]

    signature = inspect.signature(claim_support.aggregate)
    assert list(signature.parameters) == ["item_scores", "stage_a_by_id"]
    assert signature.parameters["stage_a_by_id"].default is None

    signature = inspect.signature(claim_support.calibration_agreement)
    assert list(signature.parameters) == ["pairs"]


def test_abstention_contract_names_and_signatures():
    assert callable(claim_support.normalize_abstention_verdict)
    assert callable(claim_support.abstention_aggregate)

    signature = inspect.signature(claim_support.normalize_abstention_verdict)
    assert list(signature.parameters) == ["raw"]

    signature = inspect.signature(claim_support.abstention_aggregate)
    assert list(signature.parameters) == ["entries"]


# ---- metric-module import failures (friendly RuntimeError) -------------------------

def test_missing_citation_metric_module_raises_clear_error(monkeypatch):
    monkeypatch.delattr(metrics_package, "citations", raising=False)
    monkeypatch.setitem(sys.modules, "eval.metrics.citations", None)
    with pytest.raises(RuntimeError, match=r"eval/metrics/citations\.py"):
        claim_support_runner._import_citation_metrics()


def test_mapping_metric_missing_names_raises_clear_error(monkeypatch):
    hollow = types.ModuleType("eval.metrics.mapping_accuracy")
    monkeypatch.setattr(metrics_package, "mapping_accuracy", hollow, raising=False)
    with pytest.raises(
        RuntimeError, match=r"eval/metrics/mapping_accuracy\.py is missing"
    ):
        claim_support_runner._import_mapping_metrics()


# ---- runner wiring (stub backend, offline) ------------------------------------------

STAGE_B_DOC_TEXT = (
    "Credential rotation for zzalphaevalmarker accounts is reviewed every "
    "quarter under the Fortis rotation policy.\n\n"
    "The zzbetamarker service account must never be shared between humans "
    "and automation."
)

DOC_REPLY = "Per NIST CSF PR.PS, patch management software must be kept current."

ABST_REPLY = (
    "I cannot verify that from the available evidence; please share the "
    "advisory and I will review it."
)


def _doc_item(**overrides):
    item = {
        "id": "doc-claim-1",
        "kind": "doc_qa",
        "question": "zzalphaevalmarker",
        "source_docs": ["eval_synth_doc.txt"],
        "evidence_anchors": ["zzalphaevalmarker"],
        "gold_control_ids": [{"framework": "NIST_CSF", "control_id": "PR.PS"}],
        "abstention": False,
        "expected_behavior": None,
        "difficulty": "easy",
    }
    item.update(overrides)
    return item


def _abst_item(**overrides):
    item = {
        "id": "abst-claim-1",
        "kind": "abstention",
        "question": "What pasta sauce pairs best with a firewall audit?",
        "source_docs": [],
        "evidence_anchors": [],
        "gold_claims": [],
        "gold_control_ids": [],
        "abstention": True,
        "expected_behavior": "Decline: unrelated to cybersecurity.",
        "difficulty": "easy",
    }
    item.update(overrides)
    return item


def _stage_b_dataset():
    return {
        "schema_version": "1.0",
        "description": "claim-support wiring fixture",
        "items": [_doc_item(), _abst_item()],
    }


def _stage_a_results(engagement_id, stage_items=None):
    if stage_items is None:
        stage_items = [
            {
                "id": "doc-claim-1",
                "kind": "doc_qa",
                "reply": DOC_REPLY,
                "abstained": False,
                "sentence_grounded_fraction": 0.5,
            },
            {
                "id": "abst-claim-1",
                "kind": "abstention",
                "reply": ABST_REPLY,
                "abstained": True,
            },
        ]
    return {
        "engagement_id": engagement_id,
        "backend": {"rag_level": "basic", "model": ""},
        "per_item": stage_items,
    }


def _fake_retrieve(engagement_id, query, **kwargs):
    return {
        "context_block": "",
        "user_hits": [
            {
                "id": "eval_synth_doc.txt::0",
                "text": "zzalphaevalmarker rotation is reviewed quarterly.",
                "metadata": {"filename": "eval_synth_doc.txt"},
                "distance": None,
                "rank": 1,
            }
        ],
        "framework_hits": [
            {
                "id": "NIST_CSF::PR.PS",
                "text": "Patch management software is used to keep endpoints current",
                "metadata": {"framework": "NIST_CSF", "control_id": "PR.PS"},
                "distance": None,
                "rank": 1,
            }
        ],
    }


@pytest.fixture()
def stage_b_engagement():
    store.init_db()
    engagement = store.create_engagement("Fortis Claim Support", "Claim Support Wiring")
    engagement_id = engagement["id"]
    seed_all_frameworks()
    chunks, chunk_meta = ingestion.chunk_text_enriched(
        STAGE_B_DOC_TEXT, "eval_synth_doc.txt", return_meta=True
    )
    vectorstore.add_user_document_chunks(
        engagement_id, "eval_synth_doc.txt", chunks, chunk_meta
    )
    try:
        yield engagement_id
    finally:
        vectorstore.clear_engagement(engagement_id)
        store.delete_engagement(engagement_id)


def _run_stub_stage_b(monkeypatch, engagement_id, stage_items=None, **overrides):
    monkeypatch.setattr(claim_support_runner, "retrieve_for_query", _fake_retrieve)
    overrides.setdefault("dataset", _stage_b_dataset())
    overrides.setdefault("allow_stub", True)
    overrides.setdefault("skip_calibration", True)
    return claim_support_runner.run_claim_support(
        _stage_a_results(engagement_id, stage_items), **overrides
    )


def test_stub_run_refuses_without_allow_flag(stage_b_engagement):
    with pytest.raises(RuntimeError, match="allow_stub"):
        claim_support_runner.run_claim_support(
            _stage_a_results(stage_b_engagement), dataset=_stage_b_dataset()
        )


def test_stub_run_judges_abstention_and_wires_citation_mapping_blocks(
    monkeypatch, tmp_path, stage_b_engagement
):
    result = _run_stub_stage_b(monkeypatch, stage_b_engagement)
    assert config.settings.rag_level == "standard"
    assert result["engagement_id"] == stage_b_engagement
    assert result["stage_a_items"] == 2
    assert result["items_judged"] == 1
    assert result["skipped_abstention"] == 0

    by_id = {entry["id"]: entry for entry in result["per_item"]}
    assert set(by_id) == {"doc-claim-1", "abst-claim-1"}

    doc = by_id["doc-claim-1"]
    assert doc["evidence_hits"] == 2
    assert "skipped" not in doc and "error" not in doc
    assert "extraction_error" in doc
    assert "item_verdict" in doc
    assert doc.get("citation_error") is None and doc.get("mapping_error") is None
    assert doc["citations"]["n_citations"] == 1
    assert doc["citations"]["n_resolved"] == 1
    assert doc["citations"]["n_fabricated"] == 0
    assert doc["citations"]["recall"] == pytest.approx(1.0)
    assert doc["citation_details"][0]["token"] == "PR.PS"
    assert doc["citation_details"][0]["resolved"] is True
    assert doc["mapping"]["pair_prf"]["f1"] == pytest.approx(1.0)
    assert doc["mapping"]["id_prf"]["precision"] == pytest.approx(1.0)
    assert doc["mapping"]["n_fabricated"] == 0
    assert doc["mapping"]["fabricated_pairs"] == []
    assert doc["mapping"]["demotion"]["n"] == 1
    assert doc["mapping"]["demotion"]["tp"] == 1

    abst = by_id["abst-claim-1"]
    assert abst["kind"] == "abstention"
    assert "skipped" not in abst and "error" not in abst
    assert isinstance(abst["abstention_verdict"], str) and abst["abstention_verdict"]
    assert "reason" in abst
    assert abst["stage_a_marker_abstained"] is True
    assert abst["question"].startswith("What pasta sauce")

    assert result["abstention"] is not None
    assert set(result["abstention"]) == ABSTENTION_AGG_KEYS
    assert result["abstention"]["n_traps"] == 1
    assert result["citation_aggregate"]["n_items"] == 1
    assert result["citation_aggregate"]["total_fabricated"] == 0
    assert result["mapping_aggregate"]["n_items"] == 1
    assert result["mapping_aggregate"]["n_known_pairs"] > 0

    claim_support_runner._write_outputs(tmp_path, result, log=lambda message: None)
    with open(tmp_path / "results.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == claim_support_runner._CSV_COLUMNS
    column = {name: index for index, name in enumerate(rows[0])}
    doc_row = next(row for row in rows[1:] if row[0] == "doc-claim-1")
    abst_row = next(row for row in rows[1:] if row[0] == "abst-claim-1")
    assert doc_row[column["n_citations"]] == "1"
    assert doc_row[column["abstention_verdict"]] == ""
    assert doc_row[column["map_id_f1"]] == "1.0"
    assert abst_row[column["n_citations"]] == ""
    assert abst_row[column["abstention_verdict"]] == abst["abstention_verdict"]

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## Aggregates" in summary
    assert "## Citation attribution & framework mapping" in summary
    assert "## Judge-assisted abstention" in summary
    assert "## LIMITATIONS" in summary


def test_abstention_entries_count_toward_limit(monkeypatch, stage_b_engagement):
    stage_items = [
        {
            "id": "abst-claim-1",
            "kind": "abstention",
            "reply": ABST_REPLY,
            "abstained": True,
        },
        {"id": "doc-claim-1", "kind": "doc_qa", "reply": DOC_REPLY},
    ]
    result = _run_stub_stage_b(
        monkeypatch, stage_b_engagement, stage_items=stage_items, limit=1
    )
    assert len(result["per_item"]) == 1
    entry = result["per_item"][0]
    assert entry["kind"] == "abstention"
    assert "abstention_verdict" in entry
    assert result["items_judged"] == 0
    assert result["abstention"]["n_traps"] == 1
    assert result["citation_aggregate"]["n_items"] == 0


def test_resume_reuses_legacy_abstention_skip_verbatim(
    monkeypatch, stage_b_engagement
):
    legacy = {"id": "abst-claim-1", "kind": "abstention", "skipped": "skipped_abstention"}
    result = _run_stub_stage_b(
        monkeypatch, stage_b_engagement, resume=[legacy]
    )
    by_id = {entry["id"]: entry for entry in result["per_item"]}
    assert by_id["abst-claim-1"] == legacy
    assert "abstention_verdict" not in by_id["abst-claim-1"]
    assert result["skipped_abstention"] == 1
    assert result["abstention"] is None
    assert by_id["doc-claim-1"]["citations"]["n_citations"] == 1
