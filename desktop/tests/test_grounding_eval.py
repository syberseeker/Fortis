"""Offline tests for Phase 2 grounding evaluation (Stage A): pure
deterministic scorers in eval/metrics/grounding.py and the stub-backend
wiring of eval/runners/grounding_eval.py. No GPU, no model, no network —
see conftest.py for the environment (hash-stub embeddings, stub LLM,
RERANK_BACKEND=none). The real golden dataset is never read here: every
API is driven with tiny synthetic datasets."""
import json

import pytest

from core import config, ingestion, store, vectorstore
from core.frameworks import seed_all_frameworks
from eval import dataset as eval_ds

try:
    from eval.metrics import grounding
except Exception:
    grounding = None

try:
    from eval.runners import grounding_eval
except Exception:
    grounding_eval = None

requires_metrics = pytest.mark.skipif(
    grounding is None, reason="eval.metrics.grounding is not implemented yet"
)
requires_runner = pytest.mark.skipif(
    grounding_eval is None, reason="eval.runners.grounding_eval is not implemented yet"
)

SYNTH_DOC_TEXT = (
    "Credential rotation for zzalphaevalmarker accounts is reviewed every "
    "quarter under the Fortis rotation policy.\n\n"
    "The zzbetamarker service account must never be shared between humans "
    "and automation."
)

NIST_ANCHOR = "indicators of compromise, and other potentially adverse events"

GOVERNANCE_KEYS = {
    "sentences",
    "grounded_sentences",
    "sentence_grounded_fraction",
    "anchors_in_reply",
    "anchor_hit",
}


def _doc_qa_item(**overrides):
    item = {
        "id": "doc-ground-1",
        "kind": "doc_qa",
        "question": "zzalphaevalmarker",
        "source_docs": ["eval_synth_doc.txt"],
        "evidence_anchors": ["zzalphaevalmarker"],
        "gold_claims": ["zzalphaevalmarker credentials rotate quarterly"],
        "gold_control_ids": [{"framework": "NIST_CSF", "control_id": "PR.AA"}],
        "abstention": False,
        "expected_behavior": None,
        "difficulty": "easy",
    }
    item.update(overrides)
    return item


def _abstention_item(**overrides):
    item = {
        "id": "abst-ground-1",
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


def _grounding_dataset():
    return {
        "schema_version": "1.0",
        "description": "synthetic grounding wiring fixture",
        "items": [
            _doc_qa_item(),
            _doc_qa_item(
                id="fw-ground-decm",
                kind="framework_qa",
                question="monitoring assets for anomalies and adverse events",
                source_docs=[],
                evidence_anchors=[NIST_ANCHOR],
                gold_control_ids=[{"framework": "NIST_CSF", "control_id": "DE.CM"}],
            ),
            _abstention_item(),
        ],
    }


@pytest.fixture()
def eval_engagement():
    store.init_db()
    engagement = store.create_engagement("Fortis Grounding", "Grounding Wiring")
    engagement_id = engagement["id"]
    seed_all_frameworks()
    chunks, chunk_meta = ingestion.chunk_text_enriched(
        SYNTH_DOC_TEXT, "eval_synth_doc.txt", return_meta=True
    )
    vectorstore.add_user_document_chunks(
        engagement_id, "eval_synth_doc.txt", chunks, chunk_meta
    )
    try:
        yield engagement_id
    finally:
        vectorstore.clear_engagement(engagement_id)
        store.delete_engagement(engagement_id)


# ---- extract_citation_tokens -------------------------------------------------

@requires_metrics
def test_extract_citation_tokens_matches_control_id_shapes():
    text = "Mapped to PR.PS, superseded by A01:2021, see CWE-79 and IA-2(1)."
    tokens = grounding.extract_citation_tokens(text)
    assert "PR.PS" in tokens
    assert "A01:2021" in tokens
    assert "CWE-79" in tokens
    assert "IA-2(1)" in tokens


@requires_metrics
def test_extract_citation_tokens_rejects_plain_and_spaced_tokens():
    assert grounding.extract_citation_tokens("TLS 1.0 is deprecated") == []
    assert grounding.extract_citation_tokens("open port 8080 tomorrow") == []
    assert grounding.extract_citation_tokens("one million incidents") == []


# ---- citation_metrics ----------------------------------------------------------

@requires_metrics
def test_citation_metrics_gold_hit_recall_and_unverifiable():
    result = grounding.citation_metrics(
        "Mapped to PR.AA under NIST CSF.", ["PR.AA"], ["PR.AA", "DE.CM"]
    )
    assert result["cited_tokens"] == ["PR.AA"]
    assert result["gold_citation_hits"] == ["PR.AA"]
    assert result["gold_citation_recall"] == 1.0
    assert result["unverifiable_citation_tokens"] == []


@requires_metrics
def test_citation_metrics_handles_dict_shaped_gold_entries():
    result = grounding.citation_metrics(
        "Under NIST CSF 2.0 the category is RS.MA.",
        [{"framework": "NIST_CSF", "control_id": "RS.MA"}],
        ["RS.MA", "DE.CM"],
    )
    assert result["gold_citation_hits"] == ["RS.MA"]
    assert result["gold_citation_recall"] == 1.0
    assert result["unverifiable_citation_tokens"] == []


@requires_metrics
def test_citation_metrics_unverifiable_only_for_unknown_tokens():
    result = grounding.citation_metrics(
        "See XX-99 and also PR.AA.", ["DE.CM"], ["PR.AA"]
    )
    assert result["gold_citation_hits"] == []
    assert result["gold_citation_recall"] == 0.0
    assert result["unverifiable_citation_tokens"] == ["XX-99"]


@requires_metrics
def test_citation_metrics_empty_gold_gives_none_recall_and_gold_is_verifiable():
    empty_gold = grounding.citation_metrics("Nothing cited here.", [], ["PR.AA"])
    assert empty_gold["gold_citation_recall"] is None
    assert empty_gold["gold_citation_hits"] == []

    in_gold_only = grounding.citation_metrics(
        "Covered by DE.CM.", ["DE.CM"], []
    )
    assert in_gold_only["gold_citation_hits"] == ["DE.CM"]
    assert in_gold_only["unverifiable_citation_tokens"] == []


# ---- abstention_detected -------------------------------------------------------

@requires_metrics
def test_abstention_detected_true_on_decline_markers():
    assert grounding.abstention_detected(
        "I cannot verify a patch released yesterday; please upload the advisory"
    ) is True
    assert grounding.abstention_detected(
        "WE DO NOT HAVE RECORDS OF THAT CONNRESSION."
    ) is True


@requires_metrics
def test_abstention_detected_false_on_benign_reply():
    assert grounding.abstention_detected(
        "The firewall permits port 443 with TLS 1.2, which is compliant."
    ) is False


# ---- grounding_metrics -----------------------------------------------------------

EVIDENCE_SENTENCE = (
    "The perimeter firewall permits TCP 443 from the internet zone to the "
    "web tier"
)


@requires_metrics
def test_grounding_metrics_full_copy_grounds_all_sentences():
    result = grounding.grounding_metrics(
        EVIDENCE_SENTENCE + ".", [EVIDENCE_SENTENCE + "."], []
    )
    assert result["sentences"] == 1
    assert result["grounded_sentences"] == 1
    assert result["sentence_grounded_fraction"] == 1.0


@requires_metrics
def test_grounding_metrics_unrelated_sentences_grounded_zero():
    unrelated = "Pasta pairs well with tomato basil sauce. Widgets need oil."
    result = grounding.grounding_metrics(
        unrelated, [EVIDENCE_SENTENCE + "."], []
    )
    assert result["sentences"] == 2
    assert result["grounded_sentences"] == 0
    assert result["sentence_grounded_fraction"] == 0.0

    mixed = EVIDENCE_SENTENCE + ". Pasta pairs well with tomato basil sauce."
    mixed_result = grounding.grounding_metrics(
        mixed, [EVIDENCE_SENTENCE + "."], []
    )
    assert mixed_result["sentences"] == 2
    assert mixed_result["grounded_sentences"] == 1
    assert mixed_result["sentence_grounded_fraction"] == pytest.approx(0.5)


@requires_metrics
def test_grounding_metrics_anchor_hit_and_anchors_in_reply():
    reply = "As stated in zzanchor_one, the control applies to the web tier."
    result = grounding.grounding_metrics(reply, [], ["zzanchor_one", "zzanchor_two"])
    assert result["anchor_hit"] is True
    assert result["anchors_in_reply"] == ["zzanchor_one"]

    miss = grounding.grounding_metrics("No anchors here.", [], ["zzanchor_one"])
    assert miss["anchor_hit"] is False
    assert miss["anchors_in_reply"] == []


# ---- score_item ------------------------------------------------------------------

@requires_metrics
def test_score_item_abstention_item_returns_abstained_and_grounding():
    scores = grounding.score_item(
        _abstention_item(),
        "I cannot verify that; please upload the advisory.",
        [],
    )
    assert scores["abstained"] is True
    assert GOVERNANCE_KEYS <= set(scores)

    answering = grounding.score_item(
        _abstention_item(),
        "Pasta pairs well with tomato basil sauce.",
        [],
    )
    assert answering["abstained"] is False


@requires_metrics
def test_score_item_answerable_item_computes_false_abstention():
    item = _doc_qa_item(gold_control_ids=["CWE-79"])
    hallucinated = grounding.score_item(
        item,
        "The perimeter firewall permits TCP 443 traffic. "
        "The control reference is CWE-999.",
        [EVIDENCE_SENTENCE + "."],
    )
    assert hallucinated["false_abstention"] is False
    assert "CWE-999" in hallucinated["cited_tokens"]
    assert GOVERNANCE_KEYS <= set(hallucinated)

    hedged = grounding.score_item(
        item, "I do not have that information.", [EVIDENCE_SENTENCE + "."]
    )
    assert hedged["false_abstention"] is True


# ---- aggregate ---------------------------------------------------------------------

def _trap(score_id, abstained, fraction=0.0, anchor_hit=False):
    return {
        "id": score_id,
        "abstention": True,
        "abstained": abstained,
        "sentence_grounded_fraction": fraction,
        "anchor_hit": anchor_hit,
    }


def _answerable(
    score_id,
    false_abstention,
    recall,
    fraction,
    anchor_hit,
    unverifiable=(),
):
    return {
        "id": score_id,
        "abstention": False,
        "false_abstention": false_abstention,
        "gold_citation_recall": recall,
        "sentence_grounded_fraction": fraction,
        "anchor_hit": anchor_hit,
        "unverifiable_citation_tokens": list(unverifiable),
    }


@requires_metrics
def test_aggregate_rates_over_handbuilt_scores():
    item_scores = [
        _trap("t1", True),
        _trap("t2", False, fraction=0.5),
        _answerable("a1", False, 1.0, 0.5, True, ["XX-99"]),
        _answerable("a2", True, None, 1.0, False),
    ]
    agg = grounding.aggregate(item_scores)
    assert agg["n_items"] == 4
    assert agg["abstention_rate_on_traps"] == pytest.approx(0.5)
    assert agg["false_abstention_rate_on_answerable"] == pytest.approx(0.5)
    assert agg["mean_gold_citation_recall"] == pytest.approx(1.0)
    assert agg["mean_sentence_grounded_fraction"] == pytest.approx(
        (0.0 + 0.5 + 0.5 + 1.0) / 4
    )
    assert agg["anchor_hit_rate"] == pytest.approx(0.25)
    assert agg["items_with_unverifiable_citations"] == 1


@requires_metrics
def test_aggregate_includes_zero_recall_items_in_gold_mean():
    item_scores = [
        _answerable("a1", False, 1.0, 1.0, True),
        _answerable("a2", False, 0.0, 0.0, False),
    ]
    agg = grounding.aggregate(item_scores)
    assert agg["n_items"] == 2
    assert agg["mean_gold_citation_recall"] == pytest.approx(0.5)
    assert agg["mean_sentence_grounded_fraction"] == pytest.approx(0.5)
    assert agg["anchor_hit_rate"] == pytest.approx(0.5)
    assert agg["false_abstention_rate_on_answerable"] == 0.0


@requires_metrics
def test_aggregate_is_none_safe_when_subsets_empty():
    only_answerable = grounding.aggregate(
        [_answerable("a1", False, 0.5, 1.0, True)]
    )
    assert only_answerable["n_items"] == 1
    assert only_answerable["abstention_rate_on_traps"] is None
    assert only_answerable["false_abstention_rate_on_answerable"] == 0.0

    only_traps = grounding.aggregate([_trap("t1", True)])
    assert only_traps["n_items"] == 1
    assert only_traps["false_abstention_rate_on_answerable"] is None
    assert only_traps["mean_gold_citation_recall"] is None

    empty = grounding.aggregate([])
    assert empty["n_items"] == 0
    assert empty["abstention_rate_on_traps"] is None
    assert empty["false_abstention_rate_on_answerable"] is None
    assert empty["mean_gold_citation_recall"] is None
    assert empty["mean_sentence_grounded_fraction"] is None
    assert empty["anchor_hit_rate"] is None
    assert empty["items_with_unverifiable_citations"] == 0


# ---- runner wiring (stub backend, offline) ------------------------------------------

@requires_runner
def test_run_grounding_eval_refuses_stub_without_allow_flag():
    with pytest.raises(RuntimeError, match="allow_stub"):
        grounding_eval.run_grounding_eval(_grounding_dataset())


@requires_runner
def test_run_grounding_eval_wiring_restores_rag_level(eval_engagement):
    dataset = _grounding_dataset()
    for item in dataset["items"]:
        assert eval_ds.validate_item(item) == []

    result = grounding_eval.run_grounding_eval(
        dataset,
        engagement_id=eval_engagement,
        rag_level="basic",
        allow_stub=True,
    )
    assert config.settings.rag_level == "standard"
    assert result["engagement_id"] == eval_engagement
    assert result["dataset_items"] == 3

    per_item = result["per_item"]
    assert len(per_item) == 3
    for entry in per_item:
        assert "error" not in entry, entry.get("error")
        assert "id" in entry
        assert "sentence_grounded_fraction" in entry
        assert "anchor_hit" in entry
        assert isinstance(entry["provenance"], list)
        assert all(isinstance(hit_id, str) for hit_id in entry["provenance"])

    by_id = {entry["id"]: entry for entry in per_item}
    assert by_id["abst-ground-1"]["abstained"] is False
    assert by_id["doc-ground-1"]["false_abstention"] is False
    assert by_id["doc-ground-1"]["provenance"]

    aggregate = result["aggregate"]
    assert aggregate["n_items"] == 3
    assert set(aggregate) >= {
        "abstention_rate_on_traps",
        "false_abstention_rate_on_answerable",
        "mean_gold_citation_recall",
        "mean_sentence_grounded_fraction",
        "anchor_hit_rate",
        "items_with_unverifiable_citations",
    }


@requires_runner
def test_run_grounding_eval_limit_runs_single_item(eval_engagement):
    result = grounding_eval.run_grounding_eval(
        _grounding_dataset(),
        engagement_id=eval_engagement,
        allow_stub=True,
        limit=1,
    )
    assert len(result["per_item"]) == 1
    assert result["aggregate"]["n_items"] == 1
    assert config.settings.rag_level == "standard"


@requires_runner
def test_cli_help_lists_flags_without_k():
    flags = []
    with pytest.raises(SystemExit) as exit_info:
        grounding_eval.main(["--help"])
    assert exit_info.value.code == 0


@requires_runner
def test_cli_writes_results_and_summary_with_limitations(tmp_path, monkeypatch):
    dataset = {
        "schema_version": "1.0",
        "description": "cli wiring fixture",
        "items": [
            _doc_qa_item(source_docs=[], evidence_anchors=[]),
            _abstention_item(),
        ],
    }
    monkeypatch.setattr(
        eval_ds, "load_golden_dataset", lambda *args, **kwargs: dataset
    )
    if hasattr(grounding_eval, "load_golden_dataset"):
        monkeypatch.setattr(
            grounding_eval,
            "load_golden_dataset",
            lambda *args, **kwargs: dataset,
        )

    code = grounding_eval.main(
        ["--allow-stub", "--limit", "1", "--out", str(tmp_path)]
    )
    assert code == 0
    results_json = tmp_path / "results.json"
    results_csv = tmp_path / "results.csv"
    summary_md = tmp_path / "summary.md"
    assert results_json.exists()
    assert results_csv.exists()
    assert summary_md.exists()
    written = json.loads(results_json.read_text(encoding="utf-8"))
    assert isinstance(written, dict)
    assert results_csv.read_text(encoding="utf-8").strip()
    assert "LIMITATIONS" in summary_md.read_text(encoding="utf-8")
