"""Offline tests for the Phase 3 citation-attribution and mapping-accuracy
metrics: exact P/R/F1 values including None-safe zero denominators, sentence
tokenization that must never split "PR.PS", attribution of a citation to the
retrieved framework hit that best matches its sentence, fabricated token/pair
detection, and the demotion confusion matrix — first against an injected
validator, then against the real seeded framework corpus read straight from
core/frameworks/*.json (the five files whose stem equals the seeded corpus
framework id) with the chromadb access point faked. No model, no GPU, no network.
"""
import json
import os

import pytest

from core import report_service
from core import framework_aliases
from eval.metrics import citations as cite_metrics
from eval.metrics import mapping_accuracy as map_metrics

_FRAMEWORK_DIR = os.path.join(
    os.path.dirname(os.path.abspath(framework_aliases.__file__)), "frameworks"
)

_SEEDED_FILES = {
    "nist_csf.json": "NIST_CSF",
    "cis_controls.json": "CIS_CONTROLS",
    "owasp_top10.json": "OWASP_TOP10",
    "iso_27001.json": "ISO_27001",
    "mitre_attack.json": "MITRE_ATTACK",
}


def _hit(hit_id, framework, control_id, text, rank=1):
    return {
        "id": hit_id,
        "text": text,
        "metadata": {"framework": framework, "control_id": control_id},
        "distance": 0.1,
        "rank": rank,
    }


ATTRIBUTION_HITS = [
    _hit("h_low", "NIST_CSF", "PR.PS", "The binder records asset inventory quarterly", 1),
    _hit("h_best", "NIST_CSF", "PR.PS", "Least privilege access enforcement for remote desktop", 2),
    _hit("h_aa", "NIST_CSF", "PR.AA", "Asset inventory is maintained", 3),
]
ATTRIBUTION_REPLY = (
    "Enforce least privilege for remote access via PR.PS. "
    "Rotate shared account passwords quarterly per PR.AA. Cite XX.FAKE here."
)


# ---- sentence handling and citation extraction -------------------------------

def test_norm_casefolds_and_collapses_whitespace():
    assert cite_metrics._norm("  Foo\tBAR\n\nbaz  ") == "foo bar baz"
    assert cite_metrics._norm(None) == ""


def test_split_sentences_never_splits_a_control_id():
    assert cite_metrics.split_sentences("See PR.PS. Also PR.PS again? Next\nLine here.") == [
        "See PR.PS",
        "Also PR.PS again",
        "Next",
        "Line here",
    ]
    assert cite_metrics.split_sentences("") == []
    assert cite_metrics.split_sentences("...!!\n") == []


def test_extract_citations_pairs_tokens_with_their_sentences():
    citations = cite_metrics.extract_citations(
        "Use NIST CSF PR.PS to harden asset management. "
        "PR.PS also covers least privilege. See PR.PS."
    )
    assert [c["token"] for c in citations] == ["PR.PS", "PR.PS", "PR.PS"]
    assert citations[0]["sentence"] == "Use NIST CSF PR.PS to harden asset management"
    assert citations[-1]["sentence"] == "See PR.PS"


def test_extract_citations_dedupes_pairs_and_drops_bare_sentences():
    assert cite_metrics.extract_citations("Cite PR.PS and PR.PS again.") == [
        {"token": "PR.PS", "sentence": "Cite PR.PS and PR.PS again"}
    ]
    assert cite_metrics.extract_citations("Nothing cited here.") == []
    assert cite_metrics.extract_citations(None) == []


# ---- sentence_overlap ---------------------------------------------------------

def test_sentence_overlap_is_sentence_side_containment():
    assert cite_metrics.sentence_overlap(
        "Enforce least privilege for remote access",
        "Least privilege access enforcement for remote desktop",
    ) == 0.8


def test_sentence_overlap_boundary_cases():
    assert cite_metrics.sentence_overlap("Least privilege access", "Least privilege access enforcement") == 1.0
    assert cite_metrics.sentence_overlap("Enforce least privilege", "Binder records asset inventory") == 0.0
    assert cite_metrics.sentence_overlap("The and for which with", "Least privilege access") == 0.0
    assert cite_metrics.sentence_overlap("", "Least privilege access") == 0.0
    assert cite_metrics.sentence_overlap("Least privilege access", "") == 0.0


# ---- citation precision / recall / f1 ----------------------------------------

def test_citation_precision_recall_f1_exact_values():
    assert cite_metrics.citation_precision_recall_f1(
        ["PR.AA", " CWE-79 ", "pr.aa", "ZZ.FAKE"], ["PR.AA", "DE.CM"]
    ) == {
        "precision": 0.3333,
        "recall": 0.5,
        "f1": 0.4,
        "n_emitted": 3,
        "n_gold": 2,
        "n_matched": 1,
    }


def test_citation_precision_recall_f1_accepts_gold_control_id_dicts():
    assert cite_metrics.citation_precision_recall_f1(
        ["PR.AA"], [{"framework": "NIST_CSF", "control_id": "PR.AA"}]
    ) == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_emitted": 1, "n_gold": 1, "n_matched": 1}


def test_citation_precision_recall_f1_none_denominators():
    assert cite_metrics.citation_precision_recall_f1([], []) == {
        "precision": None,
        "recall": None,
        "f1": None,
        "n_emitted": 0,
        "n_gold": 0,
        "n_matched": 0,
    }
    assert cite_metrics.citation_precision_recall_f1([], ["PR.AA"]) == {
        "precision": None, "recall": 0.0, "f1": 0.0, "n_emitted": 0, "n_gold": 1, "n_matched": 0,
    }
    assert cite_metrics.citation_precision_recall_f1(["PR.AA"], []) == {
        "precision": 0.0, "recall": None, "f1": 0.0, "n_emitted": 1, "n_gold": 0, "n_matched": 0,
    }


def test_fabricated_tokens_uppercased_sorted_deduped():
    assert cite_metrics.fabricated_tokens(
        ["PR.AA", "CWE-79", "zz.fake", "PR.AA"], ["pr.aa", "CWE-79"]
    ) == ["ZZ.FAKE"]
    assert cite_metrics.fabricated_tokens(["BB-2", "AA-1"], []) == ["AA-1", "BB-2"]
    assert cite_metrics.fabricated_tokens([], ["PR.AA"]) == []


# ---- attribution against retrieved framework hits ----------------------------

def test_attribute_citations_picks_best_overlapping_hit_per_control_id():
    attributed = cite_metrics.attribute_citations(
        cite_metrics.extract_citations(ATTRIBUTION_REPLY), ATTRIBUTION_HITS
    )
    assert [(c["token"], c["resolved"], c["match_hit_id"], c["overlap"], c["attributed"]) for c in attributed] == [
        ("PR.PS", True, "h_best", 0.8, True),
        ("PR.AA", True, "h_aa", 0.0, False),
        ("XX.FAKE", False, None, 0.0, False),
    ]


def test_attribute_citations_threshold_controls_attribution():
    citations = cite_metrics.extract_citations(ATTRIBUTION_REPLY)
    strict = cite_metrics.attribute_citations(citations, ATTRIBUTION_HITS, overlap_threshold=0.9)
    assert [c["attributed"] for c in strict] == [False, False, False]
    loose = cite_metrics.attribute_citations(citations, ATTRIBUTION_HITS, overlap_threshold=0.05)
    assert [c["attributed"] for c in loose] == [True, False, False]
    assert all(c["resolved"] == (c["token"] != "XX.FAKE") for c in loose)


def test_attribute_citations_ignores_hits_without_metadata_control_id():
    attributed = cite_metrics.attribute_citations(
        [{"token": "PR.PS", "sentence": "Least privilege access"}],
        [{"id": "h", "text": "Least privilege access", "metadata": {}, "distance": 0.1, "rank": 1}],
    )
    assert attributed == [{
        "token": "PR.PS",
        "sentence": "Least privilege access",
        "resolved": False,
        "match_hit_id": None,
        "overlap": 0.0,
        "attributed": False,
    }]


# ---- per-item citation summary and aggregation -------------------------------

def _attribution_rows():
    return cite_metrics.attribute_citations(
        cite_metrics.extract_citations(ATTRIBUTION_REPLY), ATTRIBUTION_HITS
    )


def test_citation_summary_merges_attribution_and_gold_prf():
    assert cite_metrics.citation_summary(
        _attribution_rows(), ["PR.PS", "PR.AA", "XX.FAKE"], ["PR.PS", "DE.CM"], ["PR.PS", "PR.AA"]
    ) == {
        "n_citations": 3,
        "n_resolved": 2,
        "n_attributed": 1,
        "attribution_rate": 0.5,
        "precision": 0.3333,
        "recall": 0.5,
        "f1": 0.4,
        "n_fabricated": 1,
        "fabricated_tokens": ["XX.FAKE"],
    }


def test_citation_summary_without_citations_is_all_none():
    assert cite_metrics.citation_summary([], [], [], []) == {
        "n_citations": 0,
        "n_resolved": 0,
        "n_attributed": 0,
        "attribution_rate": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "n_fabricated": 0,
        "fabricated_tokens": [],
    }


def test_citation_summary_caps_fabricated_token_list():
    emitted = [f"ZZ{i:02d}.FAKE" for i in range(14)]
    summary = cite_metrics.citation_summary([], emitted, [], [])
    assert summary["n_fabricated"] == 14
    assert len(summary["fabricated_tokens"]) == 10


def test_aggregate_items_means_and_fabricated_totals():
    agg = cite_metrics.aggregate_items([
        cite_metrics.citation_summary(
            _attribution_rows(), ["PR.PS", "PR.AA", "XX.FAKE"], ["PR.PS", "DE.CM"], ["PR.PS", "PR.AA"]
        ),
        cite_metrics.citation_summary([], [], [], []),
    ])
    assert agg == {
        "n_items": 2,
        "n_items_with_citations": 1,
        "mean_citations_per_item": 1.5,
        "mean_attribution_rate": 0.5,
        "mean_precision": 0.3333,
        "mean_recall": 0.5,
        "mean_f1": 0.4,
        "total_fabricated": 1,
        "fabricated_token_counts": {"XX.FAKE": 1},
    }
    assert cite_metrics.aggregate_items([]) == {
        "n_items": 0,
        "n_items_with_citations": 0,
        "mean_citations_per_item": 0.0,
        "mean_attribution_rate": None,
        "mean_precision": None,
        "mean_recall": None,
        "mean_f1": None,
        "total_fabricated": 0,
        "fabricated_token_counts": {},
    }


def test_aggregate_items_fabricated_counts_sorted_by_count_then_token():
    agg = cite_metrics.aggregate_items([
        {"n_citations": 1, "attribution_rate": 1.0, "precision": 1.0, "recall": 1.0,
         "f1": 1.0, "n_fabricated": 3, "fabricated_tokens": ["ZZ", "AA", "MM"]},
        {"n_citations": 0, "attribution_rate": None, "precision": None, "recall": None,
         "f1": None, "n_fabricated": 1, "fabricated_tokens": ["AA"]},
    ])
    assert list(agg["fabricated_token_counts"]) == ["AA", "MM", "ZZ"]
    assert agg["fabricated_token_counts"] == {"AA": 2, "MM": 1, "ZZ": 1}
    assert agg["total_fabricated"] == 4


# ---- framework pair extraction ------------------------------------------------

def test_extract_framework_pairs_binds_framework_to_sentence_token():
    assert map_metrics.extract_framework_pairs("Use NIST CSF PR.PS to harden.") == [
        {"framework": "NIST_CSF", "control_id": "PR.PS"}
    ]


def test_extract_framework_pairs_leaves_frameworkless_ids_unnamed():
    assert map_metrics.extract_framework_pairs("CWE-79 is relevant") == [
        {"framework": "", "control_id": "CWE-79"}
    ]


def test_extract_framework_pairs_recognizes_owasp_alias_spellings():
    assert map_metrics.extract_framework_pairs("Align the app with OWASP A01:2025 today.") == [
        {"framework": "OWASP_TOP10", "control_id": "A01:2025"}
    ]
    assert map_metrics.extract_framework_pairs("Per the OWASP Top 10, fix A01:2025.") == [
        {"framework": "OWASP_TOP10", "control_id": "A01:2025"}
    ]


def test_extract_framework_pairs_canonicalizes_to_seeded_corpus_ids():
    pairs = map_metrics.extract_framework_pairs(
        "NIST SP 800-53 IA-2(1) applies. SOC 2 maps to CIS-4. "
        "HIPAA Security Rule needs PR.AA. The NIST Cybersecurity Framework covers DE.CM."
    )
    assert pairs == [
        {"framework": "NIST_800-53", "control_id": "IA-2(1)"},
        {"framework": "SOC_2", "control_id": "CIS-4"},
        {"framework": "HIPAA", "control_id": "PR.AA"},
        {"framework": "NIST_CSF", "control_id": "DE.CM"},
    ]


def test_extract_framework_pairs_cross_product_and_dedupe():
    pairs = map_metrics.extract_framework_pairs("Map PR.AA to NIST CSF and CIS Controls CIS-4.")
    assert {(p["framework"], p["control_id"]) for p in pairs} == {
        ("NIST_CSF", "PR.AA"),
        ("NIST_CSF", "CIS-4"),
        ("CIS_CONTROLS", "PR.AA"),
        ("CIS_CONTROLS", "CIS-4"),
    }
    assert map_metrics.extract_framework_pairs("NIST CSF PR.PS. NIST CSF PR.PS again.") == [
        {"framework": "NIST_CSF", "control_id": "PR.PS"}
    ]
    assert map_metrics.extract_framework_pairs("No ids at all here.") == []


# ---- mapping precision / recall / f1 ------------------------------------------

EMITTED_PAIRS = [
    {"framework": "NIST_CSF", "control_id": "PR.AA"},
    {"framework": "", "control_id": "CWE-79"},
    {"framework": "OWASP_TOP10", "control_id": "A01:2025"},
]
GOLD_PAIRS = [
    {"framework": "NIST_CSF", "control_id": "PR.AA"},
    {"framework": "NIST_CSF", "control_id": "DE.CM"},
]


def test_mapping_prf_exact_values_at_both_levels():
    assert map_metrics.mapping_prf(EMITTED_PAIRS, GOLD_PAIRS) == {
        "pair": {
            "precision": 0.5, "recall": 0.5, "f1": 0.5,
            "n_emitted": 2, "n_gold": 2, "n_matched": 1,
        },
        "id": {
            "precision": 0.3333, "recall": 0.5, "f1": 0.4,
            "n_emitted": 3, "n_gold": 2, "n_matched": 1,
        },
    }


def test_mapping_prf_canonicalizes_gold_frameworks():
    result = map_metrics.mapping_prf(
        [{"framework": "NIST_CSF", "control_id": "PR.AA"}],
        [{"framework": "NIST CSF", "control_id": "pr.aa"}],
    )
    assert result["pair"]["n_matched"] == 1
    assert result["pair"]["f1"] == 1.0
    assert result["id"]["f1"] == 1.0


def test_mapping_prf_empty_inputs():
    assert map_metrics.mapping_prf([], []) == {
        "pair": {"precision": None, "recall": None, "f1": None, "n_emitted": 0, "n_gold": 0, "n_matched": 0},
        "id": {"precision": None, "recall": None, "f1": None, "n_emitted": 0, "n_gold": 0, "n_matched": 0},
    }


def test_fabricated_pairs_skips_known_and_frameworkless():
    emitted = EMITTED_PAIRS + [
        {"framework": "NIST_CSF", "control_id": "xx.fake"},
        {"framework": "nist_csf", "control_id": "PR.AA"},
    ]
    assert map_metrics.fabricated_pairs(emitted, {("NIST_CSF", "PR.AA")}) == [
        {"framework": "NIST_CSF", "control_id": "XX.FAKE"},
        {"framework": "OWASP_TOP10", "control_id": "A01:2025"},
    ]
    assert map_metrics.fabricated_pairs(EMITTED_PAIRS, set()) == [
        {"framework": "NIST_CSF", "control_id": "PR.AA"},
        {"framework": "OWASP_TOP10", "control_id": "A01:2025"},
    ]


# ---- demotion confusion matrix ------------------------------------------------

CONFUSION_PAIRS = [
    {"framework": "NIST_CSF", "control_id": "PR.AA"},
    {"framework": "CIS_CONTROLS", "control_id": "CIS-4"},
    {"framework": "NIST_CSF", "control_id": "XX.FAKE"},
    {"framework": "OWASP_TOP10", "control_id": "A99:2025"},
]
CONFUSION_KNOWN = {("NIST_CSF", "PR.AA"), ("CIS_CONTROLS", "CIS-4")}


def test_demotion_confusion_with_injected_validator():
    def keeps_true_plus_one_false(pairs):
        return [
            ("NIST_CSF", "PR.AA"),
            ("CIS_CONTROLS", "CIS-4"),
            ("NIST_CSF", "XX.FAKE"),
        ]

    assert map_metrics.demotion_confusion(CONFUSION_PAIRS, CONFUSION_KNOWN, keeps_true_plus_one_false) == {
        "tp": 2, "fp": 1, "tn": 1, "fn": 0, "n": 4,
        "accuracy": 0.75, "demotion_precision": 0.6667, "demotion_recall": 1.0,
    }


def test_demotion_confusion_when_everything_is_demoted():
    assert map_metrics.demotion_confusion(CONFUSION_PAIRS, CONFUSION_KNOWN, lambda pairs: []) == {
        "tp": 0, "fp": 0, "tn": 2, "fn": 2, "n": 4,
        "accuracy": 0.5, "demotion_precision": None, "demotion_recall": 0.0,
    }


def test_demotion_confusion_without_pairs_is_all_none():
    assert map_metrics.demotion_confusion([], CONFUSION_KNOWN, lambda pairs: []) == {
        "tp": 0, "fp": 0, "tn": 0, "fn": 0, "n": 0,
        "accuracy": None, "demotion_precision": None, "demotion_recall": None,
    }


def test_aggregate_items_mapping_means_and_summed_confusion():
    known = CONFUSION_KNOWN | {("OWASP_TOP10", "A01:2025")}
    agg = map_metrics.aggregate_items([
        {
            "mapping_prf": map_metrics.mapping_prf(EMITTED_PAIRS, GOLD_PAIRS),
            "fabricated_pairs": [{"framework": "NIST_CSF", "control_id": "XX.FAKE"}],
            "demotion_confusion": map_metrics.demotion_confusion(
                CONFUSION_PAIRS, CONFUSION_KNOWN,
                lambda pairs: [("NIST_CSF", "PR.AA")],
            ),
            "n_emitted_ids": 3,
        },
        {
            "pair": {"precision": None, "recall": 1.0, "f1": None, "n_emitted": 0, "n_gold": 2, "n_matched": 0},
            "id": {"precision": None, "recall": 1.0, "f1": None, "n_emitted": 0, "n_gold": 2, "n_matched": 0},
            "n_fabricated_pairs": 2,
            "n_emitted_ids": 1,
        },
    ], known)
    assert agg["n_items"] == 2
    assert agg["n_items_demotion_scored"] == 1
    assert agg["mean_pair_precision"] == 0.5
    assert agg["mean_pair_recall"] == 0.75
    assert agg["mean_pair_f1"] == 0.5
    assert agg["mean_id_precision"] == 0.3333
    assert agg["mean_id_recall"] == 0.75
    assert agg["mean_id_f1"] == 0.4
    assert agg["mean_mapping_ids_per_item"] == 2.0
    assert agg["total_fabricated_pairs"] == 3
    assert agg["demotion_confusion"] == {
        "tp": 1, "fp": 0, "tn": 2, "fn": 1, "n": 4,
        "accuracy": 0.75, "demotion_precision": 1.0, "demotion_recall": 0.5,
    }
    assert agg["n_known_pairs"] == 3


def test_aggregate_items_accepts_the_runner_mapping_block_shape():
    known = CONFUSION_KNOWN | {("OWASP_TOP10", "A01:2025")}
    emitted = EMITTED_PAIRS + [{"framework": "NIST_CSF", "control_id": "XX.FAKE"}]
    prf = map_metrics.mapping_prf(emitted, GOLD_PAIRS)
    fabricated = map_metrics.fabricated_pairs(emitted, known)
    block = {
        "pair_prf": prf["pair"],
        "id_prf": prf["id"],
        "n_fabricated": len(fabricated),
        "fabricated_pairs": fabricated[:10],
        "demotion": map_metrics.demotion_confusion(
            CONFUSION_PAIRS, CONFUSION_KNOWN, lambda pairs: [("NIST_CSF", "PR.AA")]
        ),
    }
    agg = map_metrics.aggregate_items([block], known)
    assert fabricated == [{"framework": "NIST_CSF", "control_id": "XX.FAKE"}]
    assert agg["n_items"] == 1
    assert agg["mean_pair_f1"] == prf["pair"]["f1"]
    assert agg["mean_id_precision"] == prf["id"]["precision"]
    assert agg["mean_mapping_ids_per_item"] == 4
    assert agg["total_fabricated_pairs"] == 1
    assert agg["demotion_confusion"]["n"] == 4
    assert agg["n_known_pairs"] == 3


# ---- real seeded corpus, real demotion logic -----------------------------------

@pytest.fixture
def real_corpus(monkeypatch):
    known = set()
    metadatas = []
    for filename, framework in _SEEDED_FILES.items():
        path = os.path.join(_FRAMEWORK_DIR, filename)
        with open(path, encoding="utf-8") as handle:
            entries = json.load(handle)
        for entry in entries:
            control_id = str(entry.get("id") or "").strip()
            if not control_id:
                continue
            known.add((framework, control_id.upper()))
            metadatas.append({"framework": framework, "control_id": control_id})
    assert len(known) > 50

    class _FakeCollection:
        def get(self, **kwargs):
            return {"metadatas": list(metadatas)}

    report_service.invalidate_known_framework_pairs_cache()
    monkeypatch.setattr(report_service.vectorstore, "framework_collection", lambda: _FakeCollection())
    yield known
    report_service.invalidate_known_framework_pairs_cache()


def test_demotion_confusion_default_validator_scores_real_corpus(real_corpus):
    pairs = [
        {"framework": "NIST_CSF", "control_id": "PR.AA"},
        {"framework": "CIS_CONTROLS", "control_id": "CIS-4"},
        {"framework": "OWASP_TOP10", "control_id": "A01:2025"},
        {"framework": "NIST_CSF", "control_id": "XX.FAKE"},
    ]
    for pair in pairs[:3]:
        assert (pair["framework"], pair["control_id"]) in real_corpus

    assert map_metrics.demotion_confusion(pairs, real_corpus) == {
        "tp": 3, "fp": 0, "tn": 1, "fn": 0, "n": 4,
        "accuracy": 1.0, "demotion_precision": 1.0, "demotion_recall": 1.0,
    }
    assert map_metrics.fabricated_pairs(pairs, real_corpus) == [
        {"framework": "NIST_CSF", "control_id": "XX.FAKE"}
    ]


def test_real_reply_maps_and_scores_against_real_corpus(real_corpus):
    reply = (
        "Harden remote access per NIST CSF PR.AA. Fix CIS-4 in the CIS Controls. "
        "Broken access maps to OWASP A01:2025. Ignore XX.FAKE entirely."
    )
    pairs = map_metrics.extract_framework_pairs(reply)
    assert {(p["framework"], p["control_id"]) for p in pairs} == {
        ("NIST_CSF", "PR.AA"),
        ("CIS_CONTROLS", "CIS-4"),
        ("OWASP_TOP10", "A01:2025"),
        ("", "XX.FAKE"),
    }
    result = map_metrics.demotion_confusion(pairs, real_corpus)
    assert result["accuracy"] == 1.0
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (3, 0, 1, 0)
    assert map_metrics.fabricated_pairs(pairs, real_corpus) == []
