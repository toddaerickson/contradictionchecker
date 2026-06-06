from types import SimpleNamespace

import pytest

from benchmarks.definition_eval.harness import _metrics
from benchmarks.definition_eval.mine_pairs import (
    _is_cross_reference,
    build_candidates,
    resolve_corpus_id,
)
from consistency_checker.extract.schema import Assertion


def _defn(doc, term, text):
    return Assertion.build(
        doc, f'"{term}" means {text}.', kind="definition", term=term, definition_text=text
    )


def test_build_candidates_pairs_same_canonical_term():
    defs = [
        _defn("d1", "Board", "the board of directors"),
        _defn("d2", "board", "the board of directors"),  # identical text, canonical-equal term
        _defn("d3", "Board", "the supervisory board only"),  # divergent candidate
    ]
    cands = build_candidates(defs, max_pairs=100)
    assert len(cands) == 3  # 3 unordered pairs over 3 same-canonical-term defs
    cats = {c["category"] for c in cands}
    assert "identical" in cats and "review" in cats
    for c in cands:
        assert c["label"] == ""
        assert c["term"] and c["def_a"] and c["def_b"]
        assert c["doc_a"] and c["doc_b"]


def test_build_candidates_skips_singletons():
    defs = [_defn("d1", "Quorum", "a majority"), _defn("d2", "Notice", "written notice")]
    assert build_candidates(defs, max_pairs=100) == []  # no term has >= 2 defs


def test_build_candidates_caps_and_prioritizes_review():
    # one term, 3 distinct texts -> 3 pairs all "review"; plus an identical pair on another term.
    defs = [
        _defn("a", "Term", "alpha"),
        _defn("b", "Term", "beta"),
        _defn("c", "Term", "gamma"),
        _defn("d", "Other", "same"),
        _defn("e", "Other", "same"),
    ]
    cands = build_candidates(defs, max_pairs=2)
    assert len(cands) == 2
    assert all(c["category"] == "review" for c in cands)  # review pairs sorted ahead of identical


def test_build_candidates_pair_ids_unique_and_stable():
    defs = [
        _defn("a", "Term", "alpha"),
        _defn("b", "Term", "beta"),
        _defn("c", "Term", "gamma"),
    ]
    first = build_candidates(defs, max_pairs=100)
    ids = [c["pair_id"] for c in first]
    assert len(ids) == len(set(ids))  # unique
    # stable across re-runs (content hash, not run-order)
    second = build_candidates(defs, max_pairs=100)
    assert [c["pair_id"] for c in second] == ids


def test_is_cross_reference():
    assert _is_cross_reference("has the meaning assigned to such term in Section 2.11(a)")
    assert _is_cross_reference("the meaning specified in Section 2.11")
    assert _is_cross_reference("As defined in the Credit Agreement")
    assert _is_cross_reference("shall have the meaning given to it below")
    # real definitions are NOT cross-references
    assert not _is_cross_reference("the board of directors of the Corporation")
    assert not _is_cross_reference("a majority of the members then in office")
    assert not _is_cross_reference(
        "the intended commercial meaning of the parties"
    )  # 'meaning' mid-text


def test_build_candidates_drops_cross_reference_pairs():
    defs = [
        _defn("d1", "Acceptable Discount", "has the meaning assigned in Section 2.11"),  # xref
        _defn(
            "d2", "Acceptable Discount", "the largest of the Offered Discounts specified"
        ),  # real
        _defn("d3", "Acceptable Discount", "the smallest qualifying discount offered"),  # real
    ]
    cands = build_candidates(defs, max_pairs=100)
    # pairs touching the xref def are dropped; only the (real, real) pair survives
    assert len(cands) == 1
    assert "has the meaning" not in cands[0]["def_a"]
    assert "has the meaning" not in cands[0]["def_b"]


def test_build_candidates_prioritizes_cross_document():
    defs = [
        _defn("base", "Lender", "a bank that lends"),  # base doc
        _defn(
            "amend", "Lender", "a bank or fund that lends"
        ),  # amendment doc -> cross-doc with base
        _defn("base", "Lender", "a bank or fund that lends"),  # same-doc as base (different text)
    ]
    cands = build_candidates(defs, max_pairs=100)
    # the first candidate must be a cross-document pair (doc_a != doc_b)
    assert cands[0]["doc_a"] != cands[0]["doc_b"]


def _pred(label, predicted):
    return {"label": label, "predicted": predicted}


def test_metrics_perfect():
    preds = [_pred("divergent", "divergent"), _pred("consistent", "consistent")]
    m = _metrics(preds)
    assert m["confusion"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 1}
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_metrics_mixed():
    preds = [
        _pred("divergent", "divergent"),
        _pred("divergent", "consistent"),
        _pred("consistent", "divergent"),
        _pred("consistent", "consistent"),
    ]
    m = _metrics(preds)
    assert m["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5


def test_metrics_no_positive_predictions_is_none():
    preds = [_pred("consistent", "consistent")]
    m = _metrics(preds)
    assert m["precision"] is None
    assert m["recall"] is None
    assert m["f1"] is None


def test_metrics_all_divergent_perfect():
    preds = [_pred("divergent", "divergent"), _pred("divergent", "divergent")]
    m = _metrics(preds)
    assert m["confusion"] == {"tp": 2, "fp": 0, "fn": 0, "tn": 0}
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_metrics_all_positives_wrong_is_zero_not_none():
    # tp=0 but predictions exist in both classes → F1 must be 0.0, not None.
    preds = [_pred("divergent", "consistent"), _pred("consistent", "divergent")]
    m = _metrics(preds)
    assert m["confusion"] == {"tp": 0, "fp": 1, "fn": 1, "tn": 0}
    assert m["precision"] == 0.0  # tp/(tp+fp) = 0/1
    assert m["recall"] == 0.0  # tp/(tp+fn) = 0/1
    assert m["f1"] == 0.0


def _corpus(cid, name):
    return SimpleNamespace(corpus_id=cid, corpus_name=name)


def test_resolve_corpus_id_by_name_id_none_and_error():
    corpora = [_corpus("uuid-abc", "atkins"), _corpus("uuid-def", "fcs-call-report")]
    assert resolve_corpus_id(corpora, "atkins") == "uuid-abc"  # by name
    assert resolve_corpus_id(corpora, "uuid-def") == "uuid-def"  # by raw id
    assert resolve_corpus_id(corpora, None) is None  # all corpora
    with pytest.raises(ValueError, match="unknown corpus"):
        resolve_corpus_id(corpora, "nope")
