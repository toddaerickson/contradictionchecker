from benchmarks.definition_eval.harness import _metrics
from benchmarks.definition_eval.mine_pairs import build_candidates
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
