"""Unit tests for the _filter_pairs_by_corpus helper in pipeline.py."""

from __future__ import annotations


def test_filter_pairs_by_corpus_keeps_only_intra_corpus_pairs() -> None:
    """With corpus_assertion_ids = {a1, a2}, a candidate pair (a1, b1) where
    b1 is outside the set must be dropped before reaching the judge."""
    from consistency_checker.pipeline import _filter_pairs_by_corpus

    corpus_ids = {"a1", "a2"}
    pairs = [("a1", "a2"), ("a1", "b1"), ("b1", "b2"), ("a2", "a1")]
    kept, dropped = _filter_pairs_by_corpus(pairs, corpus_ids)
    assert kept == [("a1", "a2"), ("a2", "a1")]
    assert dropped == 2


def test_filter_pairs_by_corpus_handles_empty_inputs() -> None:
    from consistency_checker.pipeline import _filter_pairs_by_corpus

    assert _filter_pairs_by_corpus([], {"a"}) == ([], 0)
    assert _filter_pairs_by_corpus([("a", "b")], set()) == ([], 1)


def test_filter_pairs_by_corpus_all_kept_when_all_in_corpus() -> None:
    from consistency_checker.pipeline import _filter_pairs_by_corpus

    corpus_ids = {"x", "y", "z"}
    pairs = [("x", "y"), ("y", "z"), ("x", "z")]
    kept, dropped = _filter_pairs_by_corpus(pairs, corpus_ids)
    assert kept == pairs
    assert dropped == 0


def test_filter_pairs_by_corpus_all_dropped_when_none_in_corpus() -> None:
    from consistency_checker.pipeline import _filter_pairs_by_corpus

    corpus_ids = {"a"}
    pairs = [("b", "c"), ("d", "e")]
    kept, dropped = _filter_pairs_by_corpus(pairs, corpus_ids)
    assert kept == []
    assert dropped == 2
