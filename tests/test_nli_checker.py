"""Tests for the Stage A NLI checker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from consistency_checker.check.nli_checker import (
    FixtureNliChecker,
    NliResult,
    TransformerNliChecker,
    _split_score,
    passes_threshold,
    score_bidirectional,
)

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "nli_pairs.jsonl"


# --- NliResult --------------------------------------------------------------


def test_nli_result_label_is_argmax() -> None:
    r = NliResult.from_scores(p_contradiction=0.7, p_entailment=0.2, p_neutral=0.1)
    assert r.label == "contradiction"
    r2 = NliResult.from_scores(p_contradiction=0.1, p_entailment=0.8, p_neutral=0.1)
    assert r2.label == "entailment"
    r3 = NliResult.from_scores(p_contradiction=0.1, p_entailment=0.1, p_neutral=0.8)
    assert r3.label == "neutral"


def test_nli_result_is_frozen() -> None:
    r = NliResult.from_scores(p_contradiction=0.5, p_entailment=0.3, p_neutral=0.2)
    with pytest.raises(AttributeError):
        r.p_contradiction = 0.9  # type: ignore[misc]


# --- FixtureNliChecker ------------------------------------------------------


def test_fixture_checker_returns_canned_result() -> None:
    expected = NliResult.from_scores(p_contradiction=0.85, p_entailment=0.05, p_neutral=0.10)
    checker = FixtureNliChecker({("Rev grew.", "Rev shrank."): expected})
    assert checker.score("Rev grew.", "Rev shrank.") == expected


def test_fixture_checker_unknown_pair_defaults_to_neutral() -> None:
    checker = FixtureNliChecker({})
    out = checker.score("foo", "bar")
    assert out.label == "neutral"
    assert out.p_neutral == 1.0


# --- score_bidirectional ----------------------------------------------------


def test_score_bidirectional_picks_higher_contradiction_direction() -> None:
    forward = NliResult.from_scores(p_contradiction=0.3, p_entailment=0.1, p_neutral=0.6)
    reverse = NliResult.from_scores(p_contradiction=0.8, p_entailment=0.1, p_neutral=0.1)
    checker = FixtureNliChecker({("A", "B"): forward, ("B", "A"): reverse})
    out = score_bidirectional(checker, "A", "B")
    assert out is reverse


def test_score_bidirectional_ties_prefer_forward() -> None:
    same = NliResult.from_scores(p_contradiction=0.5, p_entailment=0.3, p_neutral=0.2)
    checker = FixtureNliChecker({("A", "B"): same, ("B", "A"): same})
    assert score_bidirectional(checker, "A", "B") is checker.score("A", "B")


# --- passes_threshold -------------------------------------------------------


def test_passes_threshold_strict_above() -> None:
    high = NliResult.from_scores(p_contradiction=0.6, p_entailment=0.2, p_neutral=0.2)
    low = NliResult.from_scores(p_contradiction=0.4, p_entailment=0.3, p_neutral=0.3)
    assert passes_threshold(high, 0.5) is True
    assert passes_threshold(low, 0.5) is False


def test_passes_threshold_at_boundary_passes() -> None:
    boundary = NliResult.from_scores(p_contradiction=0.5, p_entailment=0.25, p_neutral=0.25)
    assert passes_threshold(boundary, 0.5) is True


# --- fixture file integrity -------------------------------------------------


def test_nli_pairs_fixture_loads() -> None:
    lines = FIXTURES_PATH.read_text().strip().splitlines()
    assert len(lines) == 5
    for raw in lines:
        row = json.loads(raw)
        assert {"premise", "hypothesis", "expected"} <= row.keys()
        assert row["expected"] in {"contradiction", "entailment", "neutral"}


# --- live model test --------------------------------------------------------


# --- _split_score -------------------------------------------------------


def _mock_score(premise: str, hypothesis: str) -> NliResult:
    # Returns high contradiction only for a specific short sentence
    if "bad" in premise:
        return NliResult.from_scores(p_contradiction=0.9, p_entailment=0.05, p_neutral=0.05)
    return NliResult.from_scores(p_contradiction=0.1, p_entailment=0.1, p_neutral=0.8)


def test_split_score_detects_contradiction_in_split_premise() -> None:
    long_premise = "This is fine. This is bad. This is also fine."
    result = _split_score(
        long_premise,
        "something",
        score_fn=_mock_score,
        max_tokens=4,  # force split (any real sentence exceeds 4 tokens)
        tokenize_fn=str.split,  # word-count proxy for tokens
    )
    assert result.p_contradiction >= 0.9


def test_split_score_passthrough_when_short() -> None:
    result = _split_score(
        "Short.",
        "Also short.",
        score_fn=_mock_score,
        max_tokens=1000,
        tokenize_fn=str.split,
    )
    # Should call score_fn once with full premise, not split
    assert result.label == "neutral"  # "Short." doesn't contain "bad"


# --- live model test --------------------------------------------------------


@pytest.mark.slow
def test_transformer_nli_checker_on_hand_labeled() -> None:
    """Real DeBERTa-class model. Must agree with ≥ 4/5 hand-labeled pairs."""
    checker = TransformerNliChecker()
    rows = [json.loads(line) for line in FIXTURES_PATH.read_text().strip().splitlines()]
    correct = 0
    for row in rows:
        result = score_bidirectional(checker, row["premise"], row["hypothesis"])
        if result.label == row["expected"]:
            correct += 1
    assert correct >= 4, f"only {correct}/5 hand-labeled pairs matched expected label"
