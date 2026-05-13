"""Tests for the CONTRADOC benchmark harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.contradoc_harness import (
    BenchmarkResult,
    ContradocEntry,
    _confusion,
    _f1_components,
    load_contradoc,
    run_benchmark,
    write_metrics_json,
)
from consistency_checker.check.llm_judge import FixtureJudge, JudgeVerdict
from consistency_checker.check.nli_checker import FixtureNliChecker, NliResult
from consistency_checker.extract.schema import Assertion

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "contradoc_sample.jsonl"


# --- load_contradoc --------------------------------------------------------


def test_load_contradoc_parses_jsonl() -> None:
    entries = load_contradoc(FIXTURE_PATH)
    assert len(entries) == 5
    assert all(isinstance(e, ContradocEntry) for e in entries)
    labels = {e.label for e in entries}
    assert labels == {"contradiction", "not_contradiction"}


def test_load_contradoc_respects_sample() -> None:
    entries = load_contradoc(FIXTURE_PATH, sample=2)
    assert len(entries) == 2


def test_load_contradoc_rejects_unknown_label(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"pair_id":"x","document_a":"a","document_b":"b","label":"maybe"}\n')
    with pytest.raises(ValueError, match="Unexpected label"):
        load_contradoc(bad)


# --- math helpers ----------------------------------------------------------


def test_confusion_counts_each_quadrant() -> None:
    from benchmarks.contradoc_harness import BenchmarkPrediction

    preds = [
        BenchmarkPrediction("a", "contradiction", "contradiction", "contradiction", 0.9, 0.8),
        BenchmarkPrediction("b", "contradiction", "not_contradiction", "uncertain", 0.4, 0.4),
        BenchmarkPrediction("c", "not_contradiction", "contradiction", "contradiction", 0.6, 0.5),
        BenchmarkPrediction(
            "d", "not_contradiction", "not_contradiction", "not_contradiction", 0.1, 0.1
        ),
    ]
    tp, fp, tn, fn = _confusion(preds)
    assert (tp, fp, tn, fn) == (1, 1, 1, 1)


def test_f1_components_perfect_score() -> None:
    precision, recall, f1 = _f1_components(tp=5, fp=0, fn=0)
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0


def test_f1_components_zero_denominators_are_safe() -> None:
    p, r, f1 = _f1_components(tp=0, fp=0, fn=0)
    assert (p, r, f1) == (0.0, 0.0, 0.0)


# --- run_benchmark ---------------------------------------------------------


def _build_fixtures(entries: list[ContradocEntry]) -> tuple[FixtureNliChecker, FixtureJudge]:
    """Construct NLI and judge fixtures that mirror each entry's gold label."""
    nli_fixtures: dict[tuple[str, str], NliResult] = {}
    judge_fixtures: dict[tuple[str, str], JudgeVerdict] = {}
    for entry in entries:
        # Stage A: give contradictions a high p_contradiction, others low.
        p_c = 0.85 if entry.label == "contradiction" else 0.2
        nli_fixtures[(entry.document_a, entry.document_b)] = NliResult.from_scores(
            p_contradiction=p_c, p_entailment=0.05, p_neutral=1.0 - p_c - 0.05
        )
        nli_fixtures[(entry.document_b, entry.document_a)] = NliResult.from_scores(
            p_contradiction=max(0.0, p_c - 0.05),
            p_entailment=0.05,
            p_neutral=1.0 - max(0.0, p_c - 0.05) - 0.05,
        )

        # Stage B: produce the canonical key against synthesised assertion ids
        # that the harness will compute. Mirror the harness's id scheme.
        a = Assertion.build(doc_id=f"{entry.pair_id}_a", assertion_text=entry.document_a)
        b = Assertion.build(doc_id=f"{entry.pair_id}_b", assertion_text=entry.document_b)
        canonical = (
            min(a.assertion_id, b.assertion_id),
            max(a.assertion_id, b.assertion_id),
        )
        if entry.label == "contradiction":
            judge_fixtures[canonical] = JudgeVerdict(
                assertion_a_id=canonical[0],
                assertion_b_id=canonical[1],
                verdict="contradiction",
                confidence=0.9,
                rationale="gold contradiction",
            )
        else:
            judge_fixtures[canonical] = JudgeVerdict(
                assertion_a_id=canonical[0],
                assertion_b_id=canonical[1],
                verdict="not_contradiction",
                confidence=0.8,
                rationale="gold consistent",
            )
    return FixtureNliChecker(nli_fixtures), FixtureJudge(judge_fixtures)


def test_run_benchmark_with_perfect_fixtures_yields_f1_1() -> None:
    entries = load_contradoc(FIXTURE_PATH)
    nli, judge = _build_fixtures(entries)
    result = run_benchmark(entries, nli_checker=nli, judge=judge, nli_threshold=0.5)
    assert result.n_pairs == 5
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_run_benchmark_skips_judge_when_nli_below_threshold() -> None:
    """If Stage A gates out a real contradiction, it counts as a false negative."""
    entries = [e for e in load_contradoc(FIXTURE_PATH) if e.label == "contradiction"]
    nli_fixtures = {
        (e.document_a, e.document_b): NliResult.from_scores(
            p_contradiction=0.1, p_entailment=0.4, p_neutral=0.5
        )
        for e in entries
    }
    nli_fixtures.update(
        {
            (e.document_b, e.document_a): NliResult.from_scores(
                p_contradiction=0.1, p_entailment=0.4, p_neutral=0.5
            )
            for e in entries
        }
    )
    # The judge would say contradiction — but Stage A never calls it.
    a0 = Assertion.build(doc_id=f"{entries[0].pair_id}_a", assertion_text=entries[0].document_a)
    b0 = Assertion.build(doc_id=f"{entries[0].pair_id}_b", assertion_text=entries[0].document_b)
    judge_key = (
        min(a0.assertion_id, b0.assertion_id),
        max(a0.assertion_id, b0.assertion_id),
    )
    judge = FixtureJudge(
        {
            judge_key: JudgeVerdict(
                assertion_a_id=judge_key[0],
                assertion_b_id=judge_key[1],
                verdict="contradiction",
                confidence=1.0,
                rationale="should never be invoked",
            )
        }
    )
    result = run_benchmark(
        entries, nli_checker=FixtureNliChecker(nli_fixtures), judge=judge, nli_threshold=0.5
    )
    assert result.n_true_positive == 0
    assert result.n_false_negative == len(entries)
    judge_invocations = [p.judge_verdict for p in result.predictions]
    assert all(v == "not_invoked" for v in judge_invocations)


def test_uncertain_verdict_treated_as_negative_prediction() -> None:
    """Stage B 'uncertain' must collapse to not_contradiction at the metrics layer."""
    entries = [e for e in load_contradoc(FIXTURE_PATH) if e.label == "contradiction"][:1]
    nli_fixtures = {
        (entries[0].document_a, entries[0].document_b): NliResult.from_scores(
            p_contradiction=0.8, p_entailment=0.05, p_neutral=0.15
        ),
        (entries[0].document_b, entries[0].document_a): NliResult.from_scores(
            p_contradiction=0.75, p_entailment=0.05, p_neutral=0.20
        ),
    }
    a = Assertion.build(doc_id=f"{entries[0].pair_id}_a", assertion_text=entries[0].document_a)
    b = Assertion.build(doc_id=f"{entries[0].pair_id}_b", assertion_text=entries[0].document_b)
    judge_key = (
        min(a.assertion_id, b.assertion_id),
        max(a.assertion_id, b.assertion_id),
    )
    judge = FixtureJudge(
        {
            judge_key: JudgeVerdict(
                assertion_a_id=judge_key[0],
                assertion_b_id=judge_key[1],
                verdict="uncertain",
                confidence=0.3,
                rationale="ambiguous",
            )
        }
    )
    result = run_benchmark(
        entries, nli_checker=FixtureNliChecker(nli_fixtures), judge=judge, nli_threshold=0.5
    )
    assert result.n_true_positive == 0
    assert result.n_false_negative == 1


# --- write_metrics_json ----------------------------------------------------


def test_write_metrics_json_round_trip(tmp_path: Path) -> None:
    result = BenchmarkResult(
        n_pairs=2,
        n_true_positive=1,
        n_false_positive=0,
        n_true_negative=1,
        n_false_negative=0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        nli_threshold=0.5,
        predictions=[],
    )
    out = tmp_path / "out" / "metrics.json"
    write_metrics_json(result, out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["precision"] == 1.0
    assert data["nli_threshold"] == 0.5
    assert "predictions" in data
