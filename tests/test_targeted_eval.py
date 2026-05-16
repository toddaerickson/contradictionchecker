"""Tests for the targeted-eval harness + bundled pairs.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.targeted_eval.harness import (
    BucketMetrics,
    TargetedPair,
    TargetedPrediction,
    TargetedResult,
    _bucket_metrics,
    _to_assertions,
    format_markdown_summary,
    load_pairs,
    main,
    run_targeted_eval,
    write_metrics_json,
)
from consistency_checker.check.llm_judge import FixtureJudge, JudgeVerdict
from consistency_checker.extract.schema import Assertion

PAIRS_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "targeted_eval" / "pairs.jsonl"


# --- load_pairs ----------------------------------------------------------


def test_load_pairs_reads_bundled_file() -> None:
    """The bundled pairs.jsonl loads cleanly and has the documented bucket totals."""
    pairs = load_pairs(PAIRS_PATH)
    assert len(pairs) == 120
    by_bucket: dict[str, int] = {}
    for p in pairs:
        by_bucket[p.bucket] = by_bucket.get(p.bucket, 0) + 1
    assert by_bucket == {
        "rule_exception": 40,
        "modal_divergence": 40,
        "temporal_scope": 40,
    }


def test_load_pairs_ground_truth_distribution_matches_readme() -> None:
    """README claims 105 not_contradiction + 15 contradiction; this guards that contract."""
    pairs = load_pairs(PAIRS_PATH)
    counts: dict[str, int] = {}
    for p in pairs:
        counts[p.ground_truth] = counts.get(p.ground_truth, 0) + 1
    assert counts.get("not_contradiction") == 105
    assert counts.get("contradiction") == 15
    assert counts.get("uncertain", 0) == 0


def test_load_pairs_all_ids_unique() -> None:
    pairs = load_pairs(PAIRS_PATH)
    ids = [p.pair_id for p in pairs]
    assert len(ids) == len(set(ids))


def test_load_pairs_rule_exception_all_override() -> None:
    """Rule/exception bucket is the panel's primary FP hypothesis; all should be override."""
    pairs = [p for p in load_pairs(PAIRS_PATH) if p.bucket == "rule_exception"]
    assert all(p.ground_truth == "not_contradiction" for p in pairs)
    assert all(p.relation == "override" for p in pairs)


def test_load_pairs_temporal_scope_all_not_contradiction() -> None:
    pairs = [p for p in load_pairs(PAIRS_PATH) if p.bucket == "temporal_scope"]
    assert all(p.ground_truth == "not_contradiction" for p in pairs)


def test_load_pairs_rejects_invalid_bucket(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps(
            {
                "pair_id": "x",
                "bucket": "not_a_bucket",
                "assertion_a": "a",
                "assertion_b": "b",
                "ground_truth": "not_contradiction",
                "relation": "override",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid bucket"):
        load_pairs(bad)


def test_load_pairs_rejects_invalid_ground_truth(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps(
            {
                "pair_id": "x",
                "bucket": "rule_exception",
                "assertion_a": "a",
                "assertion_b": "b",
                "ground_truth": "definitely_a_contradiction",
                "relation": "override",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid ground_truth"):
        load_pairs(bad)


def test_load_pairs_rejects_invalid_relation(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps(
            {
                "pair_id": "x",
                "bucket": "rule_exception",
                "assertion_a": "a",
                "assertion_b": "b",
                "ground_truth": "not_contradiction",
                "relation": "fancy_relation",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid relation"):
        load_pairs(bad)


def test_load_pairs_rejects_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not valid json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_pairs(bad)


def test_load_pairs_rejects_missing_required_field(tmp_path: Path) -> None:
    """Missing 'assertion_b' must surface as a clean ValueError, not a bare KeyError."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps(
            {
                "pair_id": "x",
                "bucket": "rule_exception",
                "assertion_a": "a",
                "ground_truth": "not_contradiction",
                "relation": "override",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required field 'assertion_b'"):
        load_pairs(bad)


def test_load_pairs_rejects_duplicate_pair_id(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    row = json.dumps(
        {
            "pair_id": "dup",
            "bucket": "rule_exception",
            "assertion_a": "a",
            "assertion_b": "b",
            "ground_truth": "not_contradiction",
            "relation": "override",
        }
    )
    bad.write_text(row + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate pair_id 'dup'"):
        load_pairs(bad)


def test_load_pairs_skips_blank_lines_and_comment_lines(tmp_path: Path) -> None:
    """Empty lines + // comment lines are ignored so the file stays editable."""
    p = tmp_path / "ok.jsonl"
    p.write_text(
        "// this is a comment\n"
        "\n"
        + json.dumps(
            {
                "pair_id": "rule_exc_a",
                "bucket": "rule_exception",
                "assertion_a": "a",
                "assertion_b": "b",
                "ground_truth": "not_contradiction",
                "relation": "override",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pairs = load_pairs(p)
    assert len(pairs) == 1
    assert pairs[0].pair_id == "rule_exc_a"


# --- _to_assertions ------------------------------------------------------


def test_to_assertions_uses_pair_id_for_doc_uniqueness() -> None:
    pair = TargetedPair(
        pair_id="x_001",
        bucket="rule_exception",
        assertion_a="text A",
        assertion_b="text B",
        ground_truth="not_contradiction",
        relation="override",
    )
    a, b = _to_assertions(pair)
    assert isinstance(a, Assertion)
    assert isinstance(b, Assertion)
    assert a.doc_id == "x_001_a"
    assert b.doc_id == "x_001_b"
    assert a.assertion_id != b.assertion_id


# --- _bucket_metrics ----------------------------------------------------


def _pred(
    *,
    pair_id: str,
    bucket: str = "rule_exception",
    ground_truth: str,
    predicted: str,
    relation: str = "override",
    confidence: float = 0.9,
) -> TargetedPrediction:
    return TargetedPrediction(
        pair_id=pair_id,
        bucket=bucket,  # type: ignore[arg-type]
        ground_truth=ground_truth,  # type: ignore[arg-type]
        relation=relation,  # type: ignore[arg-type]
        predicted=predicted,  # type: ignore[arg-type]
        judge_confidence=confidence,
        judge_rationale="r",
    )


def test_bucket_metrics_correct_on_all_correct() -> None:
    preds = [
        _pred(pair_id="a", ground_truth="not_contradiction", predicted="not_contradiction"),
        _pred(pair_id="b", ground_truth="not_contradiction", predicted="not_contradiction"),
    ]
    m = _bucket_metrics("rule_exception", preds)
    assert m.n_pairs == 2
    assert m.n_correct == 2
    assert m.accuracy == 1.0
    assert m.n_false_positive == 0
    assert m.false_positive_rate == 0.0


def test_bucket_metrics_counts_false_positive() -> None:
    preds = [
        _pred(pair_id="a", ground_truth="not_contradiction", predicted="contradiction"),
        _pred(pair_id="b", ground_truth="not_contradiction", predicted="not_contradiction"),
    ]
    m = _bucket_metrics("rule_exception", preds)
    assert m.n_false_positive == 1
    assert m.false_positive_rate == 0.5


def test_bucket_metrics_counts_false_negative() -> None:
    preds = [
        _pred(pair_id="a", ground_truth="contradiction", predicted="not_contradiction"),
        _pred(pair_id="b", ground_truth="contradiction", predicted="contradiction"),
    ]
    m = _bucket_metrics("modal_divergence", preds)
    assert m.n_false_negative == 1
    assert m.false_negative_rate == 0.5


def test_bucket_metrics_handles_abstain() -> None:
    preds = [
        _pred(pair_id="a", ground_truth="not_contradiction", predicted="uncertain"),
        _pred(pair_id="b", ground_truth="not_contradiction", predicted="not_contradiction"),
    ]
    m = _bucket_metrics("temporal_scope", preds)
    assert m.n_abstain == 1
    assert m.abstain_rate == 0.5


def test_bucket_metrics_zero_denominators_safe() -> None:
    """A bucket with only contradictions must not divide-by-zero when computing FP rate."""
    preds = [
        _pred(pair_id="a", ground_truth="contradiction", predicted="contradiction"),
    ]
    m = _bucket_metrics("modal_divergence", preds)
    assert m.false_positive_rate == 0.0  # no not_contradictions to be false-positive against


def test_bucket_metrics_fn_rate_zero_denominator_safe() -> None:
    """Symmetric guard: a bucket with only not_contradictions must not divide-by-zero on FN."""
    preds = [
        _pred(pair_id="a", ground_truth="not_contradiction", predicted="not_contradiction"),
    ]
    m = _bucket_metrics("rule_exception", preds)
    # n_contra == 0, so the FN denominator is zero; the guard must return 0.0 not NaN/Error.
    assert m.false_negative_rate == 0.0


# --- run_targeted_eval --------------------------------------------------


def _fixture_for(pair: TargetedPair, predicted: str, confidence: float = 0.9) -> JudgeVerdict:
    a, b = _to_assertions(pair)
    canonical = (min(a.assertion_id, b.assertion_id), max(a.assertion_id, b.assertion_id))
    return JudgeVerdict(
        assertion_a_id=canonical[0],
        assertion_b_id=canonical[1],
        verdict=predicted,
        confidence=confidence,
        rationale="r",
    )


def test_run_targeted_eval_produces_per_bucket_metrics() -> None:
    pairs = [
        TargetedPair("x1", "rule_exception", "A1", "B1", "not_contradiction", "override"),
        TargetedPair("x2", "modal_divergence", "A2", "B2", "contradiction", "contradiction"),
    ]
    fixtures = {}
    for pair, label in zip(pairs, ["not_contradiction", "contradiction"], strict=True):
        a, b = _to_assertions(pair)
        fixtures[(min(a.assertion_id, b.assertion_id), max(a.assertion_id, b.assertion_id))] = (
            JudgeVerdict(
                assertion_a_id=min(a.assertion_id, b.assertion_id),
                assertion_b_id=max(a.assertion_id, b.assertion_id),
                verdict=label,
                confidence=0.9,
                rationale="r",
            )
        )
    judge = FixtureJudge(fixtures)
    result = run_targeted_eval(pairs, judge=judge)
    assert result.n_pairs == 2
    assert result.overall_accuracy == 1.0
    buckets = {b.bucket: b for b in result.per_bucket}
    assert buckets["rule_exception"].n_pairs == 1
    assert buckets["modal_divergence"].n_pairs == 1


def test_run_targeted_eval_records_false_positive_against_override() -> None:
    """The whole point: judge calls override 'contradiction'; FP rate must reflect that."""
    pair = TargetedPair(
        pair_id="rule_exc_x",
        bucket="rule_exception",
        assertion_a="The Borrower shall not incur Indebtedness exceeding $5M.",
        assertion_b="Permitted Indebtedness is excluded from the foregoing restriction.",
        ground_truth="not_contradiction",
        relation="override",
    )
    judge = FixtureJudge({})  # no fixture → falls back to uncertain
    # Build a judge that returns contradiction for our specific pair
    a, b = _to_assertions(pair)
    key = (min(a.assertion_id, b.assertion_id), max(a.assertion_id, b.assertion_id))
    judge = FixtureJudge(
        {
            key: JudgeVerdict(
                assertion_a_id=key[0],
                assertion_b_id=key[1],
                verdict="contradiction",
                confidence=0.95,
                rationale="opposing predicates",
            )
        }
    )
    result = run_targeted_eval([pair], judge=judge)
    bucket = result.per_bucket[0]
    assert bucket.n_false_positive == 1
    assert bucket.false_positive_rate == 1.0


def test_run_targeted_eval_collapses_unknown_verdicts_to_uncertain() -> None:
    """A future detector might emit an unrecognised label; the harness must not crash."""

    class _OddVerdictJudge:
        def judge(
            self, a: Assertion, b: Assertion, *, numeric_context: str | None = None
        ) -> JudgeVerdict:
            del numeric_context
            return JudgeVerdict(
                assertion_a_id=a.assertion_id,
                assertion_b_id=b.assertion_id,
                verdict="numeric_short_circuit",  # outside the 3-value enum
                confidence=0.8,
                rationale="r",
            )

    pair = TargetedPair(
        pair_id="x",
        bucket="modal_divergence",
        assertion_a="A",
        assertion_b="B",
        ground_truth="not_contradiction",
        relation="independent",
    )
    result = run_targeted_eval([pair], judge=_OddVerdictJudge())
    # numeric_short_circuit collapses to 'contradiction' per the harness comment.
    assert result.predictions[0].predicted == "contradiction"


# --- format_markdown_summary --------------------------------------------


def test_format_markdown_summary_includes_all_buckets() -> None:
    result = TargetedResult(
        n_pairs=3,
        overall_accuracy=0.667,
        overall_false_positive_rate=0.5,
        overall_false_negative_rate=0.0,
        overall_abstain_rate=0.0,
        per_bucket=[
            BucketMetrics("rule_exception", 2, 1, 1, 0, 0, 0.5, 0.5, 0.0, 0.0),
            BucketMetrics("modal_divergence", 1, 1, 0, 0, 0, 1.0, 0.0, 0.0, 0.0),
        ],
    )
    out = format_markdown_summary(result)
    assert "rule_exception" in out
    assert "modal_divergence" in out
    assert "Pairs scored: **3**" in out


# --- write_metrics_json -------------------------------------------------


def test_write_metrics_json_round_trips(tmp_path: Path) -> None:
    result = TargetedResult(
        n_pairs=1,
        overall_accuracy=1.0,
        overall_false_positive_rate=0.0,
        overall_false_negative_rate=0.0,
        overall_abstain_rate=0.0,
        per_bucket=[BucketMetrics("rule_exception", 1, 1, 0, 0, 0, 1.0, 0.0, 0.0, 0.0)],
        predictions=[
            TargetedPrediction(
                pair_id="x",
                bucket="rule_exception",
                ground_truth="not_contradiction",
                relation="override",
                predicted="not_contradiction",
                judge_confidence=0.9,
                judge_rationale="r",
            )
        ],
    )
    out = tmp_path / "m.json"
    write_metrics_json(result, out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["n_pairs"] == 1
    assert loaded["per_bucket"][0]["bucket"] == "rule_exception"
    assert loaded["predictions"][0]["pair_id"] == "x"


# --- CLI entrypoint -----------------------------------------------------


def test_main_runs_against_fixture_judge_via_monkeypatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI's --judge-provider is API-bound; swap the factory for a fixture-driven one."""

    class _ConstantJudge:
        def judge(
            self, a: Assertion, b: Assertion, *, numeric_context: str | None = None
        ) -> JudgeVerdict:
            del numeric_context
            return JudgeVerdict(
                assertion_a_id=a.assertion_id,
                assertion_b_id=b.assertion_id,
                verdict="not_contradiction",
                confidence=0.5,
                rationale="constant",
            )

    monkeypatch.setattr(
        "benchmarks.targeted_eval.harness._make_judge",
        lambda provider, model: _ConstantJudge(),
    )

    out = tmp_path / "metrics.json"
    md = tmp_path / "summary.md"
    rc = main(
        [
            "--input",
            str(PAIRS_PATH),
            "--output",
            str(out),
            "--markdown",
            str(md),
        ]
    )
    assert rc == 0
    assert out.exists()
    assert md.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_pairs"] == 120
    # Constant 'not_contradiction' judge → accuracy = 105/120 (skipping the 15 contradictions)
    assert payload["overall_accuracy"] == pytest.approx(105 / 120)
    captured = capsys.readouterr()
    assert "Pairs scored: **120**" in captured.out

    # Per-bucket sanity: constant-'not_contradiction' judge gets every rule_exception
    # pair right (all 40 are not_contradiction); the 15 contradictions in modal_divergence
    # become false negatives; temporal_scope is all not_contradiction so 100% accuracy.
    by_bucket = {b["bucket"]: b for b in payload["per_bucket"]}
    assert by_bucket["rule_exception"]["n_pairs"] == 40
    assert by_bucket["rule_exception"]["accuracy"] == 1.0
    assert by_bucket["rule_exception"]["false_negative_rate"] == 0.0
    assert by_bucket["modal_divergence"]["n_pairs"] == 40
    assert by_bucket["modal_divergence"]["false_negative_rate"] == 1.0  # all 15 contras missed
    assert by_bucket["temporal_scope"]["n_pairs"] == 40
    assert by_bucket["temporal_scope"]["accuracy"] == 1.0
