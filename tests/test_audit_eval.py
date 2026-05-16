"""Tests for :mod:`consistency_checker.audit.eval` precision/calibration mining."""

from __future__ import annotations

import csv
from itertools import pairwise
from pathlib import Path

import pytest

from consistency_checker.audit.eval import (
    DEFAULT_CALIBRATION_BINS,
    CalibrationBin,
    DetectorPrecision,
    EvalRow,
    compute_calibration,
    compute_detector_precision,
    eval_filename,
    format_calibration_table,
    format_precision_table,
    iter_eval_rows,
    iter_multi_party_eval_rows,
    iter_pair_eval_rows,
    write_calibration_csv,
    write_precision_csv,
)
from consistency_checker.audit.logger import AuditLogger
from consistency_checker.check.gate import CandidatePair
from consistency_checker.check.llm_judge import JudgeVerdict
from consistency_checker.check.nli_checker import NliResult
from consistency_checker.extract.schema import Assertion, Document
from consistency_checker.index.assertion_store import AssertionStore


@pytest.fixture
def store(tmp_path: Path) -> AssertionStore:
    s = AssertionStore(tmp_path / "store.db")
    s.migrate()
    return s


def _seed_two_assertions(store: AssertionStore) -> tuple[Assertion, Assertion]:
    doc_a = Document.from_content("Body A.", source_path="a.txt")
    doc_b = Document.from_content("Body B.", source_path="b.txt")
    store.add_document(doc_a)
    store.add_document(doc_b)
    a = Assertion.build(doc_a.doc_id, "Revenue grew 12% in fiscal 2025.")
    b = Assertion.build(doc_b.doc_id, "Revenue declined 5% in fiscal 2025.")
    store.add_assertions([a, b])
    return a, b


def _seed_three_assertions(store: AssertionStore) -> tuple[Assertion, Assertion, Assertion]:
    doc_a = Document.from_content("Body A.", source_path="a.txt")
    doc_b = Document.from_content("Body B.", source_path="b.txt")
    doc_c = Document.from_content("Body C.", source_path="c.txt")
    for d in (doc_a, doc_b, doc_c):
        store.add_document(d)
    a = Assertion.build(doc_a.doc_id, "Alpha.")
    b = Assertion.build(doc_b.doc_id, "Beta.")
    c = Assertion.build(doc_c.doc_id, "Gamma.")
    store.add_assertions([a, b, c])
    return a, b, c


def _record_pair_finding(
    store: AssertionStore,
    *,
    a: Assertion,
    b: Assertion,
    verdict: str,
    confidence: float,
    run_id: str | None = None,
) -> tuple[str, str]:
    logger = AuditLogger(store)
    rid = run_id or logger.begin_run()
    candidate = CandidatePair(a=a, b=b, score=0.9)
    nli = NliResult.from_scores(p_contradiction=0.7, p_entailment=0.1, p_neutral=0.2)
    jv = JudgeVerdict(
        assertion_a_id=a.assertion_id,
        assertion_b_id=b.assertion_id,
        verdict=verdict,
        confidence=confidence,
        rationale="r",
    )
    fid = logger.record_finding(rid, candidate=candidate, nli=nli, verdict=jv)
    return rid, fid


# --- iter_pair_eval_rows ---------------------------------------------------


def test_iter_pair_eval_rows_empty_when_no_reviews(store: AssertionStore) -> None:
    a, b = _seed_two_assertions(store)
    _record_pair_finding(store, a=a, b=b, verdict="contradiction", confidence=0.8)
    assert list(iter_pair_eval_rows(store)) == []


def test_iter_pair_eval_rows_joins_reviewed_pair(store: AssertionStore) -> None:
    a, b = _seed_two_assertions(store)
    _record_pair_finding(store, a=a, b=b, verdict="contradiction", confidence=0.85)
    logger = AuditLogger(store)
    pair_key = ":".join(sorted([a.assertion_id, b.assertion_id]))
    logger.set_reviewer_verdict(
        pair_key=pair_key, detector_type="contradiction", verdict="confirmed"
    )
    rows = list(iter_pair_eval_rows(store))
    assert len(rows) == 1
    only = rows[0]
    assert only.detector_type == "contradiction"
    assert only.judge_verdict == "contradiction"
    assert only.judge_confidence == pytest.approx(0.85)
    assert only.reviewer_verdict == "confirmed"


def test_iter_pair_eval_rows_excludes_unreviewed_findings(store: AssertionStore) -> None:
    """Findings without a matching reviewer_verdict row must not appear."""
    a, b = _seed_two_assertions(store)
    _record_pair_finding(store, a=a, b=b, verdict="contradiction", confidence=0.8)
    # No reviewer_verdict set.
    assert list(iter_pair_eval_rows(store)) == []


def test_iter_pair_eval_rows_matches_definition_detector(store: AssertionStore) -> None:
    """Definition findings with detector_type='definition_inconsistency' should join."""
    from consistency_checker.check.definition_checker import (
        DefinitionFinding,
        DefinitionPair,
    )
    from consistency_checker.check.definition_judge import DefinitionJudgeVerdict

    doc_a = Document.from_content("A", source_path="a.txt")
    doc_b = Document.from_content("B", source_path="b.txt")
    store.add_document(doc_a)
    store.add_document(doc_b)
    a = Assertion.build(
        doc_a.doc_id, '"X" means foo.', kind="definition", term="X", definition_text="foo"
    )
    b = Assertion.build(
        doc_b.doc_id, '"X" means bar.', kind="definition", term="X", definition_text="bar"
    )
    store.add_assertions([a, b])
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    sorted_ids = sorted([a.assertion_id, b.assertion_id])
    finding = DefinitionFinding(
        pair=DefinitionPair(a=a, b=b, canonical_term="x"),
        verdict=DefinitionJudgeVerdict(
            assertion_a_id=sorted_ids[0],
            assertion_b_id=sorted_ids[1],
            verdict="definition_divergent",
            confidence=0.75,
            rationale="scope shift",
            evidence_spans=[],
        ),
    )
    logger.record_definition_finding(run_id, finding=finding)
    logger.set_reviewer_verdict(
        pair_key=":".join(sorted_ids),
        detector_type="definition_inconsistency",
        verdict="false_positive",
    )
    rows = list(iter_pair_eval_rows(store))
    assert len(rows) == 1
    assert rows[0].detector_type == "definition_inconsistency"
    assert rows[0].reviewer_verdict == "false_positive"


def test_iter_pair_eval_rows_join_does_not_cross_detector_types(
    store: AssertionStore,
) -> None:
    """A reviewer verdict tagged with detector X must not join a finding from detector Y."""
    a, b = _seed_two_assertions(store)
    _record_pair_finding(store, a=a, b=b, verdict="contradiction", confidence=0.8)
    pair_key = ":".join(sorted([a.assertion_id, b.assertion_id]))
    logger = AuditLogger(store)
    # Verdict tagged with the wrong detector_type — must not match.
    logger.set_reviewer_verdict(
        pair_key=pair_key,
        detector_type="definition_inconsistency",
        verdict="confirmed",
    )
    assert list(iter_pair_eval_rows(store)) == []


# --- iter_multi_party_eval_rows --------------------------------------------


def test_iter_multi_party_eval_rows_joins_reviewed_triangle(store: AssertionStore) -> None:
    a, b, c = _seed_three_assertions(store)
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    assertion_ids = [a.assertion_id, b.assertion_id, c.assertion_id]
    doc_ids = [a.doc_id, b.doc_id, c.doc_id]
    logger.record_multi_party_finding(
        run_id,
        assertion_ids=assertion_ids,
        doc_ids=doc_ids,
        judge_verdict="multi_party_contradiction",
        judge_confidence=0.7,
        judge_rationale="r",
    )
    logger.set_reviewer_verdict(
        pair_key=":".join(sorted(assertion_ids)),
        detector_type="multi_party",
        verdict="confirmed",
    )
    rows = list(iter_multi_party_eval_rows(store))
    assert len(rows) == 1
    assert rows[0].detector_type == "multi_party"
    assert rows[0].judge_verdict == "multi_party_contradiction"
    assert rows[0].judge_confidence == pytest.approx(0.7)
    assert rows[0].reviewer_verdict == "confirmed"


def test_iter_multi_party_eval_rows_empty_when_no_findings(store: AssertionStore) -> None:
    assert list(iter_multi_party_eval_rows(store)) == []


def test_iter_multi_party_eval_rows_excludes_unreviewed_triangle(
    store: AssertionStore,
) -> None:
    """Triangle findings without a matching reviewer_verdict row must not appear."""
    a, b, c = _seed_three_assertions(store)
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    logger.record_multi_party_finding(
        run_id,
        assertion_ids=[a.assertion_id, b.assertion_id, c.assertion_id],
        doc_ids=[a.doc_id, b.doc_id, c.doc_id],
        judge_verdict="multi_party_contradiction",
        judge_confidence=0.7,
        judge_rationale="r",
    )
    # No reviewer_verdict set.
    assert list(iter_multi_party_eval_rows(store)) == []


def test_iter_eval_rows_combines_both_shapes(store: AssertionStore) -> None:
    """The all-rows iterator should yield pair + multi-party rows."""
    a, b, c = _seed_three_assertions(store)
    logger = AuditLogger(store)
    # Pair finding + verdict
    _record_pair_finding(store, a=a, b=b, verdict="contradiction", confidence=0.8)
    pair_key = ":".join(sorted([a.assertion_id, b.assertion_id]))
    logger.set_reviewer_verdict(
        pair_key=pair_key, detector_type="contradiction", verdict="confirmed"
    )
    # Triangle finding + verdict
    run_id = logger.begin_run()
    tri_ids = [a.assertion_id, b.assertion_id, c.assertion_id]
    logger.record_multi_party_finding(
        run_id,
        assertion_ids=tri_ids,
        doc_ids=[a.doc_id, b.doc_id, c.doc_id],
        judge_verdict="multi_party_contradiction",
        judge_confidence=0.6,
        judge_rationale="r",
    )
    logger.set_reviewer_verdict(
        pair_key=":".join(sorted(tri_ids)), detector_type="multi_party", verdict="false_positive"
    )
    rows = list(iter_eval_rows(store))
    detectors = sorted(r.detector_type for r in rows)
    assert detectors == ["contradiction", "multi_party"]


# --- compute_detector_precision -------------------------------------------


def test_compute_precision_basic() -> None:
    rows = [
        EvalRow("contradiction", "contradiction", 0.9, "confirmed", None),
        EvalRow("contradiction", "contradiction", 0.7, "confirmed", None),
        EvalRow("contradiction", "contradiction", 0.6, "false_positive", None),
        EvalRow("contradiction", "contradiction", 0.5, "dismissed", None),
    ]
    [p] = compute_detector_precision(rows)
    assert p.detector_type == "contradiction"
    assert p.n_confirmed == 2
    assert p.n_false_positive == 1
    assert p.n_dismissed == 1
    assert p.n_reviewed == 4
    assert p.precision == pytest.approx(2 / 3)


def test_compute_precision_dismissed_excluded_from_denominator() -> None:
    rows = [
        EvalRow("contradiction", "contradiction", 0.9, "dismissed", None),
        EvalRow("contradiction", "contradiction", 0.9, "dismissed", None),
    ]
    [p] = compute_detector_precision(rows)
    assert p.precision is None  # no eval-relevant verdicts


def test_compute_precision_groups_per_detector() -> None:
    rows = [
        EvalRow("contradiction", "contradiction", 0.9, "confirmed", None),
        EvalRow("definition_inconsistency", "definition_divergent", 0.7, "false_positive", None),
    ]
    out = {p.detector_type: p for p in compute_detector_precision(rows)}
    assert out["contradiction"].precision == 1.0
    assert out["definition_inconsistency"].precision == 0.0


def test_compute_precision_returns_empty_on_no_rows() -> None:
    assert compute_detector_precision([]) == []


# --- compute_calibration --------------------------------------------------


def test_calibration_buckets_by_confidence() -> None:
    rows = [
        EvalRow("contradiction", "contradiction", 0.95, "confirmed", None),
        EvalRow("contradiction", "contradiction", 0.85, "false_positive", None),
        EvalRow("contradiction", "contradiction", 0.55, "confirmed", None),
    ]
    bins = compute_calibration(rows, detector_type="contradiction")
    by_low = {b.confidence_low: b for b in bins}
    assert by_low[0.9].n_reviewed == 1
    assert by_low[0.9].precision == 1.0
    assert by_low[0.8].n_reviewed == 1
    assert by_low[0.8].precision == 0.0
    assert by_low[0.5].n_reviewed == 1
    assert by_low[0.5].precision == 1.0


def test_calibration_skips_other_detectors() -> None:
    rows = [
        EvalRow("contradiction", "contradiction", 0.9, "confirmed", None),
        EvalRow("definition_inconsistency", "definition_divergent", 0.9, "false_positive", None),
    ]
    bins = compute_calibration(rows, detector_type="contradiction")
    top = next(b for b in bins if b.confidence_low == 0.9)
    assert top.n_reviewed == 1  # only the contradiction row counts


def test_calibration_drops_null_confidence_rows() -> None:
    rows = [EvalRow("contradiction", "numeric_short_circuit", None, "confirmed", None)]
    bins = compute_calibration(rows, detector_type="contradiction")
    assert all(b.n_reviewed == 0 for b in bins)


def test_calibration_default_bins_have_full_coverage() -> None:
    """The default bins cover [0.0, 1.0] without gaps and the top edge includes 1.0."""
    bins = DEFAULT_CALIBRATION_BINS
    for (_, prev_high), (next_low, _) in pairwise(bins):
        assert prev_high == next_low
    assert bins[0][0] == 0.0
    assert bins[-1][1] > 1.0  # 1.0001 sentinel ensures judge_confidence=1.0 is bucketed


# --- formatters ------------------------------------------------------------


def test_format_precision_table_empty_message() -> None:
    assert "No reviewed" in format_precision_table([])


def test_format_precision_table_renders_rows() -> None:
    out = format_precision_table(
        [
            DetectorPrecision(
                "contradiction",
                n_confirmed=2,
                n_false_positive=3,
                n_dismissed=1,
                n_reviewed=6,
                precision=0.4,
            )
        ]
    )
    assert "contradiction" in out
    assert "40.0%" in out


def test_format_calibration_table_renders_detector_name() -> None:
    out = format_calibration_table(
        [
            CalibrationBin(
                confidence_low=0.9,
                confidence_high=1.0001,
                n_reviewed=2,
                n_confirmed=1,
                n_false_positive=1,
                n_dismissed=0,
                precision=0.5,
            )
        ],
        detector_type="contradiction",
    )
    assert "contradiction" in out
    assert "[0.90, 1.00)" in out
    assert "50.0%" in out


def test_format_calibration_table_no_data_message() -> None:
    bins = [CalibrationBin(low, high, 0, 0, 0, 0, None) for low, high in DEFAULT_CALIBRATION_BINS]
    out = format_calibration_table(bins, detector_type="contradiction")
    assert "no reviewed" in out


# --- CSV writers ----------------------------------------------------------


def test_write_precision_csv_round_trip(tmp_path: Path) -> None:
    precisions = [
        DetectorPrecision(
            "contradiction",
            n_confirmed=3,
            n_false_positive=1,
            n_dismissed=2,
            n_reviewed=6,
            precision=0.75,
        ),
        DetectorPrecision(
            "definition_inconsistency",
            n_confirmed=0,
            n_false_positive=0,
            n_dismissed=0,
            n_reviewed=0,
            precision=None,
        ),
    ]
    out_path = tmp_path / "p.csv"
    write_precision_csv(precisions, out_path)
    with open(out_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["detector_type"] == "contradiction"
    assert rows[0]["precision"] == "0.750000"
    assert rows[1]["precision"] == ""  # None → empty cell


def test_write_calibration_csv_round_trip(tmp_path: Path) -> None:
    bins_by_detector = {
        "contradiction": [
            CalibrationBin(0.9, 1.0001, 2, 1, 1, 0, 0.5),
            CalibrationBin(0.5, 0.7, 1, 0, 1, 0, 0.0),
        ]
    }
    out_path = tmp_path / "c.csv"
    write_calibration_csv(bins_by_detector, out_path)
    with open(out_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    by_low = {float(r["confidence_low"]): r for r in rows}
    assert by_low[0.9]["precision"] == "0.500000"
    assert by_low[0.5]["precision"] == "0.000000"


def test_eval_filename_uses_cc_prefix_and_kind() -> None:
    name = eval_filename("precision")
    assert name.startswith("cc_eval_precision_")
    assert name.endswith(".csv")
