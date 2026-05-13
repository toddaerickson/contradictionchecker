"""Tests for the markdown report generator (Step 13)."""

from __future__ import annotations

from pathlib import Path

import pytest

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.audit.report import render_report
from consistency_checker.check.gate import CandidatePair
from consistency_checker.check.llm_judge import JudgeVerdict
from consistency_checker.check.nli_checker import NliResult
from consistency_checker.extract.schema import Assertion, Document
from consistency_checker.index.assertion_store import AssertionStore

EXPECTED_REPORT_PATH = Path(__file__).parent / "fixtures" / "expected_report.md"


@pytest.fixture
def seeded_store(tmp_path: Path) -> AssertionStore:
    store = AssertionStore(tmp_path / "store.db")
    store.migrate()
    return store


def _populate_three_doc_fixture(store: AssertionStore) -> tuple[Assertion, Assertion, Assertion]:
    """Three documents, three planted contradictions across two doc pairs.

    Layout:
      alpha (a1) <-> beta  (b1): clear contradiction, high confidence
      alpha (a1) <-> gamma (c1): mild contradiction, medium confidence
      beta  (b2) <-> gamma (c1): another pair on same gamma doc
    """
    doc_a = Document.from_content("Alpha body.", source_path="alpha.md", title="Alpha report")
    doc_b = Document.from_content("Beta body.", source_path="beta.txt", title="Beta brief")
    doc_c = Document.from_content("Gamma body.", source_path="gamma.txt", title="Gamma memo")
    store.add_document(doc_a)
    store.add_document(doc_b)
    store.add_document(doc_c)

    a1 = Assertion.build(doc_a.doc_id, "Revenue grew 12% in fiscal 2025.")
    b1 = Assertion.build(doc_b.doc_id, "Revenue declined 5% in fiscal 2025.")
    b2 = Assertion.build(doc_b.doc_id, "The Beta initiative began in 2024.")
    c1 = Assertion.build(doc_c.doc_id, "The Beta initiative began in 2023.")
    store.add_assertions([a1, b1, b2, c1])
    return a1, b1, c1  # for the planted contradiction pair


def _record_three_findings(store: AssertionStore, logger: AuditLogger, run_id: str) -> None:
    a_list = list(store.iter_assertions())
    by_text = {a.assertion_text: a for a in a_list}
    a1 = by_text["Revenue grew 12% in fiscal 2025."]
    b1 = by_text["Revenue declined 5% in fiscal 2025."]
    b2 = by_text["The Beta initiative began in 2024."]
    c1 = by_text["The Beta initiative began in 2023."]

    findings = [
        (
            CandidatePair(a=a1, b=b1, score=0.92),
            NliResult.from_scores(p_contradiction=0.83, p_entailment=0.05, p_neutral=0.12),
            JudgeVerdict(
                assertion_a_id=a1.assertion_id,
                assertion_b_id=b1.assertion_id,
                verdict="contradiction",
                confidence=0.90,
                rationale="Opposite revenue signs in the same fiscal year.",
                evidence_spans=["grew 12%", "declined 5%"],
            ),
        ),
        (
            CandidatePair(a=b2, b=c1, score=0.81),
            NliResult.from_scores(p_contradiction=0.66, p_entailment=0.10, p_neutral=0.24),
            JudgeVerdict(
                assertion_a_id=b2.assertion_id,
                assertion_b_id=c1.assertion_id,
                verdict="contradiction",
                confidence=0.72,
                rationale="Different start years for the same Beta initiative.",
                evidence_spans=["began in 2024", "began in 2023"],
            ),
        ),
        (
            CandidatePair(a=a1, b=c1, score=0.55),
            NliResult.from_scores(p_contradiction=0.20, p_entailment=0.10, p_neutral=0.70),
            JudgeVerdict(
                assertion_a_id=a1.assertion_id,
                assertion_b_id=c1.assertion_id,
                verdict="not_contradiction",
                confidence=0.40,
                rationale="Different subjects; no shared scope.",
            ),
        ),
    ]
    for cp, nli, verdict in findings:
        logger.record_finding(run_id, candidate=cp, nli=nli, verdict=verdict)


# --- empty / minimal --------------------------------------------------------


def test_report_with_no_contradictions(seeded_store: AssertionStore) -> None:
    logger = AuditLogger(seeded_store)
    run_id = logger.begin_run(run_id="run_empty")
    logger.end_run(run_id, n_assertions=0, n_pairs_gated=0, n_pairs_judged=0)
    report = render_report(seeded_store, logger, run_id=run_id)
    assert "# Consistency check report" in report
    assert "No contradictions met the reporting threshold" in report
    assert "## Summary" not in report


def test_report_missing_run_metadata_is_graceful(seeded_store: AssertionStore) -> None:
    logger = AuditLogger(seeded_store)
    report = render_report(seeded_store, logger, run_id="never_started")
    assert "Consistency check report" in report
    assert "never_started" in report
    assert "no metadata" in report


# --- golden-file regression ------------------------------------------------


def test_report_matches_golden_file(seeded_store: AssertionStore) -> None:
    _populate_three_doc_fixture(seeded_store)
    logger = AuditLogger(seeded_store)
    run_id = logger.begin_run(run_id="run_golden", config={"checker": "v1"})
    _record_three_findings(seeded_store, logger, run_id)
    logger.end_run(run_id, n_assertions=4, n_pairs_gated=3, n_pairs_judged=3)

    actual = render_report(seeded_store, logger, run_id=run_id)

    if not EXPECTED_REPORT_PATH.exists():
        EXPECTED_REPORT_PATH.write_text(actual, encoding="utf-8")
        pytest.fail(
            f"Golden file was missing; wrote it to {EXPECTED_REPORT_PATH}. "
            "Re-run the test to confirm round-trip."
        )

    expected = EXPECTED_REPORT_PATH.read_text(encoding="utf-8")
    assert actual == expected, (
        "Report changed; update tests/fixtures/expected_report.md if intentional"
    )


# --- filtering / sorting ---------------------------------------------------


def test_report_min_confidence_filters_out_low(seeded_store: AssertionStore) -> None:
    _populate_three_doc_fixture(seeded_store)
    logger = AuditLogger(seeded_store)
    run_id = logger.begin_run(run_id="run_filter")
    _record_three_findings(seeded_store, logger, run_id)
    logger.end_run(run_id, n_assertions=4, n_pairs_gated=3, n_pairs_judged=3)

    # Confidence threshold above the second contradiction (0.72) excludes it.
    report = render_report(seeded_store, logger, run_id=run_id, min_confidence=0.85)
    assert "Opposite revenue signs" in report
    assert "Different start years" not in report


def test_report_excludes_non_contradictions(seeded_store: AssertionStore) -> None:
    _populate_three_doc_fixture(seeded_store)
    logger = AuditLogger(seeded_store)
    run_id = logger.begin_run(run_id="run_exclude")
    _record_three_findings(seeded_store, logger, run_id)
    logger.end_run(run_id, n_assertions=4, n_pairs_gated=3, n_pairs_judged=3)
    report = render_report(seeded_store, logger, run_id=run_id)
    # The third finding was "not_contradiction" and must not appear.
    assert "Different subjects" not in report


def test_report_summary_table_sorted_by_confidence_desc(seeded_store: AssertionStore) -> None:
    _populate_three_doc_fixture(seeded_store)
    logger = AuditLogger(seeded_store)
    run_id = logger.begin_run(run_id="run_sort")
    _record_three_findings(seeded_store, logger, run_id)
    logger.end_run(run_id, n_assertions=4, n_pairs_gated=3, n_pairs_judged=3)
    report = render_report(seeded_store, logger, run_id=run_id)
    summary_start = report.index("## Summary")
    findings_start = report.index("## Findings")
    summary_block = report[summary_start:findings_start]
    rev_idx = summary_block.index("Opposite revenue signs")
    beta_idx = summary_block.index("Different start years")
    assert rev_idx < beta_idx, "higher-confidence row must precede lower-confidence row"


# --- multi-party section (F4) ---------------------------------------------


def test_report_omits_multi_party_section_when_no_findings(
    seeded_store: AssertionStore,
) -> None:
    """A pair-only run produces a report without the multi-party header."""
    _populate_three_doc_fixture(seeded_store)
    logger = AuditLogger(seeded_store)
    run_id = logger.begin_run(run_id="run_no_mp")
    _record_three_findings(seeded_store, logger, run_id)
    logger.end_run(run_id, n_assertions=4, n_pairs_gated=3, n_pairs_judged=3)
    report = render_report(seeded_store, logger, run_id=run_id)
    assert "Multi-document conditional contradictions" not in report


def test_report_includes_multi_party_section_when_findings_exist(
    seeded_store: AssertionStore,
) -> None:
    _populate_three_doc_fixture(seeded_store)
    logger = AuditLogger(seeded_store)
    run_id = logger.begin_run(run_id="run_mp")
    _record_three_findings(seeded_store, logger, run_id)
    # Add a multi-party finding by hand to mimic the F4 pass.
    a_list = list(seeded_store.iter_assertions())
    a_ids = [a.assertion_id for a in a_list][:3]
    d_ids = [a.doc_id for a in a_list][:3]
    logger.record_multi_party_finding(
        run_id,
        assertion_ids=a_ids,
        doc_ids=d_ids,
        triangle_edge_scores=[
            (a_ids[0], a_ids[1], 0.82),
            (a_ids[0], a_ids[2], 0.74),
            (a_ids[1], a_ids[2], 0.91),
        ],
        judge_verdict="multi_party_contradiction",
        judge_confidence=0.88,
        judge_rationale="A ∧ B ⇒ ¬C — chained-attribute conflict.",
        evidence_spans=["four weeks", "two weeks"],
    )
    logger.end_run(run_id, n_assertions=4, n_pairs_gated=3, n_pairs_judged=3)
    report = render_report(seeded_store, logger, run_id=run_id)
    assert "## Multi-document conditional contradictions" in report
    assert "chained-attribute conflict" in report
    # Pair section still present.
    assert "Opposite revenue signs" in report


def test_report_multi_party_excludes_uncertain(seeded_store: AssertionStore) -> None:
    """Uncertain triangles live in the audit DB but are not rendered."""
    _populate_three_doc_fixture(seeded_store)
    logger = AuditLogger(seeded_store)
    run_id = logger.begin_run(run_id="run_mp_uncertain")
    _record_three_findings(seeded_store, logger, run_id)
    a_list = list(seeded_store.iter_assertions())
    a_ids = [a.assertion_id for a in a_list][:3]
    d_ids = [a.doc_id for a in a_list][:3]
    logger.record_multi_party_finding(
        run_id,
        assertion_ids=a_ids,
        doc_ids=d_ids,
        judge_verdict="uncertain",
        judge_confidence=0.3,
        judge_rationale="scope unclear",
    )
    logger.end_run(run_id, n_assertions=4, n_pairs_gated=3, n_pairs_judged=3)
    report = render_report(seeded_store, logger, run_id=run_id)
    assert "Multi-document conditional contradictions" not in report
