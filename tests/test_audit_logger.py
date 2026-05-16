"""Tests for the SQLite-backed audit logger."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from consistency_checker.audit.logger import AuditLogger, Finding
from consistency_checker.check.gate import AllPairsGate, CandidatePair
from consistency_checker.check.llm_judge import FixtureJudge, JudgeVerdict
from consistency_checker.check.nli_checker import FixtureNliChecker, NliResult
from consistency_checker.extract.schema import Assertion, Document
from consistency_checker.index.assertion_store import AssertionStore

# --- run_status (migration 0004) -------------------------------------------


def test_run_status_column_exists(store: AssertionStore) -> None:
    """Migration 0004 adds run_status with a default of 'running'."""
    cols = {row[1] for row in store._conn.execute("PRAGMA table_info(pipeline_runs)")}
    assert "run_status" in cols


def test_begin_run_defaults_to_running(store: AssertionStore) -> None:
    logger = AuditLogger(store)
    rid = logger.begin_run(run_id="r-running-default")
    run = logger.get_run(rid)
    assert run is not None
    assert run.run_status == "running"


def test_begin_run_accepts_pending_status(store: AssertionStore) -> None:
    logger = AuditLogger(store)
    rid = logger.begin_run(run_id="r-pending", run_status="pending")
    run = logger.get_run(rid)
    assert run is not None
    assert run.run_status == "pending"


def test_update_run_status_flips_value(store: AssertionStore) -> None:
    logger = AuditLogger(store)
    rid = logger.begin_run(run_id="r-flip", run_status="pending")
    logger.update_run_status(rid, "running")
    assert logger.get_run(rid).run_status == "running"  # type: ignore[union-attr]
    logger.update_run_status(rid, "failed")
    assert logger.get_run(rid).run_status == "failed"  # type: ignore[union-attr]


def test_end_run_marks_status_done(store: AssertionStore) -> None:
    logger = AuditLogger(store)
    rid = logger.begin_run(run_id="r-done")
    logger.end_run(rid, n_assertions=1, n_pairs_gated=0, n_pairs_judged=0, n_findings=0)
    run = logger.get_run(rid)
    assert run is not None
    assert run.run_status == "done"


def _seed_two_docs(store: AssertionStore) -> tuple[Document, Document]:
    doc_a = Document.from_content("Body A.", source_path="a.txt")
    doc_b = Document.from_content("Body B.", source_path="b.txt")
    store.add_document(doc_a)
    store.add_document(doc_b)
    a1 = Assertion.build(doc_a.doc_id, "Revenue grew 12% in fiscal 2025.")
    b1 = Assertion.build(doc_b.doc_id, "Revenue declined 5% in fiscal 2025.")
    store.add_assertions([a1, b1])
    return doc_a, doc_b


@pytest.fixture
def store(tmp_path: Path) -> AssertionStore:
    s = AssertionStore(tmp_path / "store.db")
    s.migrate()
    return s


# --- migration -------------------------------------------------------------


def test_audit_migration_creates_tables(tmp_path: Path) -> None:
    s = AssertionStore(tmp_path / "x.db")
    s.migrate()
    conn = sqlite3.connect(tmp_path / "x.db")
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"pipeline_runs", "findings"} <= tables


# --- begin / end run --------------------------------------------------------


def test_begin_run_returns_id(store: AssertionStore) -> None:
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    assert isinstance(run_id, str) and run_id
    run = logger.get_run(run_id)
    assert run is not None
    assert run.run_id == run_id
    assert run.started_at is not None
    assert run.finished_at is None


def test_begin_run_persists_config(store: AssertionStore) -> None:
    logger = AuditLogger(store)
    cfg = {"top_k": 5, "threshold": 0.5, "model": "claude-sonnet-4-6"}
    run_id = logger.begin_run(config=cfg, notes="smoke run")
    run = logger.get_run(run_id)
    assert run is not None
    assert run.config_json is not None
    assert json.loads(run.config_json) == cfg
    assert run.notes == "smoke run"


def test_end_run_sets_totals_and_finished_at(store: AssertionStore) -> None:
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    logger.end_run(run_id, n_assertions=10, n_pairs_gated=15, n_pairs_judged=8, n_findings=2)
    run = logger.get_run(run_id)
    assert run is not None
    assert run.finished_at is not None
    assert run.n_assertions == 10
    assert run.n_pairs_gated == 15
    assert run.n_pairs_judged == 8
    assert run.n_findings == 2


def test_end_run_counts_findings_automatically(store: AssertionStore) -> None:
    logger = AuditLogger(store)
    _seed_two_docs(store)
    a, b = list(store.iter_assertions())
    run_id = logger.begin_run()
    logger.record_finding(
        run_id,
        candidate=CandidatePair(a=a, b=b, score=0.9),
        nli=NliResult.from_scores(p_contradiction=0.8, p_entailment=0.05, p_neutral=0.15),
        verdict=JudgeVerdict(
            assertion_a_id=a.assertion_id,
            assertion_b_id=b.assertion_id,
            verdict="contradiction",
            confidence=0.85,
            rationale="opposite signs",
        ),
    )
    logger.end_run(run_id, n_assertions=2, n_pairs_gated=1, n_pairs_judged=1)
    run = logger.get_run(run_id)
    assert run is not None
    assert run.n_findings == 1


# --- record_finding --------------------------------------------------------


def test_record_finding_persists_all_fields(store: AssertionStore) -> None:
    logger = AuditLogger(store)
    _seed_two_docs(store)
    a, b = list(store.iter_assertions())
    run_id = logger.begin_run()

    candidate = CandidatePair(a=a, b=b, score=0.91)
    nli = NliResult.from_scores(p_contradiction=0.82, p_entailment=0.05, p_neutral=0.13)
    verdict = JudgeVerdict(
        assertion_a_id=a.assertion_id,
        assertion_b_id=b.assertion_id,
        verdict="contradiction",
        confidence=0.9,
        rationale="opposite signs at the same scope",
        evidence_spans=["grew 12%", "declined 5%"],
    )
    finding_id = logger.record_finding(run_id, candidate=candidate, nli=nli, verdict=verdict)
    fetched = logger.get_finding(finding_id)
    assert fetched is not None
    assert fetched.assertion_a_id == a.assertion_id
    assert fetched.assertion_b_id == b.assertion_id
    assert fetched.gate_score == pytest.approx(0.91)
    assert fetched.nli_label == "contradiction"
    assert fetched.nli_p_contradiction == pytest.approx(0.82)
    assert fetched.judge_verdict == "contradiction"
    assert fetched.judge_confidence == pytest.approx(0.9)
    assert fetched.evidence_spans == ["grew 12%", "declined 5%"]


def test_record_finding_is_idempotent_within_run(store: AssertionStore) -> None:
    """Re-recording the same pair in the same run should not duplicate rows."""
    logger = AuditLogger(store)
    _seed_two_docs(store)
    a, b = list(store.iter_assertions())
    run_id = logger.begin_run()
    candidate = CandidatePair(a=a, b=b, score=0.5)
    verdict = JudgeVerdict(
        assertion_a_id=a.assertion_id,
        assertion_b_id=b.assertion_id,
        verdict="uncertain",
        confidence=0.0,
        rationale="repeated",
    )
    logger.record_finding(run_id, candidate=candidate, nli=None, verdict=verdict)
    logger.record_finding(run_id, candidate=candidate, nli=None, verdict=verdict)
    assert len(list(logger.iter_findings(run_id=run_id))) == 1


def test_record_finding_handles_no_nli(store: AssertionStore) -> None:
    logger = AuditLogger(store)
    _seed_two_docs(store)
    a, b = list(store.iter_assertions())
    run_id = logger.begin_run()
    verdict = JudgeVerdict(
        assertion_a_id=a.assertion_id,
        assertion_b_id=b.assertion_id,
        verdict="not_contradiction",
        confidence=0.5,
        rationale="no nli stage",
    )
    fid = logger.record_finding(
        run_id, candidate=CandidatePair(a=a, b=b, score=1.0), nli=None, verdict=verdict
    )
    fetched = logger.get_finding(fid)
    assert fetched is not None
    assert fetched.nli_label is None
    assert fetched.nli_p_contradiction is None


def test_record_finding_requires_existing_assertions(store: AssertionStore) -> None:
    """Foreign key check on assertions table — orphan finding must fail."""
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    orphan_a = Assertion.build("nonexistent", "x")
    orphan_b = Assertion.build("nonexistent", "y")
    candidate = CandidatePair(a=orphan_a, b=orphan_b, score=0.5)
    verdict = JudgeVerdict(
        assertion_a_id=orphan_a.assertion_id,
        assertion_b_id=orphan_b.assertion_id,
        verdict="uncertain",
        confidence=0.0,
        rationale="r",
    )
    with pytest.raises(sqlite3.IntegrityError):
        logger.record_finding(run_id, candidate=candidate, nli=None, verdict=verdict)


# --- iter_findings filtering ----------------------------------------------


def test_most_recent_run_returns_latest_started(store: AssertionStore) -> None:
    """The CLI's `report --run` default uses this; latest-started wins."""
    logger = AuditLogger(store)
    assert logger.most_recent_run() is None

    first = logger.begin_run(run_id="first_run")
    second = logger.begin_run(run_id="second_run")
    recent = logger.most_recent_run()
    assert recent is not None
    assert recent.run_id in {first, second}
    # `started_at` is at second granularity; assert on the column that's reliably ordered.
    if recent.run_id == first:
        # tie-break favours later run_id alphabetically per the SQL ORDER BY clause
        assert second < first
    else:
        assert recent.run_id == second


def test_iter_findings_filters_by_run(store: AssertionStore) -> None:
    logger = AuditLogger(store)
    _seed_two_docs(store)
    a, b = list(store.iter_assertions())
    run1 = logger.begin_run()
    run2 = logger.begin_run()
    base_verdict = JudgeVerdict(
        assertion_a_id=a.assertion_id,
        assertion_b_id=b.assertion_id,
        verdict="contradiction",
        confidence=0.7,
        rationale="r",
    )
    logger.record_finding(
        run1, candidate=CandidatePair(a=a, b=b, score=1.0), nli=None, verdict=base_verdict
    )
    logger.record_finding(
        run2, candidate=CandidatePair(a=a, b=b, score=1.0), nli=None, verdict=base_verdict
    )
    only_run1 = list(logger.iter_findings(run_id=run1))
    assert len(only_run1) == 1
    assert only_run1[0].run_id == run1


def test_iter_findings_filters_by_verdict(store: AssertionStore) -> None:
    logger = AuditLogger(store)
    _seed_two_docs(store)
    a, b = list(store.iter_assertions())
    run_id = logger.begin_run()
    # Two pairs would require more assertions; using the same pair with different
    # verdicts in different runs to keep idempotency intact.
    logger.record_finding(
        run_id,
        candidate=CandidatePair(a=a, b=b, score=1.0),
        nli=None,
        verdict=JudgeVerdict(
            assertion_a_id=a.assertion_id,
            assertion_b_id=b.assertion_id,
            verdict="contradiction",
            confidence=0.9,
            rationale="r",
        ),
    )
    contradictions = list(logger.iter_findings(verdict="contradiction"))
    assert len(contradictions) == 1
    uncertain = list(logger.iter_findings(verdict="uncertain"))
    assert uncertain == []


# --- end-to-end mini pipeline ----------------------------------------------


def test_end_to_end_mini_pipeline_records_findings(store: AssertionStore) -> None:
    """Drive AllPairsGate + FixtureNliChecker + FixtureJudge → AuditLogger."""
    _seed_two_docs(store)
    a, b = list(store.iter_assertions())

    nli_results = {
        (a.assertion_text, b.assertion_text): NliResult.from_scores(
            p_contradiction=0.78, p_entailment=0.07, p_neutral=0.15
        ),
        (b.assertion_text, a.assertion_text): NliResult.from_scores(
            p_contradiction=0.71, p_entailment=0.10, p_neutral=0.19
        ),
    }
    nli_checker = FixtureNliChecker(nli_results)

    canonical_key = (
        min(a.assertion_id, b.assertion_id),
        max(a.assertion_id, b.assertion_id),
    )
    judge_fixture = {
        canonical_key: JudgeVerdict(
            assertion_a_id=canonical_key[0],
            assertion_b_id=canonical_key[1],
            verdict="contradiction",
            confidence=0.9,
            rationale="opposing signs",
            evidence_spans=["grew 12%", "declined 5%"],
        )
    }
    judge = FixtureJudge(judge_fixture)

    logger = AuditLogger(store)
    run_id = logger.begin_run(config={"pipeline": "end-to-end-mini"})

    pairs_judged = 0
    for pair in AllPairsGate().candidates(store):
        nli_result = nli_checker.score(pair.a.assertion_text, pair.b.assertion_text)
        verdict = judge.judge(pair.a, pair.b)
        logger.record_finding(run_id, candidate=pair, nli=nli_result, verdict=verdict)
        pairs_judged += 1

    logger.end_run(
        run_id,
        n_assertions=2,
        n_pairs_gated=pairs_judged,
        n_pairs_judged=pairs_judged,
    )

    run = logger.get_run(run_id)
    assert run is not None
    assert run.n_pairs_judged == 1
    assert run.n_findings == 1

    findings: list[Finding] = list(logger.iter_findings(run_id=run_id))
    assert len(findings) == 1
    only = findings[0]
    assert only.judge_verdict == "contradiction"
    assert only.evidence_spans == ["grew 12%", "declined 5%"]


def test_findings_are_replayable_from_logged_inputs(store: AssertionStore) -> None:
    """A logged finding must carry enough state to reconstruct judge inputs."""
    logger = AuditLogger(store)
    _seed_two_docs(store)
    a, b = list(store.iter_assertions())
    run_id = logger.begin_run()
    fid = logger.record_finding(
        run_id,
        candidate=CandidatePair(a=a, b=b, score=0.88),
        nli=NliResult.from_scores(p_contradiction=0.6, p_entailment=0.2, p_neutral=0.2),
        verdict=JudgeVerdict(
            assertion_a_id=a.assertion_id,
            assertion_b_id=b.assertion_id,
            verdict="contradiction",
            confidence=0.7,
            rationale="r",
        ),
    )
    fetched = logger.get_finding(fid)
    assert fetched is not None

    # Use the logged ids to pull the source assertions back from the store; this
    # is exactly what the report generator (Step 13) will do.
    reloaded_a = store.get_assertion(fetched.assertion_a_id)
    reloaded_b = store.get_assertion(fetched.assertion_b_id)
    assert reloaded_a is not None
    assert reloaded_b is not None
    assert reloaded_a.assertion_text == a.assertion_text
    assert reloaded_b.assertion_text == b.assertion_text


def test_record_definition_finding_writes_detector_type(tmp_path: Path) -> None:
    from consistency_checker.check.definition_checker import (
        DefinitionFinding,
        DefinitionPair,
    )
    from consistency_checker.check.definition_judge import DefinitionJudgeVerdict
    from consistency_checker.extract.schema import Document

    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    store.add_document(Document(doc_id="docA", source_path="/A.txt"))
    store.add_document(Document(doc_id="docB", source_path="/B.txt"))
    a = Assertion.build(
        "docA", '"X" means foo.', kind="definition", term="X", definition_text="foo"
    )
    b = Assertion.build(
        "docB", '"X" means bar.', kind="definition", term="X", definition_text="bar"
    )
    store.add_assertions([a, b])
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    finding = DefinitionFinding(
        pair=DefinitionPair(a=a, b=b, canonical_term="x"),
        verdict=DefinitionJudgeVerdict(
            assertion_a_id=min(a.assertion_id, b.assertion_id),
            assertion_b_id=max(a.assertion_id, b.assertion_id),
            verdict="definition_divergent",
            confidence=0.9,
            rationale="scope shift",
            evidence_spans=["foo", "bar"],
        ),
    )
    logger.record_definition_finding(run_id, finding=finding)
    rows = store._conn.execute(
        "SELECT detector_type, judge_verdict, gate_score, nli_label, evidence_spans_json "
        "FROM findings WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["detector_type"] == "definition_inconsistency"
    assert rows[0]["judge_verdict"] == "definition_divergent"
    assert rows[0]["gate_score"] is None
    assert rows[0]["nli_label"] is None
    import json as _json

    assert _json.loads(rows[0]["evidence_spans_json"]) == ["foo", "bar"]
    store.close()


def test_record_definition_finding_idempotent(tmp_path: Path) -> None:
    from consistency_checker.check.definition_checker import (
        DefinitionFinding,
        DefinitionPair,
    )
    from consistency_checker.check.definition_judge import DefinitionJudgeVerdict
    from consistency_checker.extract.schema import Document

    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    store.add_document(Document(doc_id="docA", source_path="/A.txt"))
    store.add_document(Document(doc_id="docB", source_path="/B.txt"))
    a = Assertion.build(
        "docA", '"X" means foo.', kind="definition", term="X", definition_text="foo"
    )
    b = Assertion.build(
        "docB", '"X" means bar.', kind="definition", term="X", definition_text="bar"
    )
    store.add_assertions([a, b])
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    finding = DefinitionFinding(
        pair=DefinitionPair(a=a, b=b, canonical_term="x"),
        verdict=DefinitionJudgeVerdict(
            assertion_a_id=min(a.assertion_id, b.assertion_id),
            assertion_b_id=max(a.assertion_id, b.assertion_id),
            verdict="definition_divergent",
            confidence=0.9,
            rationale="scope shift",
            evidence_spans=[],
        ),
    )
    fid1 = logger.record_definition_finding(run_id, finding=finding)
    fid2 = logger.record_definition_finding(run_id, finding=finding)
    assert fid1 == fid2
    rows = store._conn.execute(
        "SELECT COUNT(*) FROM findings WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert rows[0] == 1
    store.close()


def test_set_reviewer_verdict_inserts_row(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    logger = AuditLogger(store)
    logger.set_reviewer_verdict(
        pair_key="a:b",
        detector_type="contradiction",
        verdict="confirmed",
    )
    rows = store._conn.execute(
        "SELECT pair_key, detector_type, verdict, note FROM reviewer_verdicts"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["pair_key"] == "a:b"
    assert rows[0]["detector_type"] == "contradiction"
    assert rows[0]["verdict"] == "confirmed"
    assert rows[0]["note"] is None
    store.close()


def test_set_reviewer_verdict_upserts_on_conflict(tmp_path: Path) -> None:
    """Second set with different verdict overwrites; set_at refreshes."""
    import time
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    logger = AuditLogger(store)
    logger.set_reviewer_verdict(
        pair_key="a:b", detector_type="contradiction", verdict="confirmed", note="initial"
    )
    first = store._conn.execute(
        "SELECT verdict, note, set_at FROM reviewer_verdicts"
    ).fetchone()
    time.sleep(1.1)
    logger.set_reviewer_verdict(
        pair_key="a:b", detector_type="contradiction", verdict="false_positive", note="changed"
    )
    second = store._conn.execute(
        "SELECT verdict, note, set_at FROM reviewer_verdicts"
    ).fetchone()
    assert second["verdict"] == "false_positive"
    assert second["note"] == "changed"
    assert second["set_at"] > first["set_at"]
    store.close()


def test_set_reviewer_verdict_same_pair_different_detector_coexist(tmp_path: Path) -> None:
    """Composite PK lets a single content pair carry independent verdicts per detector."""
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    logger = AuditLogger(store)
    logger.set_reviewer_verdict(
        pair_key="a:b", detector_type="contradiction", verdict="confirmed"
    )
    logger.set_reviewer_verdict(
        pair_key="a:b", detector_type="definition_inconsistency", verdict="false_positive"
    )
    rows = store._conn.execute(
        "SELECT detector_type, verdict FROM reviewer_verdicts ORDER BY detector_type"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["detector_type"] == "contradiction"
    assert rows[0]["verdict"] == "confirmed"
    assert rows[1]["detector_type"] == "definition_inconsistency"
    assert rows[1]["verdict"] == "false_positive"
    store.close()


def test_delete_reviewer_verdict_removes_row(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    logger = AuditLogger(store)
    logger.set_reviewer_verdict(
        pair_key="a:b", detector_type="contradiction", verdict="confirmed"
    )
    logger.delete_reviewer_verdict(pair_key="a:b", detector_type="contradiction")
    rows = store._conn.execute("SELECT COUNT(*) FROM reviewer_verdicts").fetchone()
    assert rows[0] == 0
    store.close()


def test_delete_reviewer_verdict_targets_only_matching_detector(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    logger = AuditLogger(store)
    logger.set_reviewer_verdict(
        pair_key="a:b", detector_type="contradiction", verdict="confirmed"
    )
    logger.set_reviewer_verdict(
        pair_key="a:b", detector_type="definition_inconsistency", verdict="false_positive"
    )
    logger.delete_reviewer_verdict(pair_key="a:b", detector_type="contradiction")
    rows = store._conn.execute(
        "SELECT detector_type FROM reviewer_verdicts"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["detector_type"] == "definition_inconsistency"
    store.close()
