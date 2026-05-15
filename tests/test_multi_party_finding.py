"""Tests for the multi_party_findings table + AuditLogger methods (F1)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from consistency_checker.audit.logger import AuditLogger, MultiPartyFinding
from consistency_checker.extract.schema import Assertion, Document
from consistency_checker.index.assertion_store import AssertionStore


@pytest.fixture
def store(tmp_path: Path) -> AssertionStore:
    s = AssertionStore(tmp_path / "store.db")
    s.migrate()
    return s


def _seed_three_docs(store: AssertionStore) -> list[Assertion]:
    docs = [Document.from_content(f"Body {i}.", source_path=f"d{i}.txt") for i in range(3)]
    for d in docs:
        store.add_document(d)
    assertions = [
        Assertion.build(docs[0].doc_id, "All employees get four weeks vacation."),
        Assertion.build(docs[1].doc_id, "Engineers are employees."),
        Assertion.build(docs[2].doc_id, "Engineers get two weeks vacation."),
    ]
    store.add_assertions(assertions)
    return assertions


# --- migration -------------------------------------------------------------


def test_multi_party_migration_creates_table(tmp_path: Path) -> None:
    s = AssertionStore(tmp_path / "x.db")
    s.migrate()
    conn = sqlite3.connect(tmp_path / "x.db")
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "multi_party_findings" in tables
    indexes = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='multi_party_findings'"
        )
    }
    assert {"idx_mpf_run", "idx_mpf_verdict"} <= indexes


def test_migrate_picks_up_0003_alongside_existing_migrations(tmp_path: Path) -> None:
    s = AssertionStore(tmp_path / "x.db")
    applied = s.migrate()
    assert 1 in applied  # 0001_init
    assert 2 in applied  # 0002_audit
    assert 3 in applied  # 0003_multi_party
    assert 5 in applied  # 0005_multi_party_finding_assertions
    # idempotent
    assert s.migrate() == []


def test_multi_party_finding_assertions_table_exists(tmp_path: Path) -> None:
    s = AssertionStore(tmp_path / "x.db")
    s.migrate()
    conn = sqlite3.connect(tmp_path / "x.db")
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "multi_party_finding_assertions" in tables
    indexes = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='multi_party_finding_assertions'"
        )
    }
    assert {"idx_mpfa_finding", "idx_mpfa_assertion"} <= indexes


# --- record_multi_party_finding -------------------------------------------


def test_record_multi_party_finding_round_trip(store: AssertionStore) -> None:
    assertions = _seed_three_docs(store)
    logger = AuditLogger(store)
    run_id = logger.begin_run()

    edges: list[tuple[str, str, float]] = [
        (assertions[0].assertion_id, assertions[1].assertion_id, 0.82),
        (assertions[1].assertion_id, assertions[2].assertion_id, 0.91),
        (assertions[0].assertion_id, assertions[2].assertion_id, 0.74),
    ]
    fid = logger.record_multi_party_finding(
        run_id,
        assertion_ids=[a.assertion_id for a in assertions],
        doc_ids=[a.doc_id for a in assertions],
        triangle_edge_scores=edges,
        judge_verdict="multi_party_contradiction",
        judge_confidence=0.88,
        judge_rationale="A ∧ B ⇒ ¬C — engineers are employees so the 4-week claim contradicts the 2-week claim.",
        evidence_spans=["four weeks", "two weeks"],
    )

    fetched = logger.get_multi_party_finding(fid)
    assert fetched is not None
    assert fetched.run_id == run_id
    assert fetched.assertion_ids == sorted(a.assertion_id for a in assertions)
    assert fetched.doc_ids == [a.doc_id for a in assertions]
    assert fetched.judge_verdict == "multi_party_contradiction"
    assert fetched.judge_confidence == pytest.approx(0.88)
    assert fetched.evidence_spans == ["four weeks", "two weeks"]
    # Edge scores round-trip with float values.
    assert len(fetched.triangle_edge_scores) == 3
    for orig, recon in zip(edges, fetched.triangle_edge_scores, strict=True):
        assert orig[0] == recon[0]
        assert orig[1] == recon[1]
        assert orig[2] == pytest.approx(recon[2])


def test_record_multi_party_is_idempotent_per_run(store: AssertionStore) -> None:
    """Same run + same triangle id-set → replaced, not duplicated."""
    assertions = _seed_three_docs(store)
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    common = {
        "assertion_ids": [a.assertion_id for a in assertions],
        "doc_ids": [a.doc_id for a in assertions],
        "triangle_edge_scores": None,
        "judge_verdict": "uncertain",
    }
    first = logger.record_multi_party_finding(run_id, **common)  # type: ignore[arg-type]
    second = logger.record_multi_party_finding(run_id, **common)  # type: ignore[arg-type]
    assert first == second
    assert len(list(logger.iter_multi_party_findings(run_id=run_id))) == 1


def test_record_multi_party_rejects_under_three_assertions(store: AssertionStore) -> None:
    assertions = _seed_three_docs(store)
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    with pytest.raises(ValueError, match="at least 3 assertion ids"):
        logger.record_multi_party_finding(
            run_id,
            assertion_ids=[a.assertion_id for a in assertions[:2]],
            doc_ids=[a.doc_id for a in assertions[:2]],
            judge_verdict="uncertain",
        )


def test_record_multi_party_rejects_single_document_triangle(
    store: AssertionStore,
) -> None:
    """A triangle whose three assertions all come from one doc is not multi-party."""
    doc = Document.from_content("Body.", source_path="d.txt")
    store.add_document(doc)
    a = Assertion.build(doc.doc_id, "alpha")
    b = Assertion.build(doc.doc_id, "beta")
    c = Assertion.build(doc.doc_id, "gamma")
    store.add_assertions([a, b, c])
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    with pytest.raises(ValueError, match=">= 2 distinct doc ids"):
        logger.record_multi_party_finding(
            run_id,
            assertion_ids=[a.assertion_id, b.assertion_id, c.assertion_id],
            doc_ids=[doc.doc_id, doc.doc_id, doc.doc_id],
            judge_verdict="multi_party_contradiction",
        )


def test_record_multi_party_writes_assertion_join_rows(store: AssertionStore) -> None:
    """Each recorded triangle inserts one row per assertion into the FK side table."""
    assertions = _seed_three_docs(store)
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    fid = logger.record_multi_party_finding(
        run_id,
        assertion_ids=[a.assertion_id for a in assertions],
        doc_ids=[a.doc_id for a in assertions],
        judge_verdict="multi_party_contradiction",
    )
    rows = store._conn.execute(
        "SELECT assertion_id, position FROM multi_party_finding_assertions "
        "WHERE finding_id = ? ORDER BY position",
        (fid,),
    ).fetchall()
    assert [r["position"] for r in rows] == [0, 1, 2]
    assert sorted(r["assertion_id"] for r in rows) == sorted(a.assertion_id for a in assertions)


def test_multi_party_idempotent_record_replaces_join_rows(store: AssertionStore) -> None:
    """Re-recording the same triangle doesn't leave stale join rows behind."""
    assertions = _seed_three_docs(store)
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    payload = {
        "assertion_ids": [a.assertion_id for a in assertions],
        "doc_ids": [a.doc_id for a in assertions],
        "judge_verdict": "uncertain",
    }
    fid = logger.record_multi_party_finding(run_id, **payload)  # type: ignore[arg-type]
    logger.record_multi_party_finding(run_id, **payload)  # type: ignore[arg-type]
    count = store._conn.execute(
        "SELECT COUNT(*) FROM multi_party_finding_assertions WHERE finding_id = ?",
        (fid,),
    ).fetchone()[0]
    assert count == 3


def test_multi_party_join_rows_cascade_when_finding_deleted(
    store: AssertionStore,
) -> None:
    """Deleting the parent multi_party_findings row removes the join rows too."""
    assertions = _seed_three_docs(store)
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    fid = logger.record_multi_party_finding(
        run_id,
        assertion_ids=[a.assertion_id for a in assertions],
        doc_ids=[a.doc_id for a in assertions],
        judge_verdict="uncertain",
    )
    store._conn.execute("DELETE FROM multi_party_findings WHERE finding_id = ?", (fid,))
    store._conn.commit()
    count = store._conn.execute(
        "SELECT COUNT(*) FROM multi_party_finding_assertions WHERE finding_id = ?",
        (fid,),
    ).fetchone()[0]
    assert count == 0


def test_end_run_autocount_includes_multi_party_contradictions(
    store: AssertionStore,
) -> None:
    """Counter from end_run(n_findings=None) sums pair + multi-party contradictions."""
    from consistency_checker.check.gate import CandidatePair
    from consistency_checker.check.llm_judge import JudgeVerdict

    assertions = _seed_three_docs(store)
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    # One pair contradiction.
    pair = CandidatePair(a=assertions[0], b=assertions[1], score=0.9)
    logger.record_finding(
        run_id,
        candidate=pair,
        nli=None,
        verdict=JudgeVerdict(
            assertion_a_id=assertions[0].assertion_id,
            assertion_b_id=assertions[1].assertion_id,
            verdict="contradiction",
            confidence=0.9,
            rationale="x",
        ),
    )
    # One multi-party contradiction.
    logger.record_multi_party_finding(
        run_id,
        assertion_ids=[a.assertion_id for a in assertions],
        doc_ids=[a.doc_id for a in assertions],
        judge_verdict="multi_party_contradiction",
    )
    logger.end_run(run_id)  # n_findings=None → auto-count
    run = logger.get_run(run_id)
    assert run is not None
    assert run.n_findings == 2


def test_record_multi_party_cascade_deletes_with_run(store: AssertionStore) -> None:
    """Deleting a pipeline_run row removes its multi_party_findings."""
    assertions = _seed_three_docs(store)
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    logger.record_multi_party_finding(
        run_id,
        assertion_ids=[a.assertion_id for a in assertions],
        doc_ids=[a.doc_id for a in assertions],
        judge_verdict="uncertain",
    )
    assert len(list(logger.iter_multi_party_findings(run_id=run_id))) == 1
    store._conn.execute("DELETE FROM pipeline_runs WHERE run_id = ?", (run_id,))
    store._conn.commit()
    assert list(logger.iter_multi_party_findings(run_id=run_id)) == []


# --- iter_multi_party_findings filters -------------------------------------


def test_iter_multi_party_findings_filters_by_run(store: AssertionStore) -> None:
    assertions = _seed_three_docs(store)
    logger = AuditLogger(store)
    run_a = logger.begin_run(run_id="run_a")
    run_b = logger.begin_run(run_id="run_b")
    payload = {
        "assertion_ids": [a.assertion_id for a in assertions],
        "doc_ids": [a.doc_id for a in assertions],
        "judge_verdict": "multi_party_contradiction",
    }
    logger.record_multi_party_finding(run_a, **payload)  # type: ignore[arg-type]
    logger.record_multi_party_finding(run_b, **payload)  # type: ignore[arg-type]
    only_a = list(logger.iter_multi_party_findings(run_id=run_a))
    assert len(only_a) == 1
    assert only_a[0].run_id == run_a


def test_iter_multi_party_findings_filters_by_verdict(store: AssertionStore) -> None:
    assertions = _seed_three_docs(store)
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    logger.record_multi_party_finding(
        run_id,
        assertion_ids=[a.assertion_id for a in assertions],
        doc_ids=[a.doc_id for a in assertions],
        judge_verdict="multi_party_contradiction",
    )
    contradiction = list(logger.iter_multi_party_findings(verdict="multi_party_contradiction"))
    uncertain = list(logger.iter_multi_party_findings(verdict="uncertain"))
    assert len(contradiction) == 1
    assert uncertain == []


def test_multi_party_finding_dataclass_is_frozen() -> None:
    f = MultiPartyFinding(
        finding_id="x",
        run_id="r",
        assertion_ids=["a", "b", "c"],
        doc_ids=["d1", "d2", "d3"],
        triangle_edge_scores=[],
        judge_verdict="uncertain",
        judge_confidence=None,
        judge_rationale=None,
    )
    with pytest.raises(AttributeError):
        f.run_id = "other"  # type: ignore[misc]
