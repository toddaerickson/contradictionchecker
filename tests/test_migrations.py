"""Tests for database migrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from consistency_checker.index.assertion_store import AssertionStore


def test_migration_0007_adds_kind_columns(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    cols = {
        row["name"]: row for row in store._conn.execute("PRAGMA table_info(assertions)").fetchall()
    }
    assert "kind" in cols
    assert cols["kind"]["dflt_value"] == "'claim'"
    assert cols["kind"]["notnull"] == 1
    assert "term" in cols
    assert "definition_text" in cols
    idx = {
        row["name"]
        for row in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='assertions'"
        ).fetchall()
    }
    assert "idx_assertions_kind" in idx
    assert "idx_assertions_term" in idx
    store.close()


def test_migration_0007_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    AssertionStore(db).migrate()
    store = AssertionStore(db)
    applied = store.migrate()
    assert applied == []
    store.close()


def test_migration_0008_adds_detector_type(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    cols = {
        row["name"]: row for row in store._conn.execute("PRAGMA table_info(findings)").fetchall()
    }
    assert "detector_type" in cols
    assert cols["detector_type"]["dflt_value"] == "'contradiction'"
    assert cols["detector_type"]["notnull"] == 1
    idx = {
        row["name"]
        for row in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='findings'"
        ).fetchall()
    }
    assert "idx_findings_detector" in idx
    store.close()


def test_migration_0009_adds_reviewer_verdicts_table(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    cols = {
        row["name"]: row
        for row in store._conn.execute("PRAGMA table_info(reviewer_verdicts)").fetchall()
    }
    assert set(cols.keys()) == {"pair_key", "detector_type", "verdict", "set_at", "note"}
    # Composite PK: both pair_key AND detector_type carry pk > 0
    assert cols["pair_key"]["pk"] >= 1
    assert cols["detector_type"]["pk"] >= 1
    # NOT NULL on the four required columns
    for c in ("pair_key", "detector_type", "verdict", "set_at"):
        assert cols[c]["notnull"] == 1
    # note is nullable
    assert cols["note"]["notnull"] == 0
    store.close()


def test_migration_0009_check_constraints_reject_bogus_values(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    with pytest.raises(sqlite3.IntegrityError, match=r"CHECK"):
        store._conn.execute(
            "INSERT INTO reviewer_verdicts (pair_key, detector_type, verdict) "
            "VALUES ('a:b', 'contradiction', 'banana')"
        )
    with pytest.raises(sqlite3.IntegrityError, match=r"CHECK"):
        store._conn.execute(
            "INSERT INTO reviewer_verdicts (pair_key, detector_type, verdict) "
            "VALUES ('a:b', 'made_up_detector', 'confirmed')"
        )
    store.close()


def test_migration_0009_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    AssertionStore(db).migrate()
    store = AssertionStore(db)
    assert store.migrate() == []
    store.close()


def test_migration_0013_adds_org_columns_and_findings_suppressed(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    cols_docs = {r[1] for r in store._conn.execute("PRAGMA table_info(documents)")}
    cols_findings = {r[1] for r in store._conn.execute("PRAGMA table_info(findings)")}
    assert "org_label" in cols_docs
    assert "org_reason" in cols_docs
    assert "suppressed" in cols_findings
    store.close()


def test_migration_0014_adds_corpus_id_and_backfills_legacy(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    docs_cols = {r[1] for r in store._conn.execute("PRAGMA table_info(documents)")}
    pr_cols = {r[1] for r in store._conn.execute("PRAGMA table_info(pipeline_runs)")}
    assert "corpus_id" in docs_cols
    assert "corpus_id" in pr_cols
    # legacy auto-creation: only when there are NULLs to fix; fresh DB has none
    assert (
        store._conn.execute("SELECT COUNT(*) FROM corpora WHERE corpus_name='legacy'").fetchone()[0]
        == 0
    )
    store.close()


def test_migration_0014_creates_legacy_when_orphan_docs_exist(tmp_path: Path) -> None:
    # Build a DB at the pre-0014 schema by running migrations 0001..0013 only.
    db = tmp_path / "pre.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE corpora (corpus_id TEXT PRIMARY KEY, corpus_name TEXT UNIQUE,
            corpus_path TEXT, judge_provider TEXT, created_at TEXT, updated_at TEXT);
        CREATE TABLE documents (doc_id TEXT PRIMARY KEY, source_path TEXT);
        CREATE TABLE pipeline_runs (run_id TEXT PRIMARY KEY);
        CREATE TABLE findings (finding_id TEXT PRIMARY KEY, judge_confidence REAL);
        CREATE TABLE multi_party_findings (finding_id TEXT PRIMARY KEY, judge_confidence REAL);
        CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT);
        INSERT INTO documents (doc_id, source_path) VALUES ('d1', '/x.txt');
        INSERT INTO pipeline_runs (run_id) VALUES ('r1');
        """
    )
    for v in range(1, 14):
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version, applied_at) "
            "VALUES (?, datetime('now'))",
            (v,),
        )
    conn.commit()
    conn.close()

    store = AssertionStore(db)
    store.migrate()
    legacy_id = store._conn.execute(
        "SELECT corpus_id FROM corpora WHERE corpus_name='legacy'"
    ).fetchone()[0]
    assert legacy_id is not None
    assert (
        store._conn.execute("SELECT corpus_id FROM documents WHERE doc_id='d1'").fetchone()[0]
        == legacy_id
    )
    assert (
        store._conn.execute("SELECT corpus_id FROM pipeline_runs WHERE run_id='r1'").fetchone()[0]
        == legacy_id
    )
    store.close()
