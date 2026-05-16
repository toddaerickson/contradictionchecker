"""Tests for database migrations."""

from __future__ import annotations

from pathlib import Path

from consistency_checker.index.assertion_store import AssertionStore


def test_migration_0007_adds_kind_columns(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    cols = {
        row["name"]: row
        for row in store._conn.execute("PRAGMA table_info(assertions)").fetchall()
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
        row["name"]: row
        for row in store._conn.execute("PRAGMA table_info(findings)").fetchall()
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
