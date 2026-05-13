"""Tests for the SQLite-backed assertion store."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from consistency_checker.extract.schema import Assertion, Document, hash_id
from consistency_checker.index.assertion_store import (
    DEFAULT_EXPORT_COLUMNS,
    AssertionStore,
)


def make_doc(content: str = "Alpha sentence.", source: str = "a.txt") -> Document:
    return Document.from_content(content, source_path=source)


def make_assertion(doc: Document, text: str, *, start: int = 0, end: int = 0) -> Assertion:
    return Assertion.build(doc.doc_id, text, char_start=start, char_end=end)


@pytest.fixture
def store(tmp_path: Path) -> AssertionStore:
    s = AssertionStore(tmp_path / "store" / "assertions.db")
    s.migrate()
    return s


def test_hash_id_is_deterministic() -> None:
    assert hash_id("alpha") == hash_id("alpha")
    assert hash_id("alpha") != hash_id("beta")
    assert len(hash_id("alpha")) == 16


def test_assertion_id_includes_doc_id() -> None:
    """Same text in two different documents must produce different assertion ids."""
    a1 = Assertion.build("doc_a", "Revenue grew 10%.")
    a2 = Assertion.build("doc_b", "Revenue grew 10%.")
    assert a1.assertion_id != a2.assertion_id


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    s = AssertionStore(tmp_path / "a.db")
    first = s.migrate()
    second = s.migrate()
    assert first  # at least one migration applied
    assert first == sorted(first)  # in version order
    assert second == []
    conn = sqlite3.connect(tmp_path / "a.db")
    versions = [r[0] for r in conn.execute("SELECT version FROM schema_migrations")]
    assert versions == first


def test_document_round_trip(store: AssertionStore) -> None:
    doc = Document.from_content(
        "Hello world.",
        source_path="hello.txt",
        title="Greeting",
        doc_date="2025-09-30",
    )
    store.add_document(doc)
    fetched = store.get_document(doc.doc_id)
    assert fetched is not None
    assert fetched.doc_id == doc.doc_id
    assert fetched.title == "Greeting"
    assert fetched.doc_date == "2025-09-30"
    assert fetched.ingested_at is not None


def test_assertion_round_trip_preserves_char_spans(store: AssertionStore) -> None:
    doc = make_doc("The cat sat on the mat.")
    store.add_document(doc)
    a = make_assertion(doc, "The cat sat on the mat.", start=0, end=23)
    store.add_assertion(a)

    fetched = store.get_assertion(a.assertion_id)
    assert fetched is not None
    assert fetched.assertion_text == a.assertion_text
    assert fetched.char_start == 0
    assert fetched.char_end == 23
    assert fetched.faiss_row is None
    assert fetched.embedded_at is None


def test_add_document_is_idempotent(store: AssertionStore) -> None:
    doc = make_doc()
    store.add_document(doc)
    store.add_document(doc)
    assert store.stats()["documents"] == 1


def test_add_assertion_is_idempotent(store: AssertionStore) -> None:
    doc = make_doc()
    store.add_document(doc)
    a = make_assertion(doc, "Hello.")
    store.add_assertion(a)
    store.add_assertion(a)
    assert store.stats()["assertions"] == 1


def test_assertion_requires_existing_document(store: AssertionStore) -> None:
    """Foreign key enforcement: cannot insert assertion for unknown doc_id."""
    a = Assertion.build("nonexistent_doc", "orphan claim")
    with pytest.raises(sqlite3.IntegrityError):
        store.add_assertion(a)


def test_iter_assertions_filters_by_doc(store: AssertionStore) -> None:
    doc_a = make_doc("Doc A content.", source="a.txt")
    doc_b = make_doc("Doc B content.", source="b.txt")
    store.add_document(doc_a)
    store.add_document(doc_b)
    a1 = make_assertion(doc_a, "Claim one in A.")
    a2 = make_assertion(doc_a, "Claim two in A.")
    b1 = make_assertion(doc_b, "Claim one in B.")
    store.add_assertions([a1, a2, b1])

    only_a = list(store.iter_assertions(doc_id=doc_a.doc_id))
    assert {a.assertion_id for a in only_a} == {a1.assertion_id, a2.assertion_id}

    all_three = list(store.iter_assertions())
    assert len(all_three) == 3


def test_attach_embeddings(store: AssertionStore) -> None:
    doc = make_doc()
    store.add_document(doc)
    a = make_assertion(doc, "Embedding target.")
    store.add_assertion(a)

    store.attach_embeddings([(a.assertion_id, 42)])
    fetched = store.get_assertion(a.assertion_id)
    assert fetched is not None
    assert fetched.faiss_row == 42
    assert fetched.embedded_at is not None
    assert store.stats()["embedded_assertions"] == 1


def test_export_csv_default_columns(store: AssertionStore, tmp_path: Path) -> None:
    """User-facing export: ``(doc_id, assertion_id, assertion_text)`` by default."""
    doc = make_doc()
    store.add_document(doc)
    a = make_assertion(doc, "Exported claim.")
    store.add_assertion(a)

    out = tmp_path / "out.csv"
    store.export_csv(out)
    rows = list(csv.reader(out.open()))
    assert rows[0] == list(DEFAULT_EXPORT_COLUMNS)
    assert rows[1] == [doc.doc_id, a.assertion_id, "Exported claim."]


def test_export_csv_custom_columns(store: AssertionStore, tmp_path: Path) -> None:
    doc = make_doc()
    store.add_document(doc)
    store.add_assertion(make_assertion(doc, "x", start=3, end=4))
    out = tmp_path / "c.csv"
    store.export_csv(out, columns=("assertion_id", "char_start", "char_end"))
    rows = list(csv.reader(out.open()))
    assert rows[0] == ["assertion_id", "char_start", "char_end"]
    assert rows[1][1:] == ["3", "4"]


def test_export_csv_rejects_unknown_column(store: AssertionStore, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown export columns"):
        store.export_csv(tmp_path / "x.csv", columns=("doc_id", "bogus"))


def test_export_jsonl_includes_metadata(store: AssertionStore, tmp_path: Path) -> None:
    doc = Document.from_content("Hello.", source_path="hello.txt", doc_date="2025-09-30")
    store.add_document(doc)
    store.add_assertion(make_assertion(doc, "Hello.", start=0, end=6))

    out = tmp_path / "out.jsonl"
    store.export_jsonl(out)
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["assertion_text"] == "Hello."
    assert row["source_path"] == "hello.txt"
    assert row["doc_date"] == "2025-09-30"
    assert row["char_start"] == 0


def test_context_manager(tmp_path: Path) -> None:
    db_path = tmp_path / "ctx.db"
    with AssertionStore(db_path) as store:
        store.migrate()
        doc = make_doc()
        store.add_document(doc)
    # After exit the connection is closed; opening a new store should still work.
    with AssertionStore(db_path) as reopened:
        assert reopened.get_document(doc.doc_id) is not None
