"""Tests for the Corpus dataclass and AssertionStore corpus CRUD."""

from __future__ import annotations

from pathlib import Path


def test_corpus_dataclass_carries_fields() -> None:
    from consistency_checker.extract.schema import Corpus

    c = Corpus(
        corpus_id="abc",
        corpus_name="atkins",
        corpus_path="/data/atkins",
        judge_provider="moonshot",
    )
    assert c.corpus_name == "atkins"


def test_get_or_create_corpus_creates_then_returns_same_id(tmp_path: Path) -> None:
    from consistency_checker.index.assertion_store import AssertionStore

    store = AssertionStore(tmp_path / "t.db")
    store.migrate()
    cid1 = store.get_or_create_corpus("atkins", "/data/atkins", "moonshot")
    cid2 = store.get_or_create_corpus("atkins", "/different/path", "anthropic")
    assert cid1 == cid2
    store.close()


def test_list_corpora_returns_all_in_creation_order(tmp_path: Path) -> None:
    from consistency_checker.index.assertion_store import AssertionStore

    store = AssertionStore(tmp_path / "t.db")
    store.migrate()
    store.get_or_create_corpus("alpha", "/a", "moonshot")
    store.get_or_create_corpus("beta", "/b", "moonshot")
    names = [c.corpus_name for c in store.list_corpora()]
    assert names == ["alpha", "beta"]
    store.close()


def test_delete_corpus_cascades_to_documents_and_assertions(tmp_path: Path) -> None:
    from consistency_checker.extract.schema import Assertion, Document
    from consistency_checker.index.assertion_store import AssertionStore

    store = AssertionStore(tmp_path / "t.db")
    store.migrate()
    cid = store.get_or_create_corpus("atkins", "/data/atkins", "moonshot")
    store.add_document(Document(doc_id="d1", source_path="/a"), corpus_id=cid)
    store.add_assertion(Assertion.build("d1", "a claim", kind="claim"))
    assert store._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    assert store._conn.execute("SELECT COUNT(*) FROM assertions").fetchone()[0] == 1
    store.delete_corpus(cid)
    assert store._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    assert store._conn.execute("SELECT COUNT(*) FROM assertions").fetchone()[0] == 0
    store.close()
