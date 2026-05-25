"""Tests for corpus-scoped upload and read routes (Task 10)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.check.llm_judge import FixtureJudge
from consistency_checker.check.nli_checker import FixtureNliChecker
from consistency_checker.config import Config
from consistency_checker.extract.atomic_facts import FixtureExtractor
from consistency_checker.extract.schema import Document
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.web.app import create_app
from tests.conftest import HashEmbedder


def _config(tmp_path: Path) -> Config:
    return Config(
        corpus_dir=tmp_path / "corpus",
        judge_provider="fixture",
        judge_model="test",
        data_dir=tmp_path / "store",
        log_dir=tmp_path / "logs",
        embedder_model="hash",
        nli_model="fixture",
    )


def _make_client(cfg: Config) -> TestClient:
    app = create_app(
        cfg,
        extractor=FixtureExtractor({}),
        embedder=HashEmbedder(dim=64),
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
    )
    return TestClient(app)


@pytest.fixture
def client_factory(
    tmp_path: Path,
) -> tuple[TestClient, AssertionStore]:
    """Returns a (client, store) pair sharing the same db.

    Callers are responsible for closing the store when done.
    """

    def _factory(
        tmp: Path,
    ) -> tuple[TestClient, AssertionStore]:
        cfg = _config(tmp)
        store = AssertionStore(cfg.db_path)
        store.migrate()
        client = _make_client(cfg)
        return client, store

    return _factory  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Upload requires corpus_id
# ---------------------------------------------------------------------------


def test_web_upload_requires_corpus_id(tmp_path: Path) -> None:
    """POSTing without corpus_id must 400."""
    cfg = _config(tmp_path)
    client = _make_client(cfg)
    r = client.post("/uploads", files=[("files", ("doc.txt", b"hello", "text/plain"))])
    assert r.status_code == 400
    assert "corpus" in r.text.lower()


def test_web_upload_with_corpus_id_persists_link(tmp_path: Path) -> None:
    """Upload with a real corpus_id; assert doc's corpus_id matches."""
    cfg = _config(tmp_path)
    store = AssertionStore(cfg.db_path)
    store.migrate()
    cid = store.get_or_create_corpus("atkins", "/atkins", "moonshot")
    store.close()

    client = _make_client(cfg)
    r = client.post(
        "/uploads",
        data={"corpus_id": cid},
        files=[("files", ("doc.txt", b"Atkins bylaws", "text/plain"))],
    )
    assert r.status_code in (200, 303), r.text

    verify_store = AssertionStore(cfg.db_path)
    try:
        rows = verify_store._conn.execute("SELECT corpus_id FROM documents").fetchall()
        assert rows and all(row[0] == cid for row in rows)
    finally:
        verify_store.close()


# ---------------------------------------------------------------------------
# Stats tab scoped to selected corpus
# ---------------------------------------------------------------------------


def test_stats_tab_filters_to_selected_corpus(tmp_path: Path) -> None:
    """When ?corpus=<id> is set on the Stats tab, only that corpus's banner shows."""
    cfg = _config(tmp_path)
    store = AssertionStore(cfg.db_path)
    store.migrate()
    cid_a = store.get_or_create_corpus("atkins", "/atkins", "moonshot")
    cid_b = store.get_or_create_corpus("lockhart", "/lockhart", "moonshot")

    # Seed corpus A with two distinct org docs so its banner fires.
    doc_a1 = Document.from_content(
        "A1", source_path="a1.md", title="Atkins A1", org_label="OrgA", org_reason="llm"
    )
    doc_a2 = Document.from_content(
        "A2", source_path="a2.md", title="Atkins A2", org_label="OrgB", org_reason="llm"
    )
    # Seed corpus B with single org so its banner stays silent.
    doc_b1 = Document.from_content(
        "B1", source_path="b1.md", title="Lockhart B1", org_label="OrgC", org_reason="llm"
    )
    store.add_document(doc_a1, corpus_id=cid_a)
    store.add_document(doc_a2, corpus_id=cid_a)
    store.add_document(doc_b1, corpus_id=cid_b)

    logger = AuditLogger(store)
    run_id = logger.begin_run()
    logger.end_run(run_id, n_assertions=0, n_pairs_gated=0, n_pairs_judged=0)
    store.close()

    client = _make_client(cfg)
    r = client.get(f"/tabs/stats?corpus={cid_a}")
    assert r.status_code == 200
    # Corpus A has two orgs — the banner should fire.
    assert "Corpus spans 2 organizations" in r.text
