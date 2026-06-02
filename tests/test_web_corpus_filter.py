"""Tests for corpus-scoped read routes.

ADR-0017 Phase 6 deleted the legacy ``POST /uploads``, ``GET /tabs/stats``,
and ``POST /runs`` routes. Upload + run are now ``POST /corpora/new`` and
``POST /corpora/{id}/run`` (covered in test_web_ui_collapse.py). What remains
worth asserting here is that the per-corpus Stats drawer scopes its banner to
the selected corpus and does not bleed warnings from other corpora.
"""

from __future__ import annotations

from pathlib import Path

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


# ---------------------------------------------------------------------------
# Stats drawer scoped to selected corpus
# ---------------------------------------------------------------------------


def test_stats_drawer_filters_to_selected_corpus(tmp_path: Path) -> None:
    """The per-corpus Stats drawer (/corpora/{id}/drawer/stats) scopes its
    org banner to that corpus only; a multi-org corpus fires the banner while
    a single-org sibling stays silent."""
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
    run_id = logger.begin_run(corpus_id=cid_a)
    logger.end_run(run_id, n_assertions=0, n_pairs_gated=0, n_pairs_judged=0)
    store.close()

    client = _make_client(cfg)
    # Corpus A has two orgs — the drawer banner should fire.
    r_a = client.get(f"/corpora/{cid_a}/drawer/stats")
    assert r_a.status_code == 200
    assert "Corpus spans 2 organizations" in r_a.text

    # Corpus B has one org — its banner must stay silent (no bleed from A).
    r_b = client.get(f"/corpora/{cid_b}/drawer/stats")
    assert r_b.status_code == 200
    assert "Corpus spans" not in r_b.text
