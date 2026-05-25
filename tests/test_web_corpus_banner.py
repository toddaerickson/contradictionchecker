"""Tests for the corpus-composition banner on the web Stats tab."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.check.llm_judge import FixtureJudge
from consistency_checker.check.nli_checker import FixtureNliChecker
from consistency_checker.config import Config
from consistency_checker.extract.atomic_facts import FixtureExtractor
from consistency_checker.extract.schema import Assertion, Document
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


def _seed(cfg: Config, *, org_labels: list[str | None]) -> str:
    store = AssertionStore(cfg.db_path)
    store.migrate()
    for i, label in enumerate(org_labels):
        doc = Document.from_content(
            f"Doc {i} content.",
            source_path=f"doc{i}.md",
            title=f"Doc {i}",
            org_label=label,
            org_reason="llm" if label else None,
        )
        store.add_document(doc)
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    logger.end_run(run_id, n_assertions=0, n_pairs_gated=0, n_pairs_judged=0)
    store.close()
    return run_id


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
def client_with_two_org_corpus(tmp_path: Path) -> TestClient:
    cfg = _config(tmp_path)
    _seed(cfg, org_labels=["Acme Corp", "Globex Inc"])
    return _make_client(cfg)


@pytest.fixture
def client_with_single_org_corpus(tmp_path: Path) -> TestClient:
    cfg = _config(tmp_path)
    _seed(cfg, org_labels=["Acme Corp", "Acme Corp"])
    return _make_client(cfg)


def test_stats_tab_shows_corpus_banner_when_multi_org(
    client_with_two_org_corpus: TestClient,
) -> None:
    r = client_with_two_org_corpus.get("/tabs/stats")
    assert r.status_code == 200
    assert "cc-banner" in r.text
    assert "Corpus spans 2 organizations" in r.text


def test_stats_tab_hides_banner_for_single_org(
    client_with_single_org_corpus: TestClient,
) -> None:
    r = client_with_single_org_corpus.get("/tabs/stats")
    assert r.status_code == 200
    assert "Corpus spans" not in r.text


def test_run_stats_fragment_shows_corpus_banner_when_multi_org(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    run_id = _seed(cfg, org_labels=["Acme Corp", "Globex Inc"])
    client = _make_client(cfg)
    r = client.get(f"/runs/{run_id}/stats")
    assert r.status_code == 200
    assert "cc-banner" in r.text
    assert "Corpus spans 2 organizations" in r.text


def test_run_stats_fragment_shows_suppressed_pair_count(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    store = AssertionStore(cfg.db_path)
    store.migrate()
    doc_a = Document.from_content(
        "A.", source_path="a.md", title="Doc A", org_label="Acme Corp", org_reason="llm"
    )
    doc_b = Document.from_content(
        "B.", source_path="b.md", title="Doc B", org_label="Globex Inc", org_reason="llm"
    )
    store.add_document(doc_a)
    store.add_document(doc_b)
    a1 = Assertion.build(
        doc_a.doc_id, '"X" means A.', kind="definition", term="X", definition_text="A"
    )
    a2 = Assertion.build(
        doc_b.doc_id, '"X" means B.', kind="definition", term="X", definition_text="B"
    )
    a3 = Assertion.build(
        doc_a.doc_id, '"Y" means A.', kind="definition", term="Y", definition_text="A"
    )
    a4 = Assertion.build(
        doc_b.doc_id, '"Y" means B.', kind="definition", term="Y", definition_text="B"
    )
    store.add_assertions([a1, a2, a3, a4])
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    store.insert_suppressed_finding(
        run_id=run_id,
        assertion_a_id=a1.assertion_id,
        assertion_b_id=a2.assertion_id,
    )
    store.insert_suppressed_finding(
        run_id=run_id,
        assertion_a_id=a3.assertion_id,
        assertion_b_id=a4.assertion_id,
    )
    logger.end_run(run_id, n_assertions=0, n_pairs_gated=0, n_pairs_judged=0)
    store.close()
    client = _make_client(cfg)
    r = client.get(f"/runs/{run_id}/stats")
    assert r.status_code == 200
    assert "2 cross-organization definition pair(s) were suppressed" in r.text
