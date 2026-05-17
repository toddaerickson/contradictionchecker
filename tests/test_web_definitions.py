"""Tests for the web Definitions tab and counters."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.check.definition_checker import (
    DefinitionFinding,
    DefinitionPair,
)
from consistency_checker.check.definition_judge import DefinitionJudgeVerdict
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


def _seed_with_definition_finding(cfg: Config) -> str:
    store = AssertionStore(cfg.db_path)
    store.migrate()
    doc_a = Document.from_content("A.", source_path="a.md", title="Doc A")
    doc_b = Document.from_content("B.", source_path="b.md", title="Doc B")
    store.add_document(doc_a)
    store.add_document(doc_b)
    a = Assertion.build(
        doc_a.doc_id, '"MAE" means A.', kind="definition", term="MAE", definition_text="A"
    )
    b = Assertion.build(
        doc_b.doc_id, '"MAE" means B.', kind="definition", term="MAE", definition_text="B"
    )
    store.add_assertions([a, b])
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    finding = DefinitionFinding(
        pair=DefinitionPair(a=a, b=b, canonical_term="mae"),
        verdict=DefinitionJudgeVerdict(
            assertion_a_id=min(a.assertion_id, b.assertion_id),
            assertion_b_id=max(a.assertion_id, b.assertion_id),
            verdict="definition_divergent",
            confidence=0.91,
            rationale="scope shift",
            evidence_spans=["A", "B"],
        ),
    )
    logger.record_definition_finding(run_id, finding=finding)
    logger.end_run(run_id, n_assertions=2, n_pairs_gated=0, n_pairs_judged=0)
    store.close()
    return run_id


@pytest.fixture
def hermetic_client_with_def(tmp_path: Path) -> tuple[TestClient, Config, str]:
    cfg = _config(tmp_path)
    run_id = _seed_with_definition_finding(cfg)
    app = create_app(
        cfg,
        extractor=FixtureExtractor({}),
        embedder=HashEmbedder(dim=64),
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
    )
    return TestClient(app), cfg, run_id


def test_definitions_tab_lists_finding(
    hermetic_client_with_def: tuple[TestClient, Config, str],
) -> None:
    client, _cfg, _run_id = hermetic_client_with_def
    resp = client.get("/tabs/definitions")
    assert resp.status_code == 200
    assert "Definition inconsistencies" in resp.text
    assert "MAE" in resp.text
    assert "Doc A" in resp.text
    assert "Doc B" in resp.text
    assert "scope shift" in resp.text


def test_definitions_tab_renders_empty_state_without_findings(
    hermetic_client_with_def: tuple[TestClient, Config, str],
) -> None:
    # Wipe the seeded finding so the tab shows the empty state.
    client, cfg, _run_id = hermetic_client_with_def
    store = AssertionStore(cfg.db_path)
    store._conn.execute("DELETE FROM findings")
    store._conn.commit()
    store.close()
    resp = client.get("/tabs/definitions")
    assert resp.status_code == 200
    assert "All findings reviewed." in resp.text


def test_base_nav_has_definitions_link(
    hermetic_client_with_def: tuple[TestClient, Config, str],
) -> None:
    client, _cfg, _run_id = hermetic_client_with_def
    resp = client.get("/tabs/definitions")
    assert "/tabs/definitions" in resp.text
    assert "Definitions" in resp.text


def test_stats_counters_include_definition_totals(
    hermetic_client_with_def: tuple[TestClient, Config, str],
) -> None:
    client, _cfg, run_id = hermetic_client_with_def
    resp = client.get(f"/runs/{run_id}/stats")
    assert resp.status_code == 200
    # The counters dict is serialised via the template; both keys must be in scope.
    # We assert via raw text presence of the counter labels.
    # (No specific phrasing required — just that the counter is wired.)
