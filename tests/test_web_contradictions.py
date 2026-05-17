"""Tests for the Contradictions tab + Diff partials — step G2."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.check.gate import CandidatePair
from consistency_checker.check.llm_judge import JudgeVerdict
from consistency_checker.check.nli_checker import NliResult
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


def _client(cfg: Config) -> TestClient:
    return TestClient(
        create_app(cfg, extractor=FixtureExtractor({}), embedder=HashEmbedder(dim=64))
    )


@pytest.fixture
def empty_client(tmp_path: Path) -> TestClient:
    return _client(_config(tmp_path))


@pytest.fixture
def seeded(tmp_path: Path) -> tuple[Config, TestClient, AssertionStore, str]:
    """Configure a store with one pair finding and one multi-party finding."""
    cfg = _config(tmp_path)
    store = AssertionStore(cfg.db_path)
    store.migrate()

    doc_a = Document.from_content("Alpha body.", source_path="alpha.md", title="Alpha")
    doc_b = Document.from_content("Beta body.", source_path="beta.txt", title="Beta")
    doc_c = Document.from_content("Gamma body.", source_path="gamma.md", title="Gamma")
    for d in (doc_a, doc_b, doc_c):
        store.add_document(d)
    a1 = Assertion.build(doc_a.doc_id, "Revenue grew 12% in fiscal 2025.")
    b1 = Assertion.build(doc_b.doc_id, "Revenue declined 5% in fiscal 2025.")
    a2 = Assertion.build(doc_a.doc_id, "All employees get four weeks vacation.")
    b2 = Assertion.build(doc_b.doc_id, "Engineers are employees.")
    c1 = Assertion.build(doc_c.doc_id, "Engineers get two weeks vacation.")
    store.add_assertions([a1, b1, a2, b2, c1])

    logger = AuditLogger(store)
    run_id = logger.begin_run(run_id="g2run0001")
    logger.record_finding(
        run_id,
        candidate=CandidatePair(a=a1, b=b1, score=0.92),
        nli=NliResult.from_scores(p_contradiction=0.83, p_entailment=0.05, p_neutral=0.12),
        verdict=JudgeVerdict(
            assertion_a_id=a1.assertion_id,
            assertion_b_id=b1.assertion_id,
            verdict="contradiction",
            confidence=0.91,
            rationale="Opposite revenue signs at the same fiscal-year scope.",
            evidence_spans=["grew 12%", "declined 5%"],
        ),
    )
    logger.record_multi_party_finding(
        run_id,
        assertion_ids=[a2.assertion_id, b2.assertion_id, c1.assertion_id],
        doc_ids=[a2.doc_id, b2.doc_id, c1.doc_id],
        triangle_edge_scores=[
            (a2.assertion_id, b2.assertion_id, 0.82),
            (a2.assertion_id, c1.assertion_id, 0.74),
            (b2.assertion_id, c1.assertion_id, 0.91),
        ],
        judge_verdict="multi_party_contradiction",
        judge_confidence=0.88,
        judge_rationale="A says all employees get 4w; B says engineers are employees; so engineers should get 4w — but C says 2w.",
        evidence_spans=["four weeks", "two weeks"],
    )
    logger.end_run(run_id, n_assertions=5, n_pairs_gated=3, n_pairs_judged=1, n_findings=1)
    store.close()

    client = _client(cfg)
    # Re-open the store after closing to allow inspection.
    inspect_store = AssertionStore(cfg.db_path)
    return cfg, client, inspect_store, run_id


# --- empty state -----------------------------------------------------------


def test_root_shows_empty_state_banner_when_no_runs(empty_client: TestClient) -> None:
    # With no documents, index() redirects to Ingest (U3). Follow the redirect.
    response = empty_client.get("/", follow_redirects=True)
    assert response.status_code == 200
    body = response.text
    # Landed on Ingest tab — verify the ingest form is rendered (U4).
    assert "Judge Provider" in body


def test_root_empty_state_omits_contradictions_table(empty_client: TestClient) -> None:
    response = empty_client.get("/", follow_redirects=True)
    assert response.status_code == 200
    body = response.text
    assert "cc-findings-table" not in body


# --- populated state -------------------------------------------------------


def test_root_lists_pair_and_multi_party_findings(
    seeded: tuple[Config, TestClient, AssertionStore, str],
) -> None:
    _, client, _, run_id = seeded
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert run_id in body
    assert "Statement contradictions" in body
    assert "Cross-document contradictions" in body
    assert "Opposite revenue signs" in body
    assert "engineers should get 4w" in body


def test_root_renders_active_tab_on_contradictions(
    seeded: tuple[Config, TestClient, AssertionStore, str],
) -> None:
    _, client, _, _ = seeded
    response = client.get("/")
    body = response.text
    # cc-tab--active appears on the Contradictions link in the header.
    assert "cc-tab--active" in body
    assert "Contradictions" in body


def test_root_htmx_request_returns_fragment_only(
    seeded: tuple[Config, TestClient, AssertionStore, str],
) -> None:
    _, client, _, _ = seeded
    response = client.get("/", headers={"HX-Request": "true"})
    assert response.status_code == 200
    body = response.text
    # No base chrome (top-level header + nav + dialog) on HTMX swap.
    assert 'class="cc-header"' not in body
    assert 'class="cc-tabs"' not in body
    assert 'id="cc-diff-dialog"' not in body
    assert "Statement contradictions" in body


# --- pair diff partial -----------------------------------------------------


def test_pair_diff_partial_renders_both_sides(
    seeded: tuple[Config, TestClient, AssertionStore, str],
) -> None:
    _, client, store, run_id = seeded
    logger = AuditLogger(store)
    pair_findings = list(logger.iter_findings(run_id=run_id, verdict="contradiction"))
    assert len(pair_findings) == 1
    finding_id = pair_findings[0].finding_id
    store.close()

    response = client.get(f"/findings/{finding_id}/diff")
    assert response.status_code == 200
    body = response.text
    assert "Revenue grew 12% in fiscal 2025." in body
    assert "Revenue declined 5% in fiscal 2025." in body
    assert "Alpha" in body and "Beta" in body
    assert "cc-diff-grid--2col" in body
    # Pair-specific fields.
    assert "NLI p(contradiction)" in body
    assert "Opposite revenue signs" in body


def test_pair_diff_404_for_unknown_finding(empty_client: TestClient) -> None:
    response = empty_client.get("/findings/nope/diff")
    assert response.status_code == 404


# --- multi-party diff partial ---------------------------------------------


def test_multi_party_diff_partial_renders_three_sides(
    seeded: tuple[Config, TestClient, AssertionStore, str],
) -> None:
    _, client, store, run_id = seeded
    logger = AuditLogger(store)
    mp = list(logger.iter_multi_party_findings(run_id=run_id))
    assert len(mp) == 1
    finding_id = mp[0].finding_id
    store.close()

    response = client.get(f"/multi_party_findings/{finding_id}/diff")
    assert response.status_code == 200
    body = response.text
    assert "All employees get four weeks vacation." in body
    assert "Engineers are employees." in body
    assert "Engineers get two weeks vacation." in body
    assert "cc-diff-grid--3col" in body
    assert "min edge" in body  # min-edge-score row present
    assert "multi_party_contradiction" in body


def test_multi_party_diff_404_for_unknown_finding(empty_client: TestClient) -> None:
    response = empty_client.get("/multi_party_findings/nope/diff")
    assert response.status_code == 404


# --- dialog wiring in base layout ------------------------------------------


def test_base_layout_includes_diff_dialog(empty_client: TestClient) -> None:
    """G2 adds a single <dialog> element to base; diff buttons swap into it."""
    response = empty_client.get("/", follow_redirects=True)
    body = response.text
    assert 'id="cc-diff-dialog"' in body
    assert 'id="cc-diff-content"' in body
