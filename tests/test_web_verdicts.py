"""Tests for the verdict-setting POST endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from consistency_checker.check.llm_judge import FixtureJudge
from consistency_checker.check.nli_checker import FixtureNliChecker
from consistency_checker.config import Config
from consistency_checker.extract.atomic_facts import FixtureExtractor
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.web.app import create_app
from tests.conftest import HashEmbedder


@pytest.fixture
def app_client(tmp_path: Path) -> tuple[TestClient, Config]:
    cfg = Config(
        corpus_dir=tmp_path / "corpus",
        judge_provider="fixture",
        judge_model="test",
        data_dir=tmp_path / "store",
        log_dir=tmp_path / "logs",
        embedder_model="hash",
        nli_model="fixture",
    )
    AssertionStore(cfg.db_path).migrate()
    app = create_app(
        cfg,
        extractor=FixtureExtractor({}),
        embedder=HashEmbedder(dim=64),
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
    )
    return TestClient(app), cfg


def test_post_verdicts_inserts_row(
    app_client: tuple[TestClient, Config],
) -> None:
    client, cfg = app_client
    resp = client.post(
        "/verdicts",
        data={
            "pair_key": "a:b",
            "detector_type": "contradiction",
            "verdict": "confirmed",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    store = AssertionStore(cfg.db_path)
    rows = store._conn.execute(
        "SELECT verdict FROM reviewer_verdicts WHERE pair_key = ?", ("a:b",)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["verdict"] == "confirmed"
    store.close()


def test_post_verdicts_response_contains_oob_swaps(
    app_client: tuple[TestClient, Config],
) -> None:
    client, _cfg = app_client
    resp = client.post(
        "/verdicts",
        data={
            "pair_key": "a:b",
            "detector_type": "contradiction",
            "verdict": "confirmed",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    assert 'id="cc-toast"' in resp.text
    assert 'hx-swap-oob="outerHTML"' in resp.text
    assert 'id="cc-progress-count-contradiction"' in resp.text
    assert "Real issue" in resp.text


def test_post_verdicts_rejects_bogus_verdict(
    app_client: tuple[TestClient, Config],
) -> None:
    client, _cfg = app_client
    resp = client.post(
        "/verdicts",
        data={
            "pair_key": "a:b",
            "detector_type": "contradiction",
            "verdict": "banana",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 400


def test_post_verdicts_rejects_bogus_detector_type(
    app_client: tuple[TestClient, Config],
) -> None:
    client, _cfg = app_client
    resp = client.post(
        "/verdicts",
        data={
            "pair_key": "a:b",
            "detector_type": "bogus",
            "verdict": "confirmed",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 400


def test_post_verdicts_undo_first_click_case_deletes(
    app_client: tuple[TestClient, Config],
) -> None:
    """Undo with empty prior_verdict deletes the row (first-click case)."""
    client, cfg = app_client
    client.post(
        "/verdicts",
        data={
            "pair_key": "a:b",
            "detector_type": "contradiction",
            "verdict": "confirmed",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    resp = client.post(
        "/verdicts/undo",
        data={
            "pair_key": "a:b",
            "detector_type": "contradiction",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    store = AssertionStore(cfg.db_path)
    rows = store._conn.execute("SELECT COUNT(*) FROM reviewer_verdicts").fetchone()
    assert rows[0] == 0
    store.close()


def test_post_verdicts_undo_rejudge_case_restores_prior(
    app_client: tuple[TestClient, Config],
) -> None:
    """Undo with non-empty prior_verdict re-sets to the prior value."""
    client, cfg = app_client
    client.post(
        "/verdicts",
        data={
            "pair_key": "a:b",
            "detector_type": "contradiction",
            "verdict": "confirmed",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    client.post(
        "/verdicts",
        data={
            "pair_key": "a:b",
            "detector_type": "contradiction",
            "verdict": "false_positive",
            "prior_verdict": "confirmed",
        },
        headers={"HX-Request": "true"},
    )
    resp = client.post(
        "/verdicts/undo",
        data={
            "pair_key": "a:b",
            "detector_type": "contradiction",
            "prior_verdict": "confirmed",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    store = AssertionStore(cfg.db_path)
    row = store._conn.execute(
        "SELECT verdict FROM reviewer_verdicts WHERE pair_key = ?", ("a:b",)
    ).fetchone()
    assert row["verdict"] == "confirmed"
    store.close()


def test_post_verdicts_undo_rejects_bogus_prior_verdict(
    app_client: tuple[TestClient, Config],
) -> None:
    client, _cfg = app_client
    resp = client.post(
        "/verdicts/undo",
        data={
            "pair_key": "a:b",
            "detector_type": "contradiction",
            "prior_verdict": "banana",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 400


def _seed_pair_contradiction(cfg: Config) -> tuple[str, str]:
    """Helper: ingest one pair-contradiction finding. Returns (a_id, b_id)."""
    from consistency_checker.audit.logger import AuditLogger
    from consistency_checker.check.gate import CandidatePair
    from consistency_checker.check.llm_judge import JudgeVerdict
    from consistency_checker.check.nli_checker import NliResult
    from consistency_checker.extract.schema import Assertion, Document

    store = AssertionStore(cfg.db_path)
    store.migrate()
    doc_a = Document.from_content("A body.", source_path="a.md", title="Doc A")
    doc_b = Document.from_content("B body.", source_path="b.md", title="Doc B")
    store.add_document(doc_a)
    store.add_document(doc_b)
    a = Assertion.build(doc_a.doc_id, "Revenue grew 12%.")
    b = Assertion.build(doc_b.doc_id, "Revenue declined 5%.")
    store.add_assertions([a, b])
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    logger.record_finding(
        run_id,
        candidate=CandidatePair(a=a, b=b, score=0.9),
        nli=NliResult.from_scores(p_contradiction=0.85, p_entailment=0.05, p_neutral=0.10),
        verdict=JudgeVerdict(
            assertion_a_id=a.assertion_id,
            assertion_b_id=b.assertion_id,
            verdict="contradiction",
            confidence=0.9,
            rationale="opposite revenue signs",
            evidence_spans=[],
        ),
    )
    logger.end_run(run_id, n_assertions=2, n_pairs_gated=1, n_pairs_judged=1)
    store.close()
    return a.assertion_id, b.assertion_id


def test_contradictions_tab_hides_reviewed_by_default(
    app_client: tuple[TestClient, Config],
) -> None:
    client, cfg = app_client
    a_id, b_id = _seed_pair_contradiction(cfg)
    pair_key = ":".join(sorted([a_id, b_id]))
    client.post(
        "/verdicts",
        data={
            "pair_key": pair_key,
            "detector_type": "contradiction",
            "verdict": "confirmed",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    resp = client.get("/")
    assert resp.status_code == 200
    # The finding's rationale or assertion text should not appear in the default view
    assert "opposite revenue signs" not in resp.text


def test_contradictions_tab_shows_reviewed_when_toggle_on(
    app_client: tuple[TestClient, Config],
) -> None:
    client, cfg = app_client
    a_id, b_id = _seed_pair_contradiction(cfg)
    pair_key = ":".join(sorted([a_id, b_id]))
    client.post(
        "/verdicts",
        data={
            "pair_key": pair_key,
            "detector_type": "contradiction",
            "verdict": "confirmed",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    resp = client.get("/?show_reviewed_contradiction=true")
    assert resp.status_code == 200
    assert "opposite revenue signs" in resp.text


def test_contradictions_tab_renders_progress_count(
    app_client: tuple[TestClient, Config],
) -> None:
    client, cfg = app_client
    _seed_pair_contradiction(cfg)
    resp = client.get("/")
    assert "0 of 1 reviewed" in resp.text


def test_contradictions_tab_renders_verdict_buttons_for_unreviewed(
    app_client: tuple[TestClient, Config],
) -> None:
    client, cfg = app_client
    _seed_pair_contradiction(cfg)
    resp = client.get("/")
    # Unreviewed row should have all three verdict buttons in the partial
    assert 'data-verdict="confirmed"' in resp.text
    assert 'data-verdict="false_positive"' in resp.text


def _seed_definition_finding(cfg: Config) -> tuple[str, str]:
    """Helper: ingest one definition-inconsistency finding."""
    from consistency_checker.audit.logger import AuditLogger
    from consistency_checker.check.definition_checker import DefinitionFinding, DefinitionPair
    from consistency_checker.check.definition_judge import DefinitionJudgeVerdict
    from consistency_checker.extract.schema import Assertion, Document

    store = AssertionStore(cfg.db_path)
    store.migrate()
    doc_a = Document.from_content("A.", source_path="a.md", title="Doc A")
    doc_b = Document.from_content("B.", source_path="b.md", title="Doc B")
    store.add_document(doc_a)
    store.add_document(doc_b)
    a = Assertion.build(
        doc_a.doc_id,
        '"MAE" means A.',
        kind="definition",
        term="MAE",
        definition_text="A",
    )
    b = Assertion.build(
        doc_b.doc_id,
        '"MAE" means B.',
        kind="definition",
        term="MAE",
        definition_text="B",
    )
    store.add_assertions([a, b])
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    logger.record_definition_finding(
        run_id,
        finding=DefinitionFinding(
            pair=DefinitionPair(a=a, b=b, canonical_term="mae"),
            verdict=DefinitionJudgeVerdict(
                assertion_a_id=min(a.assertion_id, b.assertion_id),
                assertion_b_id=max(a.assertion_id, b.assertion_id),
                verdict="definition_divergent",
                confidence=0.9,
                rationale="scope shift",
                evidence_spans=[],
            ),
        ),
    )
    logger.end_run(run_id, n_assertions=2, n_pairs_gated=0, n_pairs_judged=0)
    store.close()
    return a.assertion_id, b.assertion_id


def test_definitions_tab_hides_reviewed_by_default(
    app_client: tuple[TestClient, Config],
) -> None:
    client, cfg = app_client
    a_id, b_id = _seed_definition_finding(cfg)
    pair_key = ":".join(sorted([a_id, b_id]))
    client.post(
        "/verdicts",
        data={
            "pair_key": pair_key,
            "detector_type": "definition_inconsistency",
            "verdict": "confirmed",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    resp = client.get("/tabs/definitions")
    assert resp.status_code == 200
    assert "scope shift" not in resp.text


def test_definitions_tab_shows_reviewed_when_toggle_on(
    app_client: tuple[TestClient, Config],
) -> None:
    client, cfg = app_client
    a_id, b_id = _seed_definition_finding(cfg)
    pair_key = ":".join(sorted([a_id, b_id]))
    client.post(
        "/verdicts",
        data={
            "pair_key": pair_key,
            "detector_type": "definition_inconsistency",
            "verdict": "confirmed",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    resp = client.get("/tabs/definitions?show_reviewed_definition_inconsistency=true")
    assert resp.status_code == 200
    assert "scope shift" in resp.text


def test_definitions_tab_progress_count_and_buttons(
    app_client: tuple[TestClient, Config],
) -> None:
    client, cfg = app_client
    _seed_definition_finding(cfg)
    resp = client.get("/tabs/definitions")
    assert "0 of 1 reviewed" in resp.text
    assert 'data-verdict="confirmed"' in resp.text
    assert 'data-verdict="false_positive"' in resp.text
    assert 'data-verdict="dismissed"' in resp.text
    assert 'data-verdict="dismissed"' in resp.text
