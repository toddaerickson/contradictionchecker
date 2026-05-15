"""Tests for the Stats tab — step G4.

The status inference uses ``finished_at IS NULL`` as a proxy for "running"
until G5 ships the real ``run_status`` column. These tests exercise the
inference + the live/final fragment shape + the HTMX self-polling wiring.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.config import Config
from consistency_checker.extract.atomic_facts import FixtureExtractor
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


def _begin_run(cfg: Config, run_id: str) -> AuditLogger:
    store = AssertionStore(cfg.db_path)
    store.migrate()
    logger = AuditLogger(store)
    logger.begin_run(run_id=run_id)
    return logger


# --- empty state ----------------------------------------------------------


def test_stats_tab_empty_state(tmp_path: Path) -> None:
    client = _client(_config(tmp_path))
    response = client.get("/tabs/stats")
    assert response.status_code == 200
    body = response.text
    assert "Stats" in body
    assert "No runs yet" in body


# --- running state --------------------------------------------------------


def test_stats_tab_running_state_renders_live_fragment(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    logger = _begin_run(cfg, run_id="g4runrun")
    # Don't call end_run — finished_at stays None → status == "running".
    client = _client(cfg)
    response = client.get("/tabs/stats")
    assert response.status_code == 200
    body = response.text
    assert "Run in progress" in body
    # Live fragment is self-polling.
    assert 'hx-trigger="every 2s"' in body
    assert "/runs/g4runrun/stats" in body
    assert "Run complete" not in body
    # Counters dl is present.
    assert "Assertions" in body and "Candidates screened" in body and "Findings" in body
    del logger


# --- done state ----------------------------------------------------------


def test_stats_tab_done_state_renders_final_card(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    logger = _begin_run(cfg, run_id="g4rundone")
    logger.end_run("g4rundone", n_assertions=12, n_pairs_gated=4, n_pairs_judged=2, n_findings=1)
    client = _client(cfg)
    response = client.get("/tabs/stats")
    assert response.status_code == 200
    body = response.text
    assert "Run complete" in body
    # Final fragment does NOT self-poll.
    assert 'hx-trigger="every 2s"' not in body
    assert "Run in progress" not in body
    # Counters present.
    assert "12" in body  # n_assertions
    assert "View contradictions" in body


# --- polling endpoint ----------------------------------------------------


def test_stats_polling_endpoint_returns_live_fragment_while_running(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    _begin_run(cfg, run_id="g4poll1")
    client = _client(cfg)
    response = client.get("/runs/g4poll1/stats")
    assert response.status_code == 200
    body = response.text
    assert "Run in progress" in body
    assert 'hx-trigger="every 2s"' in body
    # No base chrome — partial only.
    assert 'class="cc-header"' not in body


def test_stats_polling_endpoint_returns_final_fragment_when_done(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    logger = _begin_run(cfg, run_id="g4poll2")
    logger.end_run("g4poll2", n_assertions=5, n_pairs_gated=2, n_pairs_judged=1, n_findings=0)
    client = _client(cfg)
    response = client.get("/runs/g4poll2/stats")
    assert response.status_code == 200
    body = response.text
    assert "Run complete" in body
    # Critical: no polling attribute → HTMX stops polling.
    assert 'hx-trigger="every 2s"' not in body


def test_stats_polling_endpoint_404_for_unknown_run(tmp_path: Path) -> None:
    client = _client(_config(tmp_path))
    response = client.get("/runs/no_such_run/stats")
    assert response.status_code == 404


# --- HTMX tab swap behaviour ---------------------------------------------


def test_stats_tab_htmx_request_omits_base_chrome(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _begin_run(cfg, run_id="g4chrome")
    client = _client(cfg)
    response = client.get("/tabs/stats", headers={"HX-Request": "true"})
    assert response.status_code == 200
    body = response.text
    assert 'class="cc-tabs"' not in body
    assert 'class="cc-header"' not in body
    assert "Run in progress" in body


# --- transition: running → done -----------------------------------------


def test_polling_transitions_when_run_ends(tmp_path: Path) -> None:
    """First poll while running returns the live fragment; second poll after
    end_run returns the final card — proving the polling loop self-stops."""
    cfg = _config(tmp_path)
    logger = _begin_run(cfg, run_id="g4xition")
    client = _client(cfg)

    first = client.get("/runs/g4xition/stats")
    assert "Run in progress" in first.text

    logger.end_run("g4xition", n_assertions=1, n_pairs_gated=0, n_pairs_judged=0, n_findings=0)

    second = client.get("/runs/g4xition/stats")
    assert "Run complete" in second.text
    assert 'hx-trigger="every 2s"' not in second.text
