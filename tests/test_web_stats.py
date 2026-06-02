"""Tests for the run-stats polling fragment (``GET /runs/{run_id}/stats``).

ADR-0017 Phase 6 deleted the legacy ``GET /tabs/stats`` tab; the Stats drawer
now renders via ``/corpora/{id}/drawer/stats`` (covered in
``test_web_ui_collapse.py``). The self-polling ``/runs/{run_id}/stats``
fragment survives — it's the drawer's live counter — so its coverage stays.
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
