"""Tests for POST /runs + background-task wiring — step G5.

The FastAPI TestClient runs BackgroundTasks synchronously after the response
returns, so the run row goes through pending → running → done within a
single test invocation. That's exactly the property G5 needs to guarantee.
"""

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
from consistency_checker.index.embedder import embed_pending
from consistency_checker.index.faiss_store import FaissStore
from consistency_checker.web.app import create_app
from tests.conftest import HashEmbedder


def _config(tmp_path: Path, *, pairwise_enabled: bool = True) -> Config:
    return Config(
        corpus_dir=tmp_path / "corpus",
        judge_provider="fixture",
        judge_model="test",
        data_dir=tmp_path / "store",
        log_dir=tmp_path / "logs",
        embedder_model="hash",
        nli_model="fixture",
        gate_similarity_threshold=-1.0,
        nli_contradiction_threshold=0.0,
        pairwise_enabled=pairwise_enabled,
    )


def _seed_store(cfg: Config) -> str:
    """Seed the store with two docs and return the corpus_id."""
    store = AssertionStore(cfg.db_path)
    store.migrate()
    cid = store.get_or_create_corpus("test", "/test", "moonshot")
    doc_a = Document.from_content("Alpha body.", source_path="alpha.md", title="Alpha")
    doc_b = Document.from_content("Beta body.", source_path="beta.txt", title="Beta")
    store.add_document(doc_a, corpus_id=cid)
    store.add_document(doc_b, corpus_id=cid)
    store.add_assertions(
        [
            Assertion.build(doc_a.doc_id, "Revenue grew 12%."),
            Assertion.build(doc_b.doc_id, "Revenue declined 5%."),
        ]
    )
    embedder = HashEmbedder(dim=64)
    fs = FaissStore.open_or_create(
        index_path=cfg.faiss_path,
        id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
        dim=embedder.dim,
    )
    embed_pending(store, fs, embedder)
    store.close()
    return cid


@pytest.fixture
def hermetic_client(tmp_path: Path) -> tuple[TestClient, Config, str]:
    cfg = _config(tmp_path)
    corpus_id = _seed_store(cfg)
    app = create_app(
        cfg,
        extractor=FixtureExtractor({}),
        embedder=HashEmbedder(dim=64),
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
    )
    return TestClient(app), cfg, corpus_id


# --- POST /runs + background task ----------------------------------------


def test_post_runs_redirects_non_htmx_caller(
    hermetic_client: tuple[TestClient, Config, str],
) -> None:
    """Direct (non-HTMX) callers get a 303 to /tabs/stats?run_id=..."""
    client, _cfg, corpus_id = hermetic_client
    response = client.post(
        "/runs", data={"deep": "false", "corpus_id": corpus_id}, follow_redirects=False
    )
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("/tabs/stats?run_id=")


def test_post_runs_returns_hx_redirect_for_htmx_caller(
    hermetic_client: tuple[TestClient, Config, str],
) -> None:
    client, _cfg, corpus_id = hermetic_client
    response = client.post(
        "/runs",
        data={"deep": "false", "corpus_id": corpus_id},
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 202
    assert "HX-Redirect" in response.headers
    assert response.headers["HX-Redirect"].startswith("/tabs/stats?run_id=")


def test_post_runs_creates_pending_row_immediately(
    hermetic_client: tuple[TestClient, Config, str],
) -> None:
    """The HTTP response returns before pipeline.check would have time to run
    on a real corpus. The polled Stats fragment must already find the row."""
    client, cfg, corpus_id = hermetic_client
    response = client.post(
        "/runs", data={"deep": "false", "corpus_id": corpus_id}, follow_redirects=False
    )
    run_id = response.headers["location"].rsplit("=", 1)[1]
    store = AssertionStore(cfg.db_path)
    logger = AuditLogger(store)
    run = logger.get_run(run_id)
    store.close()
    assert run is not None
    # TestClient runs the background task synchronously after the response
    # returns, so by the time we read, the run has already completed.
    assert run.run_status == "done"


def test_post_runs_background_task_marks_done_after_completion(
    hermetic_client: tuple[TestClient, Config, str],
) -> None:
    client, _cfg, corpus_id = hermetic_client
    response = client.post(
        "/runs", data={"deep": "false", "corpus_id": corpus_id}, follow_redirects=False
    )
    run_id = response.headers["location"].rsplit("=", 1)[1]
    # Stats polling endpoint should reflect a terminal state.
    stats = client.get(f"/runs/{run_id}/stats")
    assert stats.status_code == 200
    assert "Run complete" in stats.text
    assert 'hx-trigger="every 2s"' not in stats.text


def test_post_runs_deep_propagates_to_pipeline(
    hermetic_client: tuple[TestClient, Config, str],
) -> None:
    """When deep=true is posted, the background task should call
    pipeline.check with multi_party_judge enabled. We assert via the
    config_json recorded on the run row."""
    client, cfg, corpus_id = hermetic_client
    response = client.post(
        "/runs", data={"deep": "true", "corpus_id": corpus_id}, follow_redirects=False
    )
    run_id = response.headers["location"].rsplit("=", 1)[1]
    store = AssertionStore(cfg.db_path)
    logger = AuditLogger(store)
    run = logger.get_run(run_id)
    store.close()
    assert run is not None
    assert run.config_json is not None
    assert '"deep": true' in run.config_json


# --- ADR-0015: pairwise opt-in -------------------------------------------


def test_run_check_skips_nli_when_pairwise_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When config.pairwise_enabled=False, the background task must NOT
    construct TransformerNliChecker. We monkeypatch its __init__ to raise
    — the run should still complete cleanly."""
    cfg = _config(tmp_path, pairwise_enabled=False)
    corpus_id = _seed_store(cfg)

    def _boom(self: object, *args: object, **kwargs: object) -> None:
        raise AssertionError("TransformerNliChecker must not be constructed")

    monkeypatch.setattr(
        "consistency_checker.check.nli_checker.TransformerNliChecker.__init__",
        _boom,
    )

    # Pass no nli_checker — the gate in _run_check_in_background must skip
    # constructing one entirely.
    app = create_app(
        cfg,
        extractor=FixtureExtractor({}),
        embedder=HashEmbedder(dim=64),
        judge=FixtureJudge({}),
    )
    client = TestClient(app)
    response = client.post(
        "/runs", data={"deep": "false", "corpus_id": corpus_id}, follow_redirects=False
    )
    assert response.status_code == 303
    run_id = response.headers["location"].rsplit("=", 1)[1]

    store = AssertionStore(cfg.db_path)
    logger = AuditLogger(store)
    run = logger.get_run(run_id)
    store.close()
    assert run is not None
    assert run.run_status == "done"


def test_post_runs_rejects_deep_without_pairwise(tmp_path: Path) -> None:
    """deep=True under pairwise_enabled=False must 400 BEFORE creating a
    run row, with a message that names the config knob to flip."""
    cfg = _config(tmp_path, pairwise_enabled=False)
    corpus_id = _seed_store(cfg)
    app = create_app(
        cfg,
        extractor=FixtureExtractor({}),
        embedder=HashEmbedder(dim=64),
        judge=FixtureJudge({}),
    )
    client = TestClient(app)
    response = client.post(
        "/runs", data={"deep": "true", "corpus_id": corpus_id}, follow_redirects=False
    )
    assert response.status_code == 400
    body = response.json()
    assert "pairwise_enabled" in body["detail"]
    # No phantom run row created.
    store = AssertionStore(cfg.db_path)
    logger = AuditLogger(store)
    assert logger.most_recent_run() is None
    store.close()


def test_post_runs_accepts_deep_with_pairwise(
    hermetic_client: tuple[TestClient, Config, str],
) -> None:
    """When pairwise_enabled=True (default for hermetic_client), deep=true
    is accepted as before."""
    client, _cfg, corpus_id = hermetic_client
    response = client.post(
        "/runs", data={"deep": "true", "corpus_id": corpus_id}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"].startswith("/tabs/stats?run_id=")
