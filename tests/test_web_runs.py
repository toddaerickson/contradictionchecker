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


def _config(tmp_path: Path) -> Config:
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
    )


def _seed_store(cfg: Config) -> None:
    store = AssertionStore(cfg.db_path)
    store.migrate()
    doc_a = Document.from_content("Alpha body.", source_path="alpha.md", title="Alpha")
    doc_b = Document.from_content("Beta body.", source_path="beta.txt", title="Beta")
    store.add_document(doc_a)
    store.add_document(doc_b)
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


@pytest.fixture
def hermetic_client(tmp_path: Path) -> tuple[TestClient, Config]:
    cfg = _config(tmp_path)
    _seed_store(cfg)
    app = create_app(
        cfg,
        extractor=FixtureExtractor({}),
        embedder=HashEmbedder(dim=64),
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
    )
    return TestClient(app), cfg


# --- POST /runs + background task ----------------------------------------


def test_post_runs_redirects_non_htmx_caller(
    hermetic_client: tuple[TestClient, Config],
) -> None:
    """Direct (non-HTMX) callers get a 303 to /tabs/stats?run_id=..."""
    client, _cfg = hermetic_client
    response = client.post("/runs", data={"deep": "false"}, follow_redirects=False)
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("/tabs/stats?run_id=")


def test_post_runs_returns_hx_redirect_for_htmx_caller(
    hermetic_client: tuple[TestClient, Config],
) -> None:
    client, _cfg = hermetic_client
    response = client.post("/runs", data={"deep": "false"}, headers={"HX-Request": "true"})
    assert response.status_code == 202
    assert "HX-Redirect" in response.headers
    assert response.headers["HX-Redirect"].startswith("/tabs/stats?run_id=")


def test_post_runs_creates_pending_row_immediately(
    hermetic_client: tuple[TestClient, Config],
) -> None:
    """The HTTP response returns before pipeline.check would have time to run
    on a real corpus. The polled Stats fragment must already find the row."""
    client, cfg = hermetic_client
    response = client.post("/runs", data={"deep": "false"}, follow_redirects=False)
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
    hermetic_client: tuple[TestClient, Config],
) -> None:
    client, _cfg = hermetic_client
    response = client.post("/runs", data={"deep": "false"}, follow_redirects=False)
    run_id = response.headers["location"].rsplit("=", 1)[1]
    # Stats polling endpoint should reflect a terminal state.
    stats = client.get(f"/runs/{run_id}/stats")
    assert stats.status_code == 200
    assert "Run complete" in stats.text
    assert 'hx-trigger="every 2s"' not in stats.text


def test_post_runs_deep_propagates_to_pipeline(
    hermetic_client: tuple[TestClient, Config],
) -> None:
    """When deep=true is posted, the background task should call
    pipeline.check with multi_party_judge enabled. We assert via the
    config_json recorded on the run row."""
    client, cfg = hermetic_client
    response = client.post("/runs", data={"deep": "true"}, follow_redirects=False)
    run_id = response.headers["location"].rsplit("=", 1)[1]
    store = AssertionStore(cfg.db_path)
    logger = AuditLogger(store)
    run = logger.get_run(run_id)
    store.close()
    assert run is not None
    assert run.config_json is not None
    assert '"deep": true' in run.config_json
