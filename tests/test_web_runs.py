"""Tests for POST /corpora/{id}/run + background-task wiring.

ADR-0017 Phase 6 deleted the legacy ``POST /runs`` route; the only run-start
surface is now the per-corpus ``POST /corpora/{id}/run`` (the Run Check modal).
It shares the same ``_run_check_in_background`` worker, so the behaviours that
matter — done-state, deep→config propagation, the pairwise opt-in skip, the
cost-ceiling failure diagnostic, and generic-failure masking — are exercised
here against the surviving route.

The FastAPI TestClient runs BackgroundTasks synchronously after the response
returns, so the run row goes through pending → running → done within a single
test invocation.
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
    """Seed the store with two docs (embedded) and return the corpus_id."""
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


def _latest_run(cfg: Config) -> object | None:
    store = AssertionStore(cfg.db_path)
    try:
        return AuditLogger(store).most_recent_run()
    finally:
        store.close()


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


# --- POST /corpora/{id}/run + background task ----------------------------


def test_post_run_returns_run_started_trigger(
    hermetic_client: tuple[TestClient, Config, str],
) -> None:
    client, _cfg, corpus_id = hermetic_client
    response = client.post(f"/corpora/{corpus_id}/run", data={"pairwise": "true", "deep": "false"})
    assert response.status_code == 200, response.text
    assert response.headers.get("HX-Trigger") == "run-started"


def test_post_run_marks_done_after_completion(
    hermetic_client: tuple[TestClient, Config, str],
) -> None:
    """TestClient runs the background task synchronously after the response,
    so by the time we read the row the run has already completed."""
    client, cfg, corpus_id = hermetic_client
    response = client.post(f"/corpora/{corpus_id}/run", data={"pairwise": "true", "deep": "false"})
    assert response.status_code == 200, response.text
    run = _latest_run(cfg)
    assert run is not None
    assert run.run_status == "done"  # type: ignore[attr-defined]


def test_post_run_deep_propagates_to_config(
    hermetic_client: tuple[TestClient, Config, str],
) -> None:
    """deep=true must land as ``enable_multi_party`` in the begin_run config
    dict (CLI key vocabulary), proving it threads to the pipeline."""
    client, cfg, corpus_id = hermetic_client
    response = client.post(f"/corpora/{corpus_id}/run", data={"pairwise": "true", "deep": "true"})
    assert response.status_code == 200, response.text
    run = _latest_run(cfg)
    assert run is not None
    assert run.config_json is not None  # type: ignore[attr-defined]
    assert '"enable_multi_party": true' in run.config_json  # type: ignore[attr-defined]


# --- ADR-0015: pairwise opt-in -------------------------------------------


def test_run_check_skips_nli_when_pairwise_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With effective pairwise=false the background task must NOT construct
    TransformerNliChecker. We monkeypatch its __init__ to raise — the run
    should still complete cleanly via the definitions-only path."""
    cfg = _config(tmp_path, pairwise_enabled=False)
    corpus_id = _seed_store(cfg)

    def _boom(self: object, *args: object, **kwargs: object) -> None:
        raise AssertionError("TransformerNliChecker must not be constructed")

    monkeypatch.setattr(
        "consistency_checker.check.nli_checker.TransformerNliChecker.__init__",
        _boom,
    )

    app = create_app(
        cfg,
        extractor=FixtureExtractor({}),
        embedder=HashEmbedder(dim=64),
        judge=FixtureJudge({}),
    )
    client = TestClient(app)
    # pairwise="" → fall back to config default (False).
    response = client.post(f"/corpora/{corpus_id}/run", data={"pairwise": "", "deep": "false"})
    assert response.status_code == 200, response.text
    run = _latest_run(cfg)
    assert run is not None
    assert run.run_status == "done"  # type: ignore[attr-defined]


# --- ADR-0016: cost ceiling -----------------------------------------------


def test_run_check_records_cost_ceiling_failure_with_clean_diagnostic(
    tmp_path: Path,
) -> None:
    """ADR-0016: a run that trips max_cost_usd must be marked 'failed' with a
    user-friendly error_message naming the projected vs. ceiling values, not a
    raw stack trace from the generic ``except Exception`` branch."""
    cfg = _config(tmp_path).model_copy(
        update={"judge_provider": "anthropic", "max_cost_usd": 0.001}
    )
    corpus_id = _seed_store(cfg)
    app = create_app(
        cfg,
        extractor=FixtureExtractor({}),
        embedder=HashEmbedder(dim=64),
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
    )
    client = TestClient(app)

    response = client.post(f"/corpora/{corpus_id}/run", data={"pairwise": "true", "deep": "false"})
    assert response.status_code == 200, response.text
    run = _latest_run(cfg)
    assert run is not None
    assert run.run_status == "failed"  # type: ignore[attr-defined]
    assert run.error_message is not None  # type: ignore[attr-defined]
    assert "Estimated cost" in run.error_message  # type: ignore[attr-defined]
    assert "max_cost_usd" in run.error_message  # type: ignore[attr-defined]
    assert "$0.0010" in run.error_message  # type: ignore[attr-defined]


# --- generic failures must not leak raw exception text -------------------


def test_generic_run_failure_does_not_leak_raw_exception_text(
    hermetic_client: tuple[TestClient, Config, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generic pipeline failure carrying a filesystem path must be stored as
    a generic, non-leaking message — the raw ``str(exc)`` (which can contain
    absolute paths and provider text) must never reach the run's
    ``error_message`` rendered verbatim in the UI."""
    client, cfg, corpus_id = hermetic_client

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("/home/secret/data.db is locked")

    monkeypatch.setattr("consistency_checker.pipeline.check", _boom)

    response = client.post(f"/corpora/{corpus_id}/run", data={"pairwise": "true", "deep": "false"})
    assert response.status_code == 200, response.text
    run = _latest_run(cfg)
    assert run is not None
    assert run.run_status == "failed"  # type: ignore[attr-defined]
    assert run.error_message is not None  # type: ignore[attr-defined]
    assert "/home/secret" not in run.error_message  # type: ignore[attr-defined]
    assert "data.db" not in run.error_message  # type: ignore[attr-defined]
    assert (
        run.error_message == "The check failed. See server logs for details."  # type: ignore[attr-defined]
    )


# --- clear stuck run (PR 2 CRUD essentials) ----------------------------------


def _start_stuck_run(cfg: Config, corpus_id: str, *, run_kind: str = "check") -> str:
    """Insert a run wedged in 'running' (as if its worker crashed)."""
    store = AssertionStore(cfg.db_path)
    store.migrate()
    try:
        return AuditLogger(store).begin_run(
            run_status="running", corpus_id=corpus_id, run_kind=run_kind
        )
    finally:
        store.close()


def test_run_blocked_by_stuck_run_offers_clear(
    hermetic_client: tuple[TestClient, Config, str],
) -> None:
    client, cfg, corpus_id = hermetic_client
    _start_stuck_run(cfg, corpus_id)
    resp = client.post(f"/corpora/{corpus_id}/run", data={"pairwise": "false"})
    assert resp.status_code == 409
    assert "already in progress" in resp.text
    assert f'hx-post="/corpora/{corpus_id}/runs/clear"' in resp.text
    assert "Clear stuck run" in resp.text


def test_run_blocked_by_ingest_offers_kind_aware_clear(
    hermetic_client: tuple[TestClient, Config, str],
) -> None:
    """A blocking *ingest* must label the clear control as an ingest and warn it
    abandons in-flight work, so a user doesn't nuke a long-but-healthy ingest."""
    client, cfg, corpus_id = hermetic_client
    _start_stuck_run(cfg, corpus_id, run_kind="ingest")
    resp = client.post(f"/corpora/{corpus_id}/run", data={"pairwise": "false"})
    assert resp.status_code == 409
    assert "ingest is still in progress" in resp.text.lower()
    assert "Clear stuck ingest" in resp.text
    assert "abandons it" in resp.text


def test_clear_stuck_run_marks_failed_and_unblocks(
    hermetic_client: tuple[TestClient, Config, str],
) -> None:
    client, cfg, corpus_id = hermetic_client
    stuck_id = _start_stuck_run(cfg, corpus_id)

    cleared = client.post(f"/corpora/{corpus_id}/runs/clear")
    assert cleared.status_code == 200

    store = AssertionStore(cfg.db_path)
    try:
        status = store._conn.execute(
            "SELECT run_status FROM pipeline_runs WHERE run_id = ?", (stuck_id,)
        ).fetchone()[0]
    finally:
        store.close()
    assert status == "failed"

    # The corpus is no longer blocked — a fresh run starts.
    again = client.post(f"/corpora/{corpus_id}/run", data={"pairwise": "false"})
    assert again.status_code == 200
    assert again.headers.get("HX-Trigger") == "run-started"


def test_terminal_run_status_is_write_once(
    hermetic_client: tuple[TestClient, Config, str],
) -> None:
    """A force-cleared (failed) run must not be flipped back to 'done' by a late
    worker finalisation, and a completed run must not be flipped to 'failed'."""
    _client_unused, cfg, corpus_id = hermetic_client
    stuck_id = _start_stuck_run(cfg, corpus_id)

    store = AssertionStore(cfg.db_path)
    try:
        audit = AuditLogger(store)
        audit.update_run_status(stuck_id, "failed", error_message="cleared")
        # A racing worker finalisation arriving afterwards is a no-op.
        audit.end_run(stuck_id, n_assertions=5, n_findings=2, run_status="done")
        status = store._conn.execute(
            "SELECT run_status FROM pipeline_runs WHERE run_id = ?", (stuck_id,)
        ).fetchone()[0]
    finally:
        store.close()
    assert status == "failed"
