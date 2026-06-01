"""Tests for the Phase-1 single-page shell (ADR-0017).

The shell ships behind ``?new_ui=1``. The legacy tab nav must keep
working without the flag — every other test file relies on that
behaviour, so the regression bar here is "don't break what we have."
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.check.gate import CandidatePair
from consistency_checker.check.llm_judge import FixtureJudge, JudgeVerdict
from consistency_checker.check.nli_checker import FixtureNliChecker, NliResult
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
        gate_similarity_threshold=-1.0,
        nli_contradiction_threshold=0.0,
    )


def _client(cfg: Config) -> TestClient:
    return TestClient(
        create_app(
            cfg,
            extractor=FixtureExtractor({}),
            embedder=HashEmbedder(dim=64),
            nli_checker=FixtureNliChecker({}),
            judge=FixtureJudge({}),
        )
    )


def _seed_one_corpus(cfg: Config, *, name: str = "Alpha corpus") -> str:
    store = AssertionStore(cfg.db_path)
    store.migrate()
    cid = store.get_or_create_corpus(name, f"/{name}", "moonshot")
    store.close()
    return cid


def _seed_corpus_with_finding(
    cfg: Config, *, name: str = "Findings corpus"
) -> tuple[str, str, str]:
    """Seed a corpus + run + 1 contradiction finding; return ids and rationale."""
    store = AssertionStore(cfg.db_path)
    store.migrate()
    cid = store.get_or_create_corpus(name, f"/{name}", "moonshot")
    doc_a = Document.from_content("Alpha body.", source_path="alpha.md", title="Alpha")
    doc_b = Document.from_content("Beta body.", source_path="beta.txt", title="Beta")
    store.add_document(doc_a, corpus_id=cid)
    store.add_document(doc_b, corpus_id=cid)
    a = Assertion.build(doc_a.doc_id, "Revenue grew 12% in fiscal 2025.")
    b = Assertion.build(doc_b.doc_id, "Revenue declined 5% in fiscal 2025.")
    store.add_assertions([a, b])

    rationale = "Opposite revenue signs at the same fiscal-year scope."
    audit = AuditLogger(store)
    run_id = audit.begin_run(corpus_id=cid)
    audit.record_finding(
        run_id,
        candidate=CandidatePair(a=a, b=b, score=0.92),
        nli=NliResult.from_scores(p_contradiction=0.83, p_entailment=0.05, p_neutral=0.12),
        verdict=JudgeVerdict(
            assertion_a_id=a.assertion_id,
            assertion_b_id=b.assertion_id,
            verdict="contradiction",
            confidence=0.91,
            rationale=rationale,
            evidence_spans=["grew 12%", "declined 5%"],
        ),
    )
    audit.end_run(run_id, n_assertions=2, n_pairs_gated=1, n_pairs_judged=1, n_findings=1)
    store.close()
    return cid, run_id, rationale


# --- 1) flag absence: legacy tab nav still rendered ----------------------


def test_index_without_new_ui_flag_renders_legacy(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _seed_corpus_with_finding(cfg)
    client = _client(cfg)
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    assert "cc-tab" in body
    assert "cc-shell" not in body


# --- 2) flag present: new shell rendered, no tabs ------------------------


def test_index_with_new_ui_flag_renders_shell(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _seed_one_corpus(cfg)
    client = _client(cfg)
    resp = client.get("/?new_ui=1")
    assert resp.status_code == 200
    body = resp.text
    assert "cc-shell" in body
    assert 'id="cc-sidebar"' in body
    assert 'id="cc-main"' in body
    # Tab nav must NOT bleed through (cc_base.html not extended).
    assert "cc-tabs" not in body
    assert ">01</span> Ingest" not in body


# --- 3) sidebar lists all corpora ----------------------------------------


def test_sidebar_lists_corpora(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    store = AssertionStore(cfg.db_path)
    store.migrate()
    store.get_or_create_corpus("Corpus One", "/one", "moonshot")
    store.get_or_create_corpus("Corpus Two", "/two", "moonshot")
    store.close()

    client = _client(cfg)
    resp = client.get("/?new_ui=1")
    assert resp.status_code == 200
    body = resp.text
    assert "Corpus One" in body
    assert "Corpus Two" in body


# --- 4) findings panel shows findings from active corpus ----------------


def test_findings_panel_shows_findings_from_active_corpus(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _cid, _rid, rationale = _seed_corpus_with_finding(cfg)
    client = _client(cfg)
    resp = client.get("/?new_ui=1")
    assert resp.status_code == 200
    body = resp.text
    assert rationale in body
    assert "Findings corpus" in body
    # Confidence label rendered to 2 decimals.
    assert "0.91" in body


# --- 5) fragment route returns no <html>/<head> wrapper ------------------


def test_corpora_findings_fragment_returns_findings_only(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid, _rid, _rationale = _seed_corpus_with_finding(cfg)
    client = _client(cfg)
    resp = client.get(f"/corpora/{cid}/findings")
    assert resp.status_code == 200
    body = resp.text
    lower = body.lower()
    # Just the cc_findings.html body — not the full page.
    assert "<html" not in lower
    assert "<head>" not in lower
    assert "<!doctype" not in lower
    assert "Findings corpus" in body
    assert "cc-main-head" in body


def test_corpora_findings_fragment_404s_on_unknown_corpus_id(tmp_path: Path) -> None:
    """Stale HTMX links (e.g. after a corpus deletion) must 404 rather than
    silently return another corpus's findings. Path parameter = identity."""
    cfg = _config(tmp_path)
    _seed_one_corpus(cfg, name="Real corpus")
    client = _client(cfg)
    resp = client.get("/corpora/does-not-exist/findings")
    assert resp.status_code == 404
    assert "does-not-exist" in resp.text


# --- 6) empty state when active corpus has no runs -----------------------


def test_findings_panel_empty_state(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _seed_one_corpus(cfg, name="Quiet corpus")
    client = _client(cfg)
    resp = client.get("/?new_ui=1")
    assert resp.status_code == 200
    body = resp.text
    assert "Quiet corpus" in body
    assert "No runs yet" in body
    # Placeholder run button hint.
    assert "Run check" in body


# --- 7) ?corpus=<id> overrides default active-corpus pick ----------------


@pytest.mark.parametrize("which", ["a", "b"])
def test_corpus_query_param_selects_active(tmp_path: Path, which: str) -> None:
    cfg = _config(tmp_path)
    store = AssertionStore(cfg.db_path)
    store.migrate()
    cid_a = store.get_or_create_corpus("Corpus A", "/a", "moonshot")
    cid_b = store.get_or_create_corpus("Corpus B", "/b", "moonshot")
    store.close()

    target = cid_a if which == "a" else cid_b
    target_name = "Corpus A" if which == "a" else "Corpus B"

    client = _client(cfg)
    resp = client.get(f"/?new_ui=1&corpus={target}")
    assert resp.status_code == 200
    body = resp.text
    # The active corpus's name appears in the main-pane title (in the h2).
    assert f'<h2 class="cc-main-title">{target_name}</h2>' in body


# --- Phase 2: New Corpus modal (ADR-0017) -------------------------------


def test_sidebar_button_wired_to_modal_route(tmp_path: Path) -> None:
    """The sidebar's [+ New corpus] button targets the modal route, not disabled."""
    cfg = _config(tmp_path)
    _seed_one_corpus(cfg)
    client = _client(cfg)
    resp = client.get("/?new_ui=1")
    assert resp.status_code == 200
    body = resp.text
    assert 'hx-get="/corpora/new/modal"' in body
    # The Phase-1 placeholder hint must be gone from the sidebar button.
    assert 'title="Wired in Phase 2"' not in body


def test_modal_route_returns_dialog_fragment(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    client = _client(cfg)
    resp = client.get("/corpora/new/modal")
    assert resp.status_code == 200
    body = resp.text
    assert "<dialog" in body.lower()
    assert 'name="corpus_name"' in body
    assert 'name="judge_provider"' in body
    assert 'name="files"' in body
    # No full-page wrapper.
    assert "<html" not in body.lower()


def test_create_corpus_success_with_files(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    # FixtureExtractor's facts dict is keyed by chunk_id, which is a content
    # hash we don't know up front. The test verifies the ingest path runs to
    # completion by checking the document count, not the assertion count.
    client = _client(cfg)
    files = [("files", ("alpha.txt", b"Revenue grew 12% in fiscal 2025.\n", "text/plain"))]
    data = {"corpus_name": "test-alpha", "judge_provider": "moonshot"}
    resp = client.post("/corpora/new", files=files, data=data)
    assert resp.status_code == 200, resp.text
    assert "test-alpha" in resp.text
    assert resp.headers.get("HX-Trigger") == "corpus-created"

    store = AssertionStore(cfg.db_path)
    store.migrate()
    try:
        rows = store._conn.execute(
            "SELECT corpus_id FROM corpora WHERE corpus_name = ?", ("test-alpha",)
        ).fetchall()
        assert len(rows) == 1
        corpus_id = rows[0]["corpus_id"]
        # The fixture extractor returns no assertions, but the document row
        # must exist — proves the ingest pipeline ran for the uploaded file.
        n_docs = store.stats(corpus_id=corpus_id)["documents"]
        assert n_docs == 1
    finally:
        store.close()


def test_create_corpus_success_without_files(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    client = _client(cfg)
    data = {"corpus_name": "empty-bravo", "judge_provider": "moonshot"}
    resp = client.post("/corpora/new", data=data)
    assert resp.status_code == 200, resp.text
    assert "empty-bravo" in resp.text
    assert resp.headers.get("HX-Trigger") == "corpus-created"

    store = AssertionStore(cfg.db_path)
    store.migrate()
    try:
        rows = store._conn.execute(
            "SELECT corpus_id FROM corpora WHERE corpus_name = ?", ("empty-bravo",)
        ).fetchall()
        assert len(rows) == 1
        n = store.stats(corpus_id=rows[0]["corpus_id"])["assertions"]
        assert n == 0
    finally:
        store.close()


def test_create_corpus_duplicate_name_returns_409(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _seed_one_corpus(cfg, name="dup-corpus")
    client = _client(cfg)
    data = {"corpus_name": "dup-corpus", "judge_provider": "moonshot"}
    resp = client.post("/corpora/new", data=data)
    assert resp.status_code == 409
    assert "already exists" in resp.text
    assert "HX-Trigger" not in resp.headers


def test_create_corpus_invalid_name_returns_400(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    client = _client(cfg)
    data = {"corpus_name": "   ", "judge_provider": "moonshot"}
    resp = client.post("/corpora/new", data=data)
    assert resp.status_code == 400
    assert "HX-Trigger" not in resp.headers


def test_create_corpus_invalid_judge_provider_returns_400(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    client = _client(cfg)
    data = {"corpus_name": "ok-name", "judge_provider": "openai"}
    resp = client.post("/corpora/new", data=data)
    assert resp.status_code == 400
    assert "moonshot" in resp.text or "anthropic" in resp.text
    assert "HX-Trigger" not in resp.headers


def test_sidebar_refresh_endpoint_returns_fragment(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _seed_one_corpus(cfg, name="Refresh-me")
    client = _client(cfg)
    resp = client.get("/corpora/sidebar")
    assert resp.status_code == 200
    body = resp.text
    assert "Refresh-me" in body
    lower = body.lower()
    assert "<html" not in lower
    assert "<head>" not in lower
    assert "<!doctype" not in lower


def test_sidebar_refresh_honors_active_param(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg, name="Active-target")
    client = _client(cfg)
    resp = client.get(f"/corpora/sidebar?active={cid}")
    assert resp.status_code == 200
    assert "cc-corpus-row--active" in resp.text
