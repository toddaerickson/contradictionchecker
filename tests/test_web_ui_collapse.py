"""Tests for the single-page shell (ADR-0017).

As of Phase 6 the shell is the DEFAULT response from ``GET /`` — the legacy
7-tab UI and its routes are deleted. ``?legacy=1`` and every ``/tabs/*`` path
now return 410 Gone. The historical ``?new_ui=1`` flag is gone too, but the
query param is simply ignored (the shell is unconditional), so the older tests
that still pass it keep working.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

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
from consistency_checker.web.app import (
    _empty_text_files_from_notes,
    _ingest_progress_label,
    create_app,
)
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
            rationale=rationale,
            evidence_spans=["grew 12%", "declined 5%"],
        ),
    )
    audit.end_run(run_id, n_assertions=2, n_pairs_gated=1, n_pairs_judged=1, n_findings=1)
    store.close()
    return cid, run_id, rationale


# --- 1) default: single-page shell, no tabs ------------------------------


def test_index_default_renders_shell(tmp_path: Path) -> None:
    """Phase 6: ``GET /`` with no flag is now the single-page shell."""
    cfg = _config(tmp_path)
    _seed_one_corpus(cfg)
    client = _client(cfg)
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    assert "cc-shell" in body
    assert 'id="cc-sidebar"' in body
    assert 'id="cc-main"' in body
    # The legacy tab nav must NOT bleed through (cc_base.html is deleted).
    assert "cc-tabs" not in body
    assert ">01</span> Ingest" not in body


def test_index_default_renders_diff_dialog_shell(tmp_path: Path) -> None:
    """Regression guard: the assertions drawer's Open button targets
    ``#cc-diff-content`` and calls ``#cc-diff-dialog``.showModal() (see
    cc_assertions.html). Those element IDs lived in the now-deleted
    cc_base.html and must be present in the cc_single.html shell, or the
    drawer's Open button silently breaks. No test asserted this before."""
    cfg = _config(tmp_path)
    _seed_one_corpus(cfg)
    client = _client(cfg)
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    assert 'id="cc-diff-dialog"' in body
    assert 'id="cc-diff-content"' in body


# --- 1b) legacy tombstones: ?legacy=1 and /tabs/* → 410 ------------------


def test_index_legacy_flag_returns_410(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _seed_one_corpus(cfg)
    client = _client(cfg)
    resp = client.get("/?legacy=1")
    assert resp.status_code == 410
    assert "Visit /" in resp.text


@pytest.mark.parametrize(
    "tab", ["stats", "documents", "assertions", "definitions", "ingest", "process"]
)
def test_legacy_tab_routes_return_410(tmp_path: Path, tab: str) -> None:
    cfg = _config(tmp_path)
    _seed_one_corpus(cfg)
    client = _client(cfg)
    resp = client.get(f"/tabs/{tab}")
    assert resp.status_code == 410
    assert "Visit /" in resp.text


# --- 2) shell rendered, no tabs (legacy ?new_ui=1 flag now a no-op) ------


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
    # Tab nav must NOT bleed through (cc_base.html deleted).
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


def test_sidebar_progress_div_pins_its_own_swap_target(tmp_path: Path) -> None:
    """The per-corpus SSE progress slot must pin ``hx-target="this"``.

    It lives inside the corpus ``<a>`` which sets ``hx-target="#cc-main"`` for
    its click-to-load-findings behavior. htmx inherits ``hx-target`` to
    descendants, so without an explicit override the SSE ``snapshot`` swap would
    fire into ``#cc-main`` and replace the findings list + toolbar with a
    one-line progress bar on every page load. Pin it to the slot itself.
    """
    import re

    cfg = _config(tmp_path)
    store = AssertionStore(cfg.db_path)
    store.migrate()
    cid = store.get_or_create_corpus("Corpus One", "/one", "moonshot")
    store.close()

    client = _client(cfg)
    body = client.get(f"/?corpus={cid}").text
    m = re.search(r'<div class="cc-corpus-progress"[^>]*>', body)
    assert m is not None, "sidebar progress div not found"
    progress_tag = m.group(0)
    assert 'sse-swap="snapshot"' in progress_tag
    assert 'hx-target="this"' in progress_tag


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


def test_finding_card_shows_source_context(tmp_path: Path) -> None:
    """Phase 1a: each finding card carries both full assertions, their source
    documents, and the rationale (expandable in place), not just a one-liner."""
    cfg = _config(tmp_path)
    _cid, _rid, rationale = _seed_corpus_with_finding(cfg)
    client = _client(cfg)
    body = client.get("/").text
    assert 'class="cc-finding-detail"' in body  # expandable wrapper
    # Both full assertions are present (not just the truncated summary line).
    assert "Revenue grew 12% in fiscal 2025." in body
    assert "Revenue declined 5% in fiscal 2025." in body
    # Both source-document labels are shown.
    assert "Alpha" in body and "Beta" in body
    # The full rationale is rendered.
    assert rationale in body


def test_finding_cards_wired_for_keyboard_triage(tmp_path: Path) -> None:
    """Phase 1b/1c: cards are focusable, verdict buttons carry data-verdict, and
    the j/k + c/f/d handler ships in the shell."""
    cfg = _config(tmp_path)
    _seed_corpus_with_finding(cfg)
    body = _client(cfg).get("/").text
    assert 'class="cc-finding" id="finding-' in body and 'tabindex="0"' in body
    assert 'data-verdict="confirmed"' in body
    assert 'data-verdict="false_positive"' in body
    assert 'data-verdict="dismissed"' in body
    assert "e.key === 'j'" in body  # the keyboard handler ships in the shell


def test_empty_state_distinguishes_no_run_from_found_nothing(tmp_path: Path) -> None:
    """Phase 1d: a completed run with zero findings is the success case, not the
    same 'no run yet' message."""
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg, name="Quiet")
    client = _client(cfg)
    assert "No runs yet." in client.get(f"/?corpus={cid}").text

    store = AssertionStore(cfg.db_path)
    store.migrate()
    audit = AuditLogger(store)
    rid = audit.begin_run(corpus_id=cid)
    audit.end_run(rid, n_assertions=3, n_pairs_gated=1, n_pairs_judged=1, n_findings=0)
    store.close()

    body = client.get(f"/?corpus={cid}").text
    assert "No contradictions found" in body
    assert "No runs yet." not in body


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


# --- 6b) onboarding "How it works" guide --------------------------------


def test_onboarding_guide_on_no_corpora(tmp_path: Path) -> None:
    """With no corpora the main pane shows the numbered walkthrough."""
    cfg = _config(tmp_path)
    AssertionStore(cfg.db_path).migrate()  # empty DB, zero corpora
    client = _client(cfg)
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    assert "How it works" in body
    assert 'class="cc-steps"' in body
    # First and last step reference the real button labels.
    assert "New corpus" in body
    assert "Run check" in body


def test_onboarding_disclosure_when_corpus_selected(tmp_path: Path) -> None:
    """When a corpus is active the guide collapses into a <details> disclosure."""
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg, name="Howto corpus")
    client = _client(cfg)
    resp = client.get(f"/?corpus={cid}")
    assert resp.status_code == 200
    body = resp.text
    assert 'class="cc-howto"' in body
    assert "<summary>How it works</summary>" in body


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


@pytest.mark.parametrize("evil_name", ["..", "...", "."])
def test_create_corpus_path_traversal_name_stored_as_id_path(
    tmp_path: Path, evil_name: str
) -> None:
    """A dot-name that the char validator lets through must NOT leak into the
    stored corpus_path. The path must be the id-based one and resolve under
    data_dir/corpora — never escape to data_dir or embed the raw name."""
    cfg = _config(tmp_path)
    client = _client(cfg)
    data = {"corpus_name": evil_name, "judge_provider": "moonshot"}
    resp = client.post("/corpora/new", data=data)
    assert resp.status_code == 200, resp.text

    corpora_root = (cfg.data_dir / "corpora").resolve()
    store = AssertionStore(cfg.db_path)
    store.migrate()
    try:
        rows = store._conn.execute(
            "SELECT corpus_id, corpus_path FROM corpora WHERE corpus_name = ?",
            (evil_name,),
        ).fetchall()
        assert len(rows) == 1
        corpus_id = rows[0]["corpus_id"]
        stored_path = Path(rows[0]["corpus_path"])
        assert stored_path == cfg.data_dir / "corpora" / corpus_id
        assert evil_name not in stored_path.parts
        assert stored_path.resolve().is_relative_to(corpora_root)
        assert stored_path.resolve() != corpora_root
    finally:
        store.close()


def test_create_corpus_normal_name_stored_as_id_path(tmp_path: Path) -> None:
    """A benign name still produces an id-based path under data_dir/corpora."""
    cfg = _config(tmp_path)
    client = _client(cfg)
    data = {"corpus_name": "Plain Corpus", "judge_provider": "moonshot"}
    resp = client.post("/corpora/new", data=data)
    assert resp.status_code == 200, resp.text

    store = AssertionStore(cfg.db_path)
    store.migrate()
    try:
        rows = store._conn.execute(
            "SELECT corpus_id, corpus_path FROM corpora WHERE corpus_name = ?",
            ("Plain Corpus",),
        ).fetchall()
        assert len(rows) == 1
        corpus_id = rows[0]["corpus_id"]
        stored_path = Path(rows[0]["corpus_path"])
        assert stored_path == cfg.data_dir / "corpora" / corpus_id
        assert "Plain Corpus" not in stored_path.parts
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


def test_create_corpus_unique_raises_integrity_error(tmp_path: Path) -> None:
    """AssertionStore.create_corpus must raise on a duplicate corpus_name so the
    web handler can map the concurrent-insert race to a 409 instead of a 500."""
    cfg = _config(tmp_path)
    store = AssertionStore(cfg.db_path)
    store.migrate()
    try:
        store.create_corpus("id-1", "race-name", "/p1", "moonshot")
        with pytest.raises(sqlite3.IntegrityError):
            store.create_corpus("id-2", "race-name", "/p2", "moonshot")
    finally:
        store.close()


def test_create_corpus_concurrent_duplicate_returns_409_not_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate the concurrent-duplicate race deterministically: a real row with
    the same name already exists, but the handler's pre-INSERT duplicate SELECT
    is forced to miss it (as it would in the race window). The subsequent INSERT
    must hit the UNIQUE constraint and surface as the normal 409 modal, not a 500.
    """
    cfg = _config(tmp_path)
    seed = AssertionStore(cfg.db_path)
    seed.migrate()
    seed.create_corpus("existing-id", "race-dup", "/race", "moonshot")
    seed.close()

    class _BlindCursor:
        def fetchone(self) -> None:
            return None

    class _BlindConn:
        """Proxy over a real sqlite3 connection whose corpus_name dedup SELECT
        always misses, so the handler proceeds to INSERT and trips UNIQUE."""

        def __init__(self, real: sqlite3.Connection) -> None:
            self._real = real

        def execute(self, sql: str, *params: object) -> Any:
            stripped = sql.strip().upper()
            if "FROM CORPORA WHERE CORPUS_NAME" in stripped and stripped.startswith("SELECT"):
                return _BlindCursor()
            return self._real.execute(sql, *params)

        def __enter__(self) -> sqlite3.Connection:
            return self._real.__enter__()

        def __exit__(self, *exc: object) -> Any:
            return self._real.__exit__(*exc)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real, name)

    original_init = AssertionStore.__init__

    def _patched_init(self: AssertionStore, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._conn = _BlindConn(self._conn)  # type: ignore[assignment]

    monkeypatch.setattr(AssertionStore, "__init__", _patched_init)

    client = _client(cfg)
    data = {"corpus_name": "race-dup", "judge_provider": "moonshot"}
    resp = client.post("/corpora/new", data=data)
    assert resp.status_code == 409, resp.text
    assert "already exists" in resp.text
    assert "HX-Trigger" not in resp.headers


def test_create_corpus_timestamps_are_microsecond_iso(tmp_path: Path) -> None:
    """Rows created via the HTMX handler must carry microsecond ISO-8601
    timestamps (matching the REST path), not SQLite's space-separated
    CURRENT_TIMESTAMP form, so list_corpora ORDER BY created_at is consistent."""
    cfg = _config(tmp_path)
    client = _client(cfg)
    data = {"corpus_name": "stamped", "judge_provider": "moonshot"}
    resp = client.post("/corpora/new", data=data)
    assert resp.status_code == 200, resp.text

    store = AssertionStore(cfg.db_path)
    store.migrate()
    try:
        row = store._conn.execute(
            "SELECT created_at, updated_at FROM corpora WHERE corpus_name = ?",
            ("stamped",),
        ).fetchone()
        for stamp in (row["created_at"], row["updated_at"]):
            assert "T" in stamp, stamp
            assert "." in stamp, stamp
            assert " " not in stamp, stamp
    finally:
        store.close()


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


def _latest_run_for_corpus(cfg: Config, corpus_name: str) -> tuple[str, str, str | None] | None:
    """(run_kind, run_status, error_message) of a corpus's latest run, or None."""
    store = AssertionStore(cfg.db_path)
    try:
        cid_row = store._conn.execute(
            "SELECT corpus_id FROM corpora WHERE corpus_name = ?", (corpus_name,)
        ).fetchone()
        if cid_row is None:
            return None
        run = store._conn.execute(
            "SELECT run_kind, run_status, error_message FROM pipeline_runs "
            "WHERE corpus_id = ? ORDER BY started_at DESC, run_id DESC LIMIT 1",
            (cid_row[0],),
        ).fetchone()
        return (run[0], run[1], run[2]) if run is not None else None
    finally:
        store.close()


def test_create_corpus_ingest_failure_records_failed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-0019: ingest runs in a background task. A failure there must NOT 500
    the POST; the request returns 'ingest started' (200) and the corpus is kept
    with a FAILED ingest run carrying a user-facing message — so the user can
    see *why* it failed instead of a frozen/disappeared corpus."""
    cfg = _config(tmp_path)
    client = _client(cfg)

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic ingest failure")

    monkeypatch.setattr("consistency_checker.web.app._ingest_uploaded_paths", _boom)

    data = {"corpus_name": "fails-me", "judge_provider": "moonshot"}
    files = {"files": ("a.txt", b"some content", "text/plain")}
    resp = client.post("/corpora/new", data=data, files=files)
    assert resp.status_code == 200
    assert "Ingest started" in resp.text
    assert resp.headers.get("HX-Trigger") == "corpus-created"

    # TestClient runs the BackgroundTask synchronously after the response, so the
    # ingest run is already terminal: corpus retained, run marked failed.
    latest = _latest_run_for_corpus(cfg, "fails-me")
    assert latest is not None, "corpus must be kept so the failure is visible"
    run_kind, run_status, error_message = latest
    assert run_kind == "ingest"
    assert run_status == "failed"
    assert error_message  # a user-facing reason is recorded
    # A terminal failure stamps finished_at (not left NULL).
    assert _latest_run_row(cfg, "fails-me")["finished_at"] is not None


def test_create_corpus_cross_corpus_collision_records_failed_run(
    tmp_path: Path,
) -> None:
    """A cross-corpus document collision surfaces as a failed ingest run with an
    explanatory message, not a 409 that rolls the corpus away."""
    cfg = _config(tmp_path)

    # Seed an existing corpus + ingest one document so its content hash is in
    # the store, then attempt to create a NEW corpus that uploads the same
    # bytes. The background ingest raises CrossCorpusDocumentError.
    store = AssertionStore(cfg.db_path)
    store.migrate()
    existing_cid = store.get_or_create_corpus("existing", "/existing", "moonshot")
    doc = Document.from_content("collision body", source_path="dup.txt", title="dup.txt")
    store.add_document(doc, corpus_id=existing_cid)
    store.close()

    client = _client(cfg)
    data = {"corpus_name": "victim-corpus", "judge_provider": "moonshot"}
    files = {"files": ("dup.txt", b"collision body", "text/plain")}
    resp = client.post("/corpora/new", data=data, files=files)
    assert resp.status_code == 200
    assert "Ingest started" in resp.text

    latest = _latest_run_for_corpus(cfg, "victim-corpus")
    assert latest is not None
    run_kind, run_status, error_message = latest
    assert run_kind == "ingest"
    assert run_status == "failed"
    assert "already exists" in (error_message or "")


def _corpus_id_by_name(cfg: Config, name: str) -> str | None:
    store = AssertionStore(cfg.db_path)
    try:
        row = store._conn.execute(
            "SELECT corpus_id FROM corpora WHERE corpus_name = ?", (name,)
        ).fetchone()
        return row[0] if row else None
    finally:
        store.close()


def test_delete_corpus_removes_it_and_redirects(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg, name="deleteme")
    client = _client(cfg)
    resp = client.post(f"/corpora/{cid}/delete")
    assert resp.status_code == 200
    assert resp.headers.get("HX-Redirect") == "/"
    assert _corpus_id_by_name(cfg, "deleteme") is None


def test_delete_corpus_404_on_unknown_id(tmp_path: Path) -> None:
    client = _client(_config(tmp_path))
    assert client.post("/corpora/does-not-exist/delete").status_code == 404


def test_failed_ingest_corpus_can_be_deleted_then_name_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Escape hatch for ADR-0019's 'keep the corpus on failure': a failed ingest
    keeps the corpus, but Delete frees the name so the same name is reusable —
    closing the orphan-corpus 409-on-retry trap."""
    cfg = _config(tmp_path)
    client = _client(cfg)

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic ingest failure")

    monkeypatch.setattr("consistency_checker.web.app._ingest_uploaded_paths", _boom)
    data = {"corpus_name": "retry-name", "judge_provider": "moonshot"}
    files = {"files": ("a.txt", b"some content", "text/plain")}
    assert client.post("/corpora/new", data=data, files=files).status_code == 200

    # Same name is blocked while the failed corpus lingers...
    blocked = client.post("/corpora/new", data=data, files=files)
    assert blocked.status_code == 409
    assert "already exists" in blocked.text

    # ...until the user deletes it, after which the name is reusable.
    cid = _corpus_id_by_name(cfg, "retry-name")
    assert cid is not None
    assert client.post(f"/corpora/{cid}/delete").status_code == 200
    monkeypatch.undo()  # let the retry actually ingest
    assert client.post("/corpora/new", data=data, files=files).status_code == 200


def _latest_run_row(cfg: Config, corpus_name: str) -> dict[str, object] | None:
    """Full latest-run row for a corpus as a dict, or None."""
    store = AssertionStore(cfg.db_path)
    store._conn.row_factory = sqlite3.Row
    try:
        cid_row = store._conn.execute(
            "SELECT corpus_id FROM corpora WHERE corpus_name = ?", (corpus_name,)
        ).fetchone()
        if cid_row is None:
            return None
        row = store._conn.execute(
            "SELECT * FROM pipeline_runs WHERE corpus_id = ? "
            "ORDER BY started_at DESC, run_id DESC LIMIT 1",
            (cid_row[0],),
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        store.close()


def test_create_corpus_ingest_runs_as_background_job(tmp_path: Path) -> None:
    """ADR-0019: a normal upload returns instantly and completes as a DONE
    ingest run (kind='ingest') with the file counters filled in."""
    cfg = _config(tmp_path)
    client = _client(cfg)
    data = {"corpus_name": "bg-corpus", "judge_provider": "moonshot"}
    files = {"files": ("doc.txt", b"Some ordinary body text.", "text/plain")}
    resp = client.post("/corpora/new", data=data, files=files)
    assert resp.status_code == 200
    assert "Ingest started" in resp.text
    assert resp.headers.get("HX-Trigger") == "corpus-created"

    row = _latest_run_row(cfg, "bg-corpus")
    assert row is not None
    assert row["run_kind"] == "ingest"
    assert row["run_status"] == "done"
    assert row["n_files_total"] == 1
    assert row["n_files_done"] == 1
    assert row["finished_at"] is not None

    # The upload staging dir is cleaned up after the job — no per-upload leak.
    uploads_root = cfg.data_dir / "uploads"
    assert not uploads_root.exists() or list(uploads_root.iterdir()) == []


def test_ingest_surfaces_empty_text_files(tmp_path: Path) -> None:
    """A file that yields no extractable text (scanned image / OCR unavailable)
    is recorded in the run notes rather than silently counted as success (C)."""
    cfg = _config(tmp_path)
    client = _client(cfg)
    data = {"corpus_name": "empty-corpus", "judge_provider": "moonshot"}
    files = {"files": ("scan.txt", b"   \n  \t ", "text/plain")}
    resp = client.post("/corpora/new", data=data, files=files)
    assert resp.status_code == 200

    row = _latest_run_row(cfg, "empty-corpus")
    assert row is not None
    assert row["run_status"] == "done"
    assert "scan.txt" in _empty_text_files_from_notes(row["notes"])


def test_ingest_progress_label_render() -> None:
    """The ingest progress label branches on status and HTML-escapes filenames."""
    running = _ingest_progress_label(
        {"n_files_total": 3, "n_files_done": 1, "n_assertions": 7}, "running"
    )
    assert "Ingesting" in running and "1/3 files" in running and "7 assertions" in running

    done = _ingest_progress_label(
        {
            "n_files_total": 1,
            "n_files_done": 1,
            "n_assertions": 0,
            "notes": '{"empty_text_files": ["<scan>.pdf"]}',
        },
        "done",
    )
    assert "no extractable text" in done
    assert "&lt;scan&gt;.pdf" in done  # escaped, not raw
    assert "<scan>" not in done

    failed = _ingest_progress_label({"error_message": "boom <x>"}, "failed")
    assert "Ingest failed" in failed and "boom &lt;x&gt;" in failed


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


# --- Phase 3: Run Check modal + per-corpus SSE (ADR-0017) ---------------


def test_findings_button_wired_to_run_modal(tmp_path: Path) -> None:
    """The findings-pane Run check button targets the per-corpus modal route."""
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg, name="Run-me")
    client = _client(cfg)
    resp = client.get("/?new_ui=1")
    assert resp.status_code == 200
    body = resp.text
    assert f'hx-get="/corpora/{cid}/run/modal"' in body
    # The Phase-1 placeholder hint must be gone from the Run check button row.
    assert 'title="Wired in Phase 3"' not in body


def test_run_modal_route_returns_dialog_fragment(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg, name="Modal-corpus")
    client = _client(cfg)
    resp = client.get(f"/corpora/{cid}/run/modal")
    assert resp.status_code == 200
    body = resp.text
    assert "<dialog" in body.lower()
    assert 'name="pairwise"' in body
    assert 'name="no_definitions"' in body
    assert 'name="deep"' in body
    assert 'name="max_cost"' in body
    assert f'hx-post="/corpora/{cid}/run"' in body
    # No full-page wrapper.
    assert "<html" not in body.lower()


def test_run_modal_404s_on_unknown_corpus(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    client = _client(cfg)
    resp = client.get("/corpora/does-not-exist/run/modal")
    assert resp.status_code == 404


def _seed_ingested_corpus(cfg: Config, *, name: str = "Bg-run") -> str:
    """Seed a corpus with two embedded assertions so FAISS exists on disk.

    The check background task opens FAISS without a ``dim`` arg, which
    requires the index to already exist — that only happens after ingest
    has embedded at least one assertion. Use this for tests that POST a run.
    """
    from consistency_checker.index.embedder import embed_pending
    from consistency_checker.index.faiss_store import FaissStore

    store = AssertionStore(cfg.db_path)
    store.migrate()
    cid = store.get_or_create_corpus(name, f"/{name}", "moonshot")
    doc_a = Document.from_content("Alpha body.", source_path="a.md", title="A")
    doc_b = Document.from_content("Beta body.", source_path="b.md", title="B")
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


def test_post_run_starts_background_task(tmp_path: Path) -> None:
    """POST /corpora/<id>/run begins a run row + emits run-started trigger."""
    cfg = _config(tmp_path)
    cid = _seed_ingested_corpus(cfg, name="Bg-run")
    client = _client(cfg)
    resp = client.post(
        f"/corpora/{cid}/run",
        data={"pairwise": "false", "no_definitions": "true", "max_cost": ""},
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers.get("HX-Trigger") == "run-started"
    assert "Run started" in resp.text or "started" in resp.text.lower()

    store = AssertionStore(cfg.db_path)
    try:
        rows = store._conn.execute(
            "SELECT run_id, corpus_id, run_status FROM pipeline_runs WHERE corpus_id = ?",
            (cid,),
        ).fetchall()
        assert len(rows) == 1
        # Background tasks run after the response in TestClient — by the time
        # we read here the run has either reached "done" (definitions only,
        # nothing to judge) or remained "pending"/"running". Either way the
        # row exists with the right corpus_id, which is what this test asserts.
        assert rows[0]["corpus_id"] == cid
    finally:
        store.close()


def test_post_run_begin_run_config_matches_cli_shape(tmp_path: Path) -> None:
    """ADR-0017 review: web /corpora/{id}/run's begin_run config dict must
    mirror the CLI's keys so audit-log replay sees one shape regardless of
    which surface started the run."""
    cfg = _config(tmp_path)
    cid = _seed_ingested_corpus(cfg, name="Replay-parity")
    client = _client(cfg)
    resp = client.post(
        f"/corpora/{cid}/run",
        data={"pairwise": "true", "no_definitions": "false", "deep": "true"},
    )
    assert resp.status_code == 200, resp.text

    store = AssertionStore(cfg.db_path)
    try:
        row = store._conn.execute(
            "SELECT config_json FROM pipeline_runs WHERE corpus_id = ?", (cid,)
        ).fetchone()
        assert row is not None
        cfg_json = row["config_json"]
        # CLI key vocabulary — not "deep" or "no_definitions".
        assert '"enable_multi_party": true' in cfg_json
        assert '"definitions_enabled": true' in cfg_json
        assert '"pairwise_enabled": true' in cfg_json
        # Quoted key check — bare "deep"/"no_definitions" would false-fail on
        # any future model name or value that contains those substrings.
        assert '"deep":' not in cfg_json
        assert '"no_definitions":' not in cfg_json
        # Missing-from-Phase-3 fields the review caught.
        assert "nli_contradiction_threshold" in cfg_json
        assert "gate_top_k" in cfg_json
        assert "gate_similarity_threshold" in cfg_json
        assert "max_triangles_per_run" in cfg_json
        assert "max_cost_usd" in cfg_json
    finally:
        store.close()


def test_post_run_deep_without_pairwise_rejected(tmp_path: Path) -> None:
    """deep=true with effective pairwise=false renders a 400 error in-modal."""
    cfg = _config(tmp_path)
    # cfg.pairwise_enabled defaults to False; pass pairwise="" so the config
    # default kicks in and deep is rejected.
    cid = _seed_one_corpus(cfg, name="Deep-no-pair")
    client = _client(cfg)
    resp = client.post(
        f"/corpora/{cid}/run",
        data={"pairwise": "", "deep": "true"},
    )
    assert resp.status_code == 400
    body = resp.text
    assert "deep" in body.lower()
    assert "pairwise" in body.lower()
    # No run row created on a 400.
    store = AssertionStore(cfg.db_path)
    try:
        rows = store._conn.execute(
            "SELECT 1 FROM pipeline_runs WHERE corpus_id = ?", (cid,)
        ).fetchall()
        assert rows == []
    finally:
        store.close()


def test_post_run_404s_on_unknown_corpus(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    client = _client(cfg)
    resp = client.post(
        "/corpora/does-not-exist/run",
        data={"pairwise": "false"},
    )
    assert resp.status_code == 404


def test_post_run_rejects_negative_max_cost(tmp_path: Path) -> None:
    """Web layer must match the CLI's min=0 guard on --max-cost."""
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg, name="Neg-cost")
    client = _client(cfg)
    resp = client.post(
        f"/corpora/{cid}/run",
        data={"pairwise": "false", "max_cost": "-1.0"},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert "max_cost" in str(body)


def test_post_run_rejects_infinite_max_cost(tmp_path: Path) -> None:
    """inf bypasses the cost ceiling check; the web route must refuse it."""
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg, name="Inf-cost")
    client = _client(cfg)
    resp = client.post(
        f"/corpora/{cid}/run",
        data={"pairwise": "false", "max_cost": "inf"},
    )
    assert resp.status_code == 422


def test_post_run_rejects_concurrent_run_on_same_corpus(tmp_path: Path) -> None:
    """Second submit while a run is pending/running returns 409 and does
    NOT spawn a parallel background task (which would race on FAISS)."""
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg, name="Busy-corpus")

    # Manually seed a 'running' run row to simulate an in-flight check.
    store = AssertionStore(cfg.db_path)
    AuditLogger(store).begin_run(corpus_id=cid, run_status="running")
    store.close()

    client = _client(cfg)
    resp = client.post(
        f"/corpora/{cid}/run",
        data={"pairwise": "false"},
    )
    assert resp.status_code == 409
    assert "in progress" in resp.text.lower()

    # No second run row created.
    store = AssertionStore(cfg.db_path)
    try:
        rows = store._conn.execute(
            "SELECT run_id FROM pipeline_runs WHERE corpus_id = ?", (cid,)
        ).fetchall()
        assert len(rows) == 1
    finally:
        store.close()


def test_progress_sse_unknown_status_renders_safe_label(tmp_path: Path) -> None:
    """If a future migration adds a run_status value we don't recognise,
    the rendered HTML must not interpolate the raw string into the SSE
    payload (defence against XSS via the sidebar innerHTML swap)."""
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg, name="Mystery-status")

    # Seed a run, then mutate the run_status to a string that would be
    # dangerous if interpolated raw.
    store = AssertionStore(cfg.db_path)
    rid = AuditLogger(store).begin_run(corpus_id=cid, run_status="pending")
    store._conn.execute(
        "UPDATE pipeline_runs SET run_status = ? WHERE run_id = ?",
        ("<script>alert(1)</script>", rid),
    )
    store._conn.commit()
    store.close()

    # Force-finite SSE so the test doesn't hang.
    import consistency_checker.web.app as app_module

    poll = app_module.PROGRESS_POLL_SECONDS
    max_iter = app_module.PROGRESS_MAX_ITERATIONS
    tail = app_module.PROGRESS_DONE_TAIL_SECONDS
    app_module.PROGRESS_POLL_SECONDS = 0
    app_module.PROGRESS_MAX_ITERATIONS = 1
    app_module.PROGRESS_DONE_TAIL_SECONDS = 0
    try:
        client = _client(cfg)
        with client.stream("GET", f"/corpora/{cid}/progress") as resp:
            chunks = []
            for chunk in resp.iter_text():
                chunks.append(chunk)
                if len(chunks) >= 3:
                    break
            body = "".join(chunks)
        # The htmx-sse extension only innerHTML-swaps named events that
        # match `sse-swap="snapshot"`. Defence-in-depth: the rendered HTML
        # fragment for the `snapshot` event must NEVER contain raw status
        # text from the DB. (The unnamed JSON debug event is consumed only
        # by tests/native JS that parse JSON, so its contents are out of
        # scope for this XSS check.)
        snapshot_event_html = "".join(
            block for block in body.split("\n\n") if block.startswith("event: snapshot")
        )
        assert "<script>" not in snapshot_event_html
        assert "Unknown" in snapshot_event_html
    finally:
        app_module.PROGRESS_POLL_SECONDS = poll
        app_module.PROGRESS_MAX_ITERATIONS = max_iter
        app_module.PROGRESS_DONE_TAIL_SECONDS = tail


def _drain_sse(client: TestClient, url: str, *, max_events: int = 4) -> list[str]:
    """Read SSE chunks from ``url`` until we either see a 'done' event or hit
    ``max_events`` raw chunks. Returns the list of UTF-8 chunks."""
    chunks: list[str] = []
    with client.stream("GET", url) as resp:
        assert resp.status_code == 200
        for chunk in resp.iter_text():
            chunks.append(chunk)
            if "event: done" in chunk or len(chunks) >= max_events:
                break
    return chunks


def test_progress_sse_emits_snapshot_for_active_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An in-progress run should produce at least one snapshot event with the
    run_id payload before the stream closes."""
    cfg = _config(tmp_path)
    cid, run_id, _rationale = _seed_corpus_with_finding(cfg)
    # Force the most-recent run back to "running" so the SSE generator
    # treats it as in-progress.
    store = AssertionStore(cfg.db_path)
    try:
        with store._conn:
            store._conn.execute(
                "UPDATE pipeline_runs SET run_status = 'running', finished_at = NULL "
                "WHERE run_id = ?",
                (run_id,),
            )
    finally:
        store.close()

    # Make the generator finite + fast: poll once, cap tail at one tick.
    from consistency_checker.web import app as app_module

    monkeypatch.setattr(app_module, "PROGRESS_POLL_SECONDS", 0.0)
    monkeypatch.setattr(app_module, "PROGRESS_MAX_ITERATIONS", 2)
    monkeypatch.setattr(app_module, "PROGRESS_DONE_TAIL_SECONDS", 0.0)

    client = _client(cfg)
    chunks = _drain_sse(client, f"/corpora/{cid}/progress", max_events=6)
    combined = "".join(chunks)
    assert "event: snapshot" in combined
    assert run_id in combined
    assert '"status": "running"' in combined


def test_progress_sse_emits_done_then_closes_for_finished_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A done run should emit one final snapshot + a 'done' event, then close."""
    cfg = _config(tmp_path)
    cid, _run_id, _rationale = _seed_corpus_with_finding(cfg)
    # Seeded run is already in 'done' state via end_run().

    from consistency_checker.web import app as app_module

    monkeypatch.setattr(app_module, "PROGRESS_POLL_SECONDS", 0.0)
    monkeypatch.setattr(app_module, "PROGRESS_MAX_ITERATIONS", 4)
    monkeypatch.setattr(app_module, "PROGRESS_DONE_TAIL_SECONDS", 0.0)

    client = _client(cfg)
    chunks = _drain_sse(client, f"/corpora/{cid}/progress", max_events=8)
    combined = "".join(chunks)
    assert "event: snapshot" in combined
    assert "event: done" in combined
    assert '"status": "done"' in combined


def test_progress_sse_404s_on_unknown_corpus(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    client = _client(cfg)
    resp = client.get("/corpora/does-not-exist/progress")
    assert resp.status_code == 404


# --- Phase 4: Slide-over drawers (ADR-0017) -----------------------------


def _seed_corpus_with_definition_finding(
    cfg: Config, *, name: str = "Definitions corpus", term: str = "EBITDA"
) -> tuple[str, str]:
    """Seed a corpus + run with one definition_divergent finding. Returns
    (corpus_id, term) so the assertion can check the rendered drawer."""
    from consistency_checker.check.definition_checker import (
        DefinitionFinding,
        DefinitionPair,
    )
    from consistency_checker.check.definition_judge import DefinitionJudgeVerdict

    store = AssertionStore(cfg.db_path)
    store.migrate()
    cid = store.get_or_create_corpus(name, f"/{name}", "moonshot")
    doc_a = Document.from_content("Alpha body.", source_path="alpha.md", title="Alpha")
    doc_b = Document.from_content("Beta body.", source_path="beta.md", title="Beta")
    store.add_document(doc_a, corpus_id=cid)
    store.add_document(doc_b, corpus_id=cid)
    a = Assertion.build(
        doc_a.doc_id,
        f"{term} excludes restructuring charges.",
        kind="definition",
        term=term,
    )
    b = Assertion.build(
        doc_b.doc_id,
        f"{term} includes restructuring charges.",
        kind="definition",
        term=term,
    )
    store.add_assertions([a, b])
    audit = AuditLogger(store)
    run_id = audit.begin_run(corpus_id=cid)
    audit.record_definition_finding(
        run_id,
        finding=DefinitionFinding(
            pair=DefinitionPair(a=a, b=b, canonical_term=term),
            verdict=DefinitionJudgeVerdict(
                assertion_a_id=a.assertion_id,
                assertion_b_id=b.assertion_id,
                verdict="definition_divergent",
                rationale=f"Different treatment of restructuring in {term}.",
                evidence_spans=[],
            ),
        ),
    )
    audit.end_run(run_id, n_assertions=2, n_pairs_gated=1, n_pairs_judged=1, n_findings=1)
    store.close()
    return cid, term


def test_drawer_buttons_wired_when_corpus_active(tmp_path: Path) -> None:
    """The Assertions/Definitions/Stats buttons must point at the per-corpus
    drawer routes and must NOT be disabled once a corpus is active."""
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg, name="Drawer-target")
    client = _client(cfg)
    resp = client.get("/?new_ui=1")
    assert resp.status_code == 200
    body = resp.text
    assert f'hx-get="/corpora/{cid}/drawer/assertions"' in body
    assert f'hx-get="/corpora/{cid}/drawer/definitions"' in body
    assert f'hx-get="/corpora/{cid}/drawer/stats"' in body
    # Phase 1 placeholder hint must be gone from the drawer button row.
    assert 'title="Wired in Phase 4"' not in body


def test_drawer_assertions_returns_fragment_with_close_button(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg, name="Assertions-drawer")
    # Seed one assertion so the drawer's table has visible content.
    store = AssertionStore(cfg.db_path)
    store.migrate()
    doc = Document.from_content("Body.", source_path="a.md", title="A")
    store.add_document(doc, corpus_id=cid)
    seeded_text = "Revenue grew 12% in fiscal 2025."
    store.add_assertions([Assertion.build(doc.doc_id, seeded_text)])
    store.close()

    client = _client(cfg)
    resp = client.get(f"/corpora/{cid}/drawer/assertions")
    assert resp.status_code == 200
    body = resp.text
    lower = body.lower()
    assert "<html" not in lower
    assert "<!doctype" not in lower
    assert '<aside class="cc-drawer"' in body
    assert "cc-drawer-close" in body
    assert seeded_text in body


def test_drawer_definitions_returns_fragment(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid, term = _seed_corpus_with_definition_finding(cfg, name="Def-drawer")
    client = _client(cfg)
    resp = client.get(f"/corpora/{cid}/drawer/definitions")
    assert resp.status_code == 200
    body = resp.text
    lower = body.lower()
    assert "<html" not in lower
    assert '<aside class="cc-drawer"' in body
    # The term should appear in the drawer body.
    assert term in body


def test_drawer_stats_returns_fragment(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid, _run_id, _rationale = _seed_corpus_with_finding(cfg, name="Stats-drawer")
    client = _client(cfg)
    resp = client.get(f"/corpora/{cid}/drawer/stats")
    assert resp.status_code == 200
    body = resp.text
    lower = body.lower()
    assert "<html" not in lower
    assert '<aside class="cc-drawer"' in body
    # n_assertions=2 + n_findings=1 from _seed_corpus_with_finding.
    assert "Stats" in body


def test_drawer_404s_on_unknown_corpus(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    client = _client(cfg)
    for view in ("assertions", "definitions", "stats"):
        resp = client.get(f"/corpora/does-not-exist/drawer/{view}")
        assert resp.status_code == 404, f"{view} should 404 on unknown corpus"


def test_drawer_includes_focus_trap_script(tmp_path: Path) -> None:
    """ADR-0017 line 47: drawer focus management is the explicit Phase 4
    deliverable. The rendered fragment must include the keydown listener +
    Escape handler + listener-cleanup path so accessibility doesn't silently
    regress."""
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg, name="A11y-drawer")
    client = _client(cfg)
    resp = client.get(f"/corpora/{cid}/drawer/assertions")
    assert resp.status_code == 200
    body = resp.text
    assert "addEventListener('keydown'" in body
    assert "Escape" in body
    assert "data-cc-drawer-backdrop" in body
    # Listener-leak regression: every close path (X button, Esc, click
    # outside, re-open) must route through close() so the document-level
    # listeners are removed. The Phase-4 review caught the X-button
    # onclick previously bypassing this; assert both halves now.
    assert "removeEventListener('keydown'" in body
    assert "removeEventListener('click'" in body
    assert "window.__ccDrawerClose" in body


def test_drawer_definitions_toggle_rewires_to_drawer_route(tmp_path: Path) -> None:
    """The 'Show reviewed' toggle in cc_definitions.html hardcodes a legacy
    /tabs/definitions hx-get targeting #cc-tab-content. Inside the drawer
    both are wrong — the route must override them so the toggle refreshes
    the drawer in-place instead of silently no-op'ing.

    The toggle only renders when the corpus has at least one run, so seed
    a run via the existing definition-finding helper used by other tests.
    """
    cfg = _config(tmp_path)
    cid, _rid, _rationale = _seed_corpus_with_finding(cfg, name="Toggle-rewire")
    client = _client(cfg)
    resp = client.get(f"/corpora/{cid}/drawer/definitions")
    assert resp.status_code == 200
    body = resp.text
    # Drawer-scoped refresh URL, not the legacy /tabs/definitions.
    assert f"/corpora/{cid}/drawer/definitions" in body
    assert 'hx-target="#cc-drawer-region"' in body
    # Legacy target must NOT appear inside the rendered drawer fragment.
    assert "#cc-tab-content" not in body


def test_drawer_assertions_pagination_targets_drawer(tmp_path: Path) -> None:
    """PR #77 review caught that cc__pagination.html's hardcoded
    hx-target=#cc-tab-content silently no-op'd inside the drawer. The
    assertions drawer route must override the target so Prev/Next
    actually paginate."""
    cfg = _config(tmp_path)

    # Seed enough assertions to force pagination (DRAWER_PAGE_SIZE=25).
    store = AssertionStore(cfg.db_path)
    store.migrate()
    cid = store.get_or_create_corpus("Paginated", "/paginated", "moonshot")
    doc = Document.from_content("body", source_path="d.md", title="d")
    store.add_document(doc, corpus_id=cid)
    store.add_assertions([Assertion.build(doc.doc_id, f"fact {i}") for i in range(30)])
    store.close()

    client = _client(cfg)
    resp = client.get(f"/corpora/{cid}/drawer/assertions")
    assert resp.status_code == 200
    body = resp.text
    # Pagination renders.
    assert "Page 1 of 2" in body
    # Next link points at the drawer route + drawer target, not the legacy tab.
    assert f"/corpora/{cid}/drawer/assertions?page=2" in body
    assert 'hx-target="#cc-drawer-region"' in body
    assert "#cc-tab-content" not in body


def test_drawer_definitions_suppresses_global_reviewer_counter(tmp_path: Path) -> None:
    """PR #77 review caught that _count_total_findings and
    count_reviewer_verdicts are both global. Inside a per-corpus drawer
    that produces a mismatched 'X of N reviewed' ratio. Until the
    reviewer-verdicts schema supports a corpus join, the drawer should
    suppress the counter rather than show a wrong one."""
    cfg = _config(tmp_path)
    cid, _rid, _rationale = _seed_corpus_with_finding(cfg, name="Counter-suppressed")
    client = _client(cfg)
    resp = client.get(f"/corpora/{cid}/drawer/definitions")
    assert resp.status_code == 200
    body = resp.text
    assert "of " not in body or "reviewed" not in body or "cc-progress-count" not in body


# --- Phase 5: inline verdicts + filter chips + cost gauge (ADR-0017) ----


def _moonshot_config(tmp_path: Path) -> Config:
    """Cost-gauge tests need a non-fixture provider so per-call costs are > 0."""
    return Config(
        corpus_dir=tmp_path / "corpus",
        judge_provider="moonshot",
        judge_model="kimi-test",
        data_dir=tmp_path / "store",
        log_dir=tmp_path / "logs",
        embedder_model="hash",
        nli_model="fixture",
        gate_similarity_threshold=-1.0,
        nli_contradiction_threshold=0.0,
    )


def _pair_key_for_finding(cfg: Config, run_id: str) -> tuple[str, str]:
    """Lookup pair_key + detector_type for the single finding seeded by helper."""
    store = AssertionStore(cfg.db_path)
    try:
        audit = AuditLogger(store)
        findings = list(audit.iter_findings(run_id=run_id))
        assert len(findings) == 1
        f = findings[0]
        return ":".join(sorted([f.assertion_a_id, f.assertion_b_id])), "contradiction"
    finally:
        store.close()


def test_finding_verdict_button_posts_inline(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid, run_id, _rationale = _seed_corpus_with_finding(cfg, name="Verdict-target")
    pair_key, detector_type = _pair_key_for_finding(cfg, run_id)

    client = _client(cfg)
    resp = client.get(f"/?new_ui=1&corpus={cid}")
    assert resp.status_code == 200
    body = resp.text
    assert 'hx-post="/verdicts"' in body
    assert pair_key in body
    assert '"detector_type": "contradiction"' in body
    assert '"verdict": "confirmed"' in body

    # POST the verdict and verify it was persisted.
    resp2 = client.post(
        "/verdicts",
        data={
            "pair_key": pair_key,
            "detector_type": detector_type,
            "verdict": "confirmed",
        },
    )
    assert resp2.status_code == 200, resp2.text

    store = AssertionStore(cfg.db_path)
    try:
        audit = AuditLogger(store)
        rv = audit.get_reviewer_verdicts_bulk([(pair_key, "contradiction")])
        assert (pair_key, "contradiction") in rv
        assert rv[(pair_key, "contradiction")].verdict == "confirmed"
    finally:
        store.close()


def test_finding_shows_marked_state_after_verdict_set(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid, run_id, _rationale = _seed_corpus_with_finding(cfg, name="Marked-corpus")
    pair_key, _dt = _pair_key_for_finding(cfg, run_id)

    store = AssertionStore(cfg.db_path)
    try:
        AuditLogger(store).set_reviewer_verdict(
            pair_key=pair_key,
            detector_type="contradiction",
            verdict="false_positive",
        )
    finally:
        store.close()

    client = _client(cfg)
    resp = client.get(f"/?new_ui=1&corpus={cid}")
    assert resp.status_code == 200
    body = resp.text
    assert "Marked" in body
    # VERDICT_LABELS["false_positive"] == "Not an issue".
    assert "Not an issue" in body
    assert "undo" in body
    assert 'hx-post="/verdicts/undo"' in body
    # PR #77 review fix: the undo button MUST send prior_verdict="" so the
    # verdict row is deleted. Sending the current verdict re-applies it.
    assert '"prior_verdict": ""' in body


def test_finding_undo_button_clears_reviewer_verdict(tmp_path: Path) -> None:
    """End-to-end: pre-set a verdict, hit the undo URL with the empty
    prior_verdict that the marked-card button sends, and assert the
    reviewer_verdicts row is gone."""
    cfg = _config(tmp_path)
    _cid, run_id, _rationale = _seed_corpus_with_finding(cfg, name="Undo-clear")
    pair_key, _dt = _pair_key_for_finding(cfg, run_id)

    store = AssertionStore(cfg.db_path)
    audit = AuditLogger(store)
    audit.set_reviewer_verdict(
        pair_key=pair_key, detector_type="contradiction", verdict="confirmed"
    )
    pre = audit.get_reviewer_verdicts_bulk([(pair_key, "contradiction")])
    assert pre.get((pair_key, "contradiction")) is not None
    store.close()

    client = _client(cfg)
    resp = client.post(
        "/verdicts/undo",
        data={
            "pair_key": pair_key,
            "detector_type": "contradiction",
            "prior_verdict": "",
        },
    )
    assert resp.status_code == 200

    store = AssertionStore(cfg.db_path)
    audit = AuditLogger(store)
    post = audit.get_reviewer_verdicts_bulk([(pair_key, "contradiction")])
    assert (pair_key, "contradiction") not in post
    store.close()


def test_filter_chip_open_excludes_marked_findings(tmp_path: Path) -> None:
    """Seed two findings, mark one, filter=open shows only the unmarked one."""
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg, name="Two-findings")

    # Seed two findings directly via the audit logger so we control both
    # assertion pairs separately.
    store = AssertionStore(cfg.db_path)
    doc_a = Document.from_content("body a.", source_path="a.md", title="A")
    doc_b = Document.from_content("body b.", source_path="b.md", title="B")
    store.add_document(doc_a, corpus_id=cid)
    store.add_document(doc_b, corpus_id=cid)
    a1 = Assertion.build(doc_a.doc_id, "Revenue grew 12%.")
    b1 = Assertion.build(doc_b.doc_id, "Revenue declined 5%.")
    a2 = Assertion.build(doc_a.doc_id, "EBITDA up 8%.")
    b2 = Assertion.build(doc_b.doc_id, "EBITDA down 3%.")
    store.add_assertions([a1, b1, a2, b2])

    audit = AuditLogger(store)
    run_id = audit.begin_run(corpus_id=cid)
    audit.record_finding(
        run_id,
        candidate=CandidatePair(a=a1, b=b1, score=0.9),
        nli=NliResult.from_scores(p_contradiction=0.8, p_entailment=0.1, p_neutral=0.1),
        verdict=JudgeVerdict(
            assertion_a_id=a1.assertion_id,
            assertion_b_id=b1.assertion_id,
            verdict="contradiction",
            rationale="Revenue contradiction.",
            evidence_spans=[],
        ),
    )
    audit.record_finding(
        run_id,
        candidate=CandidatePair(a=a2, b=b2, score=0.9),
        nli=NliResult.from_scores(p_contradiction=0.8, p_entailment=0.1, p_neutral=0.1),
        verdict=JudgeVerdict(
            assertion_a_id=a2.assertion_id,
            assertion_b_id=b2.assertion_id,
            verdict="contradiction",
            rationale="EBITDA contradiction.",
            evidence_spans=[],
        ),
    )
    audit.end_run(run_id, n_assertions=4, n_pairs_gated=2, n_pairs_judged=2, n_findings=2)

    # Mark the first finding.
    pair_key_1 = ":".join(sorted([a1.assertion_id, b1.assertion_id]))
    audit.set_reviewer_verdict(
        pair_key=pair_key_1, detector_type="contradiction", verdict="confirmed"
    )
    store.close()

    client = _client(cfg)
    resp = client.get(f"/?new_ui=1&corpus={cid}&filter=open")
    assert resp.status_code == 200
    body = resp.text
    assert "EBITDA contradiction." in body
    assert "Revenue contradiction." not in body


def test_filter_chip_confirmed_includes_only_confirmed(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid, run_id, _rationale = _seed_corpus_with_finding(cfg, name="Confirmed-only")
    pair_key, _dt = _pair_key_for_finding(cfg, run_id)

    store = AssertionStore(cfg.db_path)
    try:
        AuditLogger(store).set_reviewer_verdict(
            pair_key=pair_key, detector_type="contradiction", verdict="confirmed"
        )
    finally:
        store.close()

    client = _client(cfg)
    resp_confirmed = client.get(f"/?new_ui=1&corpus={cid}&filter=confirmed")
    assert resp_confirmed.status_code == 200
    body_conf = resp_confirmed.text
    # The seeded rationale appears for the single (marked) finding.
    assert "Opposite revenue signs" in body_conf

    resp_fp = client.get(f"/?new_ui=1&corpus={cid}&filter=false_positive")
    assert resp_fp.status_code == 200
    body_fp = resp_fp.text
    # No findings should match FP filter.
    assert "Opposite revenue signs" not in body_fp


def test_filter_chip_counts_match_actual_findings(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid, run_id, _rationale = _seed_corpus_with_finding(cfg, name="Count-corpus")
    pair_key, _dt = _pair_key_for_finding(cfg, run_id)

    # Mark the single finding as confirmed; counts should be all=1, open=0,
    # confirmed=1, false_positive=0, dismissed=0.
    store = AssertionStore(cfg.db_path)
    try:
        AuditLogger(store).set_reviewer_verdict(
            pair_key=pair_key, detector_type="contradiction", verdict="confirmed"
        )
    finally:
        store.close()

    client = _client(cfg)
    resp = client.get(f"/?new_ui=1&corpus={cid}")
    assert resp.status_code == 200
    body = resp.text
    assert "All (1)" in body
    assert "Open (0)" in body
    assert "Confirmed (1)" in body
    assert "FP (0)" in body
    assert "Dismissed (0)" in body


def test_cost_gauge_renders_estimate_when_run_has_judged_pairs(tmp_path: Path) -> None:
    cfg = _moonshot_config(tmp_path)
    cid = _seed_one_corpus(cfg, name="Cost-corpus")
    # Seed a run with n_pairs_judged=5 directly.
    store = AssertionStore(cfg.db_path)
    audit = AuditLogger(store)
    run_id = audit.begin_run(corpus_id=cid)
    audit.end_run(run_id, n_assertions=10, n_pairs_gated=5, n_pairs_judged=5, n_findings=0)
    store.close()

    client = _client(cfg)
    resp = client.get(f"/?new_ui=1&corpus={cid}")
    assert resp.status_code == 200
    body = resp.text
    assert "Est. spent" in body
    # moonshot per_call_high = 0.001; 5 * 0.001 = 0.0050 (rendered to 4 decimals).
    assert "$0.0050" in body


def test_cost_gauge_sse_emits_cost_update_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _moonshot_config(tmp_path)
    cid, run_id, _rationale = _seed_corpus_with_finding(cfg, name="Cost-sse")

    # Flip the seeded run back to running so the SSE sees an in-progress run.
    store = AssertionStore(cfg.db_path)
    try:
        with store._conn:
            store._conn.execute(
                "UPDATE pipeline_runs SET run_status = 'running', finished_at = NULL "
                "WHERE run_id = ?",
                (run_id,),
            )
    finally:
        store.close()

    from consistency_checker.web import app as app_module

    monkeypatch.setattr(app_module, "PROGRESS_POLL_SECONDS", 0.0)
    monkeypatch.setattr(app_module, "PROGRESS_MAX_ITERATIONS", 2)
    monkeypatch.setattr(app_module, "PROGRESS_DONE_TAIL_SECONDS", 0.0)

    client = _client(cfg)
    chunks = _drain_sse(client, f"/corpora/{cid}/progress", max_events=8)
    combined = "".join(chunks)
    assert "event: cost_update" in combined
    assert "Est. spent" in combined


def test_cost_gauge_shows_dashes_when_no_active_corpus(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    client = _client(cfg)
    resp = client.get("/?new_ui=1")
    assert resp.status_code == 200
    body = resp.text
    # No corpora → no active corpus → gauge fragment renders the placeholder.
    assert 'class="cc-cost-spent">--</span>' in body
    assert 'class="cc-cost-ceiling">--</span>' in body


# --- UI review fixes: in-place verdict card, live toast, live chip counts ----


def test_verdict_post_swaps_card_actions_in_place(tmp_path: Path) -> None:
    """POST /verdicts returns the finding's actions span (by id) in marked
    state so the card flips in place, plus an HX-Trigger to refresh chips."""
    cfg = _config(tmp_path)
    _cid, run_id, _rationale = _seed_corpus_with_finding(cfg, name="Inplace")
    pair_key, detector_type = _pair_key_for_finding(cfg, run_id)
    pk_safe = pair_key.replace(":", "-")

    client = _client(cfg)
    resp = client.post(
        "/verdicts",
        data={"pair_key": pair_key, "detector_type": detector_type, "verdict": "confirmed"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.text
    # The main-target swap re-renders the actions span by its stable id...
    assert f'id="cc-actions-{pk_safe}-{detector_type}"' in body
    # ...in marked state (VERDICT_LABELS["confirmed"] == "Real issue").
    assert "Marked" in body and "Real issue" in body
    assert 'hx-post="/verdicts/undo"' in body
    # Toast now OOB-inserts into the region rather than a missing #cc-toast.
    assert 'hx-swap-oob="afterbegin:#cc-toast-region"' in body
    # Chips self-refresh on this event.
    assert resp.headers.get("HX-Trigger") == "verdict-changed"


def test_open_finding_buttons_target_actions_span(tmp_path: Path) -> None:
    """The open-state verdict buttons target their own #cc-actions span
    (outerHTML), not the toast region."""
    cfg = _config(tmp_path)
    cid, run_id, _rationale = _seed_corpus_with_finding(cfg, name="Targets")
    pair_key, detector_type = _pair_key_for_finding(cfg, run_id)
    pk_safe = pair_key.replace(":", "-")

    client = _client(cfg)
    body = client.get(f"/?new_ui=1&corpus={cid}").text
    assert f'hx-target="#cc-actions-{pk_safe}-{detector_type}"' in body
    assert 'hx-swap="outerHTML"' in body


def test_verdict_undo_returns_open_actions_not_redirect(tmp_path: Path) -> None:
    """Undo now swaps the card back to open state in place (no HX-Redirect)."""
    cfg = _config(tmp_path)
    _cid, run_id, _rationale = _seed_corpus_with_finding(cfg, name="UndoInplace")
    pair_key, detector_type = _pair_key_for_finding(cfg, run_id)
    pk_safe = pair_key.replace(":", "-")

    store = AssertionStore(cfg.db_path)
    try:
        AuditLogger(store).set_reviewer_verdict(
            pair_key=pair_key, detector_type=detector_type, verdict="confirmed"
        )
    finally:
        store.close()

    client = _client(cfg)
    resp = client.post(
        "/verdicts/undo",
        data={"pair_key": pair_key, "detector_type": detector_type, "prior_verdict": ""},
    )
    assert resp.status_code == 200
    assert "HX-Redirect" not in resp.headers
    assert resp.headers.get("HX-Trigger") == "verdict-changed"
    body = resp.text
    assert f'id="cc-actions-{pk_safe}-{detector_type}"' in body
    # Back to open state: the three verdict buttons are present again.
    assert "✓ Confirmed" in body and "✗ False positive" in body


def test_chips_endpoint_returns_live_counts(tmp_path: Path) -> None:
    """GET /corpora/{id}/chips renders the chip bar with self-refresh wiring
    and recomputed counts; marking a finding moves it out of `open`."""
    cfg = _config(tmp_path)
    cid, run_id, _rationale = _seed_corpus_with_finding(cfg, name="Chips")
    pair_key, detector_type = _pair_key_for_finding(cfg, run_id)

    client = _client(cfg)
    before = client.get(f"/corpora/{cid}/chips").text
    assert 'hx-trigger="verdict-changed from:body"' in before
    assert "Open (1)" in before and "Confirmed (0)" in before

    store = AssertionStore(cfg.db_path)
    try:
        AuditLogger(store).set_reviewer_verdict(
            pair_key=pair_key, detector_type=detector_type, verdict="confirmed"
        )
    finally:
        store.close()

    after = client.get(f"/corpora/{cid}/chips").text
    assert "Open (0)" in after and "Confirmed (1)" in after


def test_chips_endpoint_404s_on_unknown_corpus(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    client = _client(cfg)
    assert client.get("/corpora/nope/chips").status_code == 404


def test_chips_and_rows_are_keyboard_accessible(tmp_path: Path) -> None:
    """Filter chips and corpus rows are real <a href> with ARIA so they are
    focusable and degrade without JavaScript."""
    cfg = _config(tmp_path)
    cid, _run_id, _rationale = _seed_corpus_with_finding(cfg, name="A11y")

    client = _client(cfg)
    body = client.get(f"/?new_ui=1&corpus={cid}").text
    # Chips: real <a href> with aria-current on the active one — NOT a broken
    # tablist (no tabpanel/arrow-key contract exists).
    assert f'href="/?corpus={cid}&filter=open"' in body
    assert 'aria-current="true"' in body  # the active "all" chip
    assert 'role="tab"' not in body
    assert "aria-selected" not in body
    # Corpus row in the sidebar is a real link.
    assert f'href="/?corpus={cid}"' in body


# --- add files to existing corpus + rename corpus (CRUD) ---------------------


def test_add_files_modal_renders(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg, name="Growable")
    client = _client(cfg)
    body = client.get(f"/corpora/{cid}/add-files/modal").text
    assert "Add files" in body and "Growable" in body
    assert f'hx-post="/corpora/{cid}/add"' in body


def test_add_files_ingests_into_existing_corpus(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg, name="Growable")
    client = _client(cfg)
    resp = client.post(
        f"/corpora/{cid}/add",
        files={"files": ("more.txt", b"some additional content", "text/plain")},
    )
    assert resp.status_code == 200
    assert "Files added" in resp.text
    assert resp.headers.get("HX-Trigger") == "corpus-created"
    # Background ingest runs synchronously under TestClient -> a done ingest run.
    row = _latest_run_row(cfg, "Growable")
    assert row is not None
    assert row["run_kind"] == "ingest" and row["run_status"] == "done"
    assert row["n_files_total"] == 1


def test_add_files_requires_at_least_one_file(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg)
    client = _client(cfg)
    resp = client.post(f"/corpora/{cid}/add")
    assert resp.status_code == 400
    assert "at least one file" in resp.text


def test_add_files_blocked_by_active_run(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg)
    store = AssertionStore(cfg.db_path)
    store.migrate()
    AuditLogger(store).begin_run(corpus_id=cid, run_status="running", run_kind="ingest")
    store.close()
    client = _client(cfg)
    resp = client.post(f"/corpora/{cid}/add", files={"files": ("x.txt", b"body", "text/plain")})
    assert resp.status_code == 409
    assert "already in progress" in resp.text


def test_add_files_404_on_unknown_corpus(tmp_path: Path) -> None:
    client = _client(_config(tmp_path))
    assert client.get("/corpora/nope/add-files/modal").status_code == 404
    assert (
        client.post("/corpora/nope/add", files={"files": ("x.txt", b"b", "text/plain")}).status_code
        == 404
    )


def test_rename_modal_prefills_current_name(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg, name="Old name")
    client = _client(cfg)
    body = client.get(f"/corpora/{cid}/rename/modal").text
    assert "Rename corpus" in body
    assert 'value="Old name"' in body


def test_rename_corpus_succeeds(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg, name="Before")
    client = _client(cfg)
    resp = client.post(f"/corpora/{cid}/rename", data={"new_name": "After"})
    assert resp.status_code == 200
    assert resp.headers.get("HX-Redirect") == f"/?corpus={cid}"
    assert _corpus_id_by_name(cfg, "After") == cid
    assert _corpus_id_by_name(cfg, "Before") is None


def test_rename_rejects_duplicate_name(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid_a = _seed_one_corpus(cfg, name="Alpha")
    _seed_one_corpus(cfg, name="Beta")
    client = _client(cfg)
    resp = client.post(f"/corpora/{cid_a}/rename", data={"new_name": "Beta"})
    assert resp.status_code == 409
    assert "already exists" in resp.text
    assert _corpus_id_by_name(cfg, "Alpha") == cid_a  # unchanged


def test_rename_rejects_invalid_name(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg, name="Keep")
    client = _client(cfg)
    resp = client.post(f"/corpora/{cid}/rename", data={"new_name": "bad/name"})
    assert resp.status_code == 400
    assert "invalid characters" in resp.text
    assert _corpus_id_by_name(cfg, "Keep") == cid


def test_rename_404_on_unknown_corpus(tmp_path: Path) -> None:
    client = _client(_config(tmp_path))
    assert client.post("/corpora/nope/rename", data={"new_name": "X"}).status_code == 404


def test_rename_to_same_name_is_noop_redirect(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg, name="Same")
    client = _client(cfg)
    resp = client.post(f"/corpora/{cid}/rename", data={"new_name": "Same"})
    assert resp.status_code == 200
    assert resp.headers.get("HX-Redirect") == f"/?corpus={cid}"
    assert _corpus_id_by_name(cfg, "Same") == cid
