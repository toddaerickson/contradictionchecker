"""Hermetic tests for the web upload/ingest path.

ADR-0017 Phase 6 deleted the legacy ``GET /tabs/ingest`` tab and the
``POST /uploads`` route. Corpus creation + ingest now flows exclusively
through ``POST /corpora/new`` (the New Corpus modal). The per-request DoS
caps live on that route, so their coverage stays here. Modal-shape and
success-path coverage lives in ``test_web_ui_collapse.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from consistency_checker.config import Config
from consistency_checker.extract.atomic_facts import FixtureExtractor
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.web.app import create_app
from tests.conftest import HashEmbedder


def _config(tmp_path: Path) -> Config:
    return Config(
        corpus_dir=tmp_path / "corpus",  # unused by web layer
        judge_provider="fixture",
        judge_model="test",
        data_dir=tmp_path / "store",
        log_dir=tmp_path / "logs",
        embedder_model="hash",
        nli_model="fixture",
    )


def _client(cfg: Config) -> TestClient:
    app = create_app(
        cfg,
        extractor=FixtureExtractor({}),
        embedder=HashEmbedder(dim=64),
    )
    return TestClient(app)


# --- per-request DoS caps on POST /corpora/new (Task D) ----------------------


def _uploads_root(cfg: Config) -> Path:
    return cfg.data_dir / "uploads"


def _staged_dirs(cfg: Config) -> list[Path]:
    root = _uploads_root(cfg)
    return list(root.iterdir()) if root.exists() else []


def _corpus_names(cfg: Config) -> list[str]:
    store = AssertionStore(cfg.db_path)
    store.migrate()
    try:
        return sorted(c.corpus_name for c in store.list_corpora())
    finally:
        store.close()


def test_create_corpus_rejects_too_many_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    app = create_app(cfg, extractor=FixtureExtractor({}), embedder=HashEmbedder(dim=64))
    client = TestClient(app)
    monkeypatch.setattr("consistency_checker.web.app.MAX_UPLOAD_FILES", 2)
    files = [("files", (f"d{i}.txt", b"body.", "text/plain")) for i in range(3)]
    data = {"corpus_name": "too-many", "judge_provider": "moonshot"}
    resp = client.post("/corpora/new", files=files, data=data)
    assert resp.status_code == 413, resp.text
    assert "many files" in resp.text.lower()
    assert _staged_dirs(cfg) == []
    assert _corpus_names(cfg) == []


def test_create_corpus_rejects_total_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config(tmp_path)
    app = create_app(cfg, extractor=FixtureExtractor({}), embedder=HashEmbedder(dim=64))
    client = TestClient(app)
    monkeypatch.setattr("consistency_checker.web.app.MAX_UPLOAD_TOTAL_BYTES", 10)
    files = [
        ("files", ("a.txt", b"x" * 8, "text/plain")),
        ("files", ("b.txt", b"y" * 8, "text/plain")),
    ]
    data = {"corpus_name": "too-big", "judge_provider": "moonshot"}
    resp = client.post("/corpora/new", files=files, data=data)
    assert resp.status_code == 413, resp.text
    assert "too large" in resp.text.lower()
    assert _staged_dirs(cfg) == []
    assert _corpus_names(cfg) == []


def test_create_corpus_rolls_back_on_rejection_so_name_is_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cap rejection must not leave an orphan corpus row that 409s every retry."""
    cfg = _config(tmp_path)
    app = create_app(cfg, extractor=FixtureExtractor({}), embedder=HashEmbedder(dim=64))
    client = TestClient(app)
    monkeypatch.setattr("consistency_checker.web.app.MAX_UPLOAD_FILES", 2)
    data = {"corpus_name": "retry-me", "judge_provider": "moonshot"}
    rejected = client.post(
        "/corpora/new",
        files=[("files", (f"d{i}.txt", b"body.", "text/plain")) for i in range(3)],
        data=data,
    )
    assert rejected.status_code == 413, rejected.text
    assert _corpus_names(cfg) == []

    retry = client.post(
        "/corpora/new",
        files=[("files", ("ok.txt", b"body.", "text/plain"))],
        data=data,
    )
    assert retry.status_code == 200, retry.text
    assert _corpus_names(cfg) == ["retry-me"]


# --- static files -----------------------------------------------------------


def test_static_assets_served(tmp_path: Path) -> None:
    client = _client(_config(tmp_path))
    css = client.get("/static/cc_style.css")
    assert css.status_code == 200
    js = client.get("/static/htmx.min.js")
    assert js.status_code == 200


# --- _require_filename: explicit guard replacing a -O-stripped assert (audit #5) ---


def test_require_filename_returns_truthy_name() -> None:
    from types import SimpleNamespace

    from consistency_checker.web.app import _require_filename

    assert _require_filename(SimpleNamespace(filename="report.pdf")) == "report.pdf"  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_name", [None, ""])
def test_require_filename_rejects_missing_name(bad_name: str | None) -> None:
    from types import SimpleNamespace

    from fastapi import HTTPException

    from consistency_checker.web.app import _require_filename

    with pytest.raises(HTTPException) as exc_info:
        _require_filename(SimpleNamespace(filename=bad_name))  # type: ignore[arg-type]
    assert exc_info.value.status_code == 400
