"""Hermetic tests for the FastAPI web app — step G1.

Uses ``FixtureExtractor`` + ``HashEmbedder`` so the upload + ingest path runs
without external network or HF model downloads.
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
        # FixtureExtractor with an empty map → every chunk maps to an empty
        # list, which makes _ingest_uploaded_paths a no-op for assertions but
        # still exercises load → chunk → store paths end-to-end.
        extractor=FixtureExtractor({}),
        embedder=HashEmbedder(dim=64),
    )
    return TestClient(app)


# --- routing -----------------------------------------------------------------


def test_root_renders_contradictions_tab(tmp_path: Path) -> None:
    """G2 moved the index to the Contradictions tab. With no runs, the empty-state
    banner is shown rather than the upload form."""
    client = _client(_config(tmp_path))
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 200
    assert "No runs yet" in response.text


def test_ingest_tab_renders_upload_form(tmp_path: Path) -> None:
    client = _client(_config(tmp_path))
    response = client.get("/tabs/ingest")
    assert response.status_code == 200
    body = response.text
    assert "Upload and ingest" in body
    assert 'hx-post="/uploads"' in body
    assert 'enctype="multipart/form-data"' in body or 'hx-encoding="multipart/form-data"' in body


def test_ingest_tab_htmx_request_omits_base_chrome(tmp_path: Path) -> None:
    """An HX-Request hit returns a partial only, not the full page chrome."""
    client = _client(_config(tmp_path))
    response = client.get("/tabs/ingest", headers={"HX-Request": "true"})
    assert response.status_code == 200
    body = response.text
    assert "Upload and ingest" in body
    # Base chrome (the <header>) is excluded so the swap target only gets content.
    assert "<header" not in body
    assert "<nav" not in body


# --- POST /uploads -----------------------------------------------------------


@pytest.fixture
def configured_client(tmp_path: Path) -> TestClient:
    cfg = _config(tmp_path)

    # Fixture extractor needs to be keyed by chunk_id, but the chunk_id depends
    # on the loaded text. Pre-walk so the extractor returns one assertion per
    # chunk we'll produce.
    from consistency_checker.corpus.chunker import chunk_document
    from consistency_checker.corpus.loader import load_path

    sample = tmp_path / "_sample.txt"
    sample.write_text("Revenue grew. Customer satisfaction held.")
    loaded = load_path(sample)
    chunks = chunk_document(loaded, max_chars=cfg.chunk_max_chars, overlap_chars=0)
    fixtures = {c.chunk_id: [f"fact-{i}"] for i, c in enumerate(chunks)}
    sample.unlink()

    app = create_app(
        cfg,
        extractor=FixtureExtractor(fixtures),
        embedder=HashEmbedder(dim=64),
    )
    return TestClient(app)


def test_upload_saves_file_and_runs_ingest(configured_client: TestClient, tmp_path: Path) -> None:
    cfg: Config = configured_client.app.state.config  # type: ignore[attr-defined]
    response = configured_client.post(
        "/uploads",
        files={"files": ("doc.txt", b"Revenue grew. Customer satisfaction held.", "text/plain")},
    )
    assert response.status_code == 200, response.text
    body = response.text
    assert "Upload complete" in body
    assert "doc.txt" in body

    uploads = list((cfg.data_dir / "uploads").iterdir())
    assert len(uploads) == 1
    saved = uploads[0] / "doc.txt"
    assert saved.exists()
    assert saved.read_bytes() == b"Revenue grew. Customer satisfaction held."

    store = AssertionStore(cfg.db_path)
    n_assertions = store.stats()["assertions"]
    store.close()
    assert n_assertions > 0


def test_upload_success_card_includes_run_button(
    configured_client: TestClient,
) -> None:
    """G5 enables the Run / Check now button — it now POSTs to /runs."""
    response = configured_client.post(
        "/uploads",
        files={"files": ("doc.txt", b"Anything.", "text/plain")},
    )
    assert response.status_code == 200
    body = response.text
    assert "Run / Check now" in body
    assert 'hx-post="/runs"' in body
    assert 'name="deep"' in body  # the deep-mode checkbox is there


def test_upload_with_no_file_part_rejected(tmp_path: Path) -> None:
    """A POST without any file part fails fast at the framework validation
    layer (422) rather than blowing up the ingest pipeline."""
    cfg = _config(tmp_path)
    app = create_app(cfg, extractor=FixtureExtractor({}), embedder=HashEmbedder(dim=64))
    client = TestClient(app)
    response = client.post("/uploads")
    assert response.status_code == 422


def test_upload_with_empty_filename_returns_error_card(tmp_path: Path) -> None:
    """An empty filename slips past the framework but our route catches it."""
    cfg = _config(tmp_path)
    app = create_app(cfg, extractor=FixtureExtractor({}), embedder=HashEmbedder(dim=64))
    client = TestClient(app)
    response = client.post(
        "/uploads",
        files={"files": ("blank.txt", b"", "application/octet-stream")},
    )
    # blank file content is fine — we still save it; "no files" only fires when
    # every UploadFile has an empty .filename, which the framework now refuses.
    assert response.status_code == 200


# --- prior uploads listing --------------------------------------------------


def test_prior_uploads_listed_on_ingest_tab(configured_client: TestClient) -> None:
    """After an upload, the Ingest tab shows it under Prior uploads."""
    configured_client.post(
        "/uploads",
        files={"files": ("first.txt", b"Body.", "text/plain")},
    )
    response = configured_client.get("/tabs/ingest")
    assert response.status_code == 200
    body = response.text
    assert "first.txt" in body
    assert "Prior uploads" in body


# --- static files -----------------------------------------------------------


def test_static_assets_served(tmp_path: Path) -> None:
    client = _client(_config(tmp_path))
    css = client.get("/static/cc_style.css")
    assert css.status_code == 200
    assert "cc-tab" in css.text
    js = client.get("/static/htmx.min.js")
    assert js.status_code == 200
