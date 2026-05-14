"""FastAPI + HTMX app for the consistency checker — ADR-0007, step G1.

This module exposes :func:`create_app`, a factory the CLI's ``serve``
subcommand (G5) and tests call to produce a :class:`FastAPI` app bound to a
specific :class:`~consistency_checker.config.Config`. The web layer is a
**presentation layer** — every route delegates to the existing
``pipeline.ingest`` / ``pipeline.check`` / ``AuditLogger`` surface.

Step G1 ships just enough to drop documents in:

- ``GET /`` — placeholder landing page. G2 replaces with the Contradictions tab.
- ``GET /tabs/ingest`` — HTMX partial: upload form + list of files already
  ingested. The tab strip in ``cc_base.html`` swaps the partial into a
  ``#tab-content`` slot via ``hx-get``.
- ``POST /uploads`` — multipart endpoint, writes each uploaded file to
  ``data_dir / "uploads" / <upload_id>`` (timestamp + 8-char hash), runs the
  ingest pipeline synchronously, returns the success-card partial.

The success card includes a disabled "Run / Check now" button as a layout
anchor; G5 enables it.
"""

from __future__ import annotations

import secrets
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from consistency_checker.config import Config
from consistency_checker.corpus.chunker import chunk_document
from consistency_checker.corpus.loader import load_path
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.index.embedder import Embedder, embed_pending
from consistency_checker.index.faiss_store import FaissStore
from consistency_checker.logging_setup import get_logger

if TYPE_CHECKING:
    from consistency_checker.extract.atomic_facts import Extractor

_log = get_logger(__name__)

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"


def _generate_upload_id() -> str:
    """Per-upload directory name: timestamp + short random suffix.

    Sortable on disk; the random suffix prevents collisions when two browsers
    submit in the same wall-clock second.
    """
    return f"{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}_{secrets.token_hex(4)}"


def _is_htmx(request: Request) -> bool:
    """True when the request was issued by an HTMX swap (vs. a direct URL hit)."""
    return request.headers.get("HX-Request", "").lower() == "true"


def create_app(
    config: Config,
    *,
    extractor: Extractor | None = None,
    embedder: Embedder | None = None,
) -> FastAPI:
    """Build the FastAPI app bound to ``config``.

    Tests inject ``extractor`` / ``embedder`` to keep ingest hermetic. In
    production both default to ``None`` and the app falls back to the same
    factories the CLI uses.
    """
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    app = FastAPI(
        title="consistency-checker",
        version="0.3",
        docs_url=None,  # No /docs surface for a localhost tool.
        redoc_url=None,
    )
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.state.config = config
    app.state.extractor_override = extractor
    app.state.embedder_override = embedder

    config.data_dir.mkdir(parents=True, exist_ok=True)
    (config.data_dir / "uploads").mkdir(parents=True, exist_ok=True)

    def _make_embedder() -> Embedder:
        if app.state.embedder_override is not None:
            return app.state.embedder_override  # type: ignore[no-any-return]
        from consistency_checker.pipeline import make_embedder

        return make_embedder(config)

    def _make_extractor() -> Extractor:
        if app.state.extractor_override is not None:
            return app.state.extractor_override  # type: ignore[no-any-return]
        from consistency_checker.pipeline import make_extractor

        return make_extractor(config)

    def _open_stores() -> tuple[AssertionStore, FaissStore, Embedder]:
        store = AssertionStore(config.db_path)
        store.migrate()
        embedder_inst = _make_embedder()
        faiss_store = FaissStore.open_or_create(
            index_path=config.faiss_path,
            id_map_path=config.faiss_path.with_suffix(".idmap.json"),
            dim=embedder_inst.dim,
        )
        return store, faiss_store, embedder_inst

    @app.get("/", response_class=HTMLResponse)
    def index() -> RedirectResponse:
        """Placeholder until G2 replaces with the Contradictions tab."""
        return RedirectResponse(url="/tabs/ingest", status_code=303)

    @app.get("/tabs/ingest", response_class=HTMLResponse)
    def tab_ingest(request: Request) -> HTMLResponse:
        uploads_root = config.data_dir / "uploads"
        prior_uploads = (
            sorted(
                (p for p in uploads_root.iterdir() if p.is_dir()),
                reverse=True,
            )
            if uploads_root.exists()
            else []
        )
        return templates.TemplateResponse(
            request,
            "cc_ingest.html",
            {
                "htmx": _is_htmx(request),
                "active_tab": "ingest",
                "prior_uploads": [
                    {
                        "name": p.name,
                        "files": sorted(child.name for child in p.iterdir() if child.is_file()),
                    }
                    for p in prior_uploads
                ],
            },
        )

    @app.post("/uploads", response_class=HTMLResponse)
    async def post_uploads(
        request: Request,
        files: list[UploadFile],
    ) -> HTMLResponse:
        upload_id = _generate_upload_id()
        upload_dir = config.data_dir / "uploads" / upload_id
        upload_dir.mkdir(parents=True, exist_ok=False)

        saved: list[Path] = []
        for file in files:
            if not file.filename:
                continue
            target = upload_dir / Path(file.filename).name
            target.write_bytes(await file.read())
            saved.append(target)

        if not saved:
            return templates.TemplateResponse(
                request,
                "cc__upload_success.html",
                {"upload_id": upload_id, "saved": [], "n_assertions": 0, "error": "no files"},
                status_code=400,
            )

        store, faiss_store, embedder_inst = _open_stores()
        try:
            extractor = _make_extractor()
            n_assertions = _ingest_uploaded_paths(
                saved,
                store=store,
                faiss_store=faiss_store,
                embedder=embedder_inst,
                extractor=extractor,
                config=config,
            )
        finally:
            store.close()

        _log.info(
            "Upload %s: %d files saved, %d assertions extracted",
            upload_id,
            len(saved),
            n_assertions,
        )
        return templates.TemplateResponse(
            request,
            "cc__upload_success.html",
            {
                "upload_id": upload_id,
                "saved": [p.name for p in saved],
                "n_assertions": n_assertions,
                "error": None,
            },
        )

    return app


def _ingest_uploaded_paths(
    paths: list[Path],
    *,
    store: AssertionStore,
    faiss_store: FaissStore,
    embedder: Embedder,
    extractor: Extractor,
    config: Config,
) -> int:
    """Run loader → chunker → extractor → embedder on a list of file paths.

    Mirrors :func:`pipeline.ingest` but takes explicit paths instead of walking
    a directory — the web upload doesn't have a single corpus_dir, just a
    handful of just-saved files. Returns the number of assertions added.
    """
    n_assertions = 0
    for path in paths:
        loaded = load_path(path)
        store.add_document(loaded.document)
        for chunk in chunk_document(
            loaded,
            max_chars=config.chunk_max_chars,
            overlap_chars=config.chunk_overlap_chars,
        ):
            assertions = extractor.extract(chunk)
            if assertions:
                store.add_assertions(assertions)
                n_assertions += len(assertions)
    embed_pending(store, faiss_store, embedder)
    return n_assertions
