"""FastAPI + HTMX app for the consistency checker — ADR-0007, steps G1+G2.

This module exposes :func:`create_app`, a factory the CLI's ``serve``
subcommand (G5) and tests call to produce a :class:`FastAPI` app bound to a
specific :class:`~consistency_checker.config.Config`. The web layer is a
**presentation layer** — every route delegates to the existing
``pipeline.ingest`` / ``pipeline.check`` / ``AuditLogger`` surface.

Routes shipped through G2:

- ``GET /`` — **Contradictions tab**. Pulls findings for the most recent run
  via :meth:`AuditLogger.most_recent_run`. Renders pair findings (verdict in
  ``{contradiction, numeric_short_circuit}``) and multi-party findings
  (verdict ``multi_party_contradiction``). Empty-state banner with a
  disabled "Check now" button when no run exists.
- ``GET /findings/{finding_id}/diff`` — HTMX partial: side-by-side card view
  of the two assertions that triggered a pair finding.
- ``GET /multi_party_findings/{finding_id}/diff`` — HTMX partial: 3-pane
  side-by-side view of the assertions in a triangle finding.
- ``GET /tabs/ingest`` — Upload form + list of prior uploads.
- ``POST /uploads`` — multipart endpoint, writes files, runs ingest.

The "Run / Check now" buttons on the Ingest success card and Contradictions
empty-state banner both target ``POST /runs``, which G5 wires up.
"""

from __future__ import annotations

import secrets
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from consistency_checker.audit.logger import AuditLogger
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


def _document_label(doc: Any, fallback_doc_id: str) -> str:
    """Best display label for a Document — title, then source path, then doc id."""
    if doc is None:
        return fallback_doc_id
    return doc.title or doc.source_path or fallback_doc_id


def _truncate(text: str, n: int) -> str:
    """Trim ``text`` to at most ``n`` chars with a trailing ellipsis."""
    return text if len(text) <= n else text[: max(0, n - 1)] + "…"


def _infer_run_status(run: Any) -> str:
    """Return ``running`` / ``done`` from the existing pipeline_runs columns.

    G5 will replace this inference with a real ``run_status`` column written
    by ``pipeline.check``. Until then, ``finished_at IS NULL`` means a run is
    still in flight (the audit logger sets ``finished_at`` in ``end_run``).
    """
    if run is None:
        return "none"
    return "running" if run.finished_at is None else "done"


def _live_counters(run: Any) -> dict[str, Any]:
    """Pull the counter columns that survive the live → final transition.

    Mid-run these stay at their initial 0 (G5 will fill them in as the
    pipeline progresses); post-run they hold the final tallies. Same shape
    in both phases so the live and final templates can use the same field
    names.
    """
    return {
        "run_id": run.run_id,
        "n_assertions": run.n_assertions,
        "n_pairs_gated": run.n_pairs_gated,
        "n_pairs_judged": run.n_pairs_judged,
        "n_findings": run.n_findings,
        "started_at": run.started_at.isoformat(timespec="seconds") if run.started_at else None,
        "finished_at": run.finished_at.isoformat(timespec="seconds") if run.finished_at else None,
    }


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

    def _open_audit() -> tuple[AssertionStore, AuditLogger]:
        store = AssertionStore(config.db_path)
        store.migrate()
        return store, AuditLogger(store)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        """Contradictions tab — the main page (ADR-0007)."""
        store, audit = _open_audit()
        try:
            run = audit.most_recent_run()
            pair_findings: list[dict[str, Any]] = []
            multi_party_findings: list[dict[str, Any]] = []
            if run is not None:
                for raw in (
                    *audit.iter_findings(run_id=run.run_id, verdict="contradiction"),
                    *audit.iter_findings(run_id=run.run_id, verdict="numeric_short_circuit"),
                ):
                    a = store.get_assertion(raw.assertion_a_id)
                    b = store.get_assertion(raw.assertion_b_id)
                    if a is None or b is None:
                        continue
                    doc_a = store.get_document(a.doc_id)
                    doc_b = store.get_document(b.doc_id)
                    pair_findings.append(
                        {
                            "finding_id": raw.finding_id,
                            "verdict": raw.judge_verdict,
                            "confidence": raw.judge_confidence,
                            "doc_a_label": _document_label(doc_a, a.doc_id),
                            "doc_b_label": _document_label(doc_b, b.doc_id),
                            "rationale_first_line": (raw.judge_rationale or "").splitlines()[:1],
                        }
                    )
                pair_findings.sort(key=lambda f: -(f["confidence"] or 0.0))

                for mp in audit.iter_multi_party_findings(
                    run_id=run.run_id, verdict="multi_party_contradiction"
                ):
                    multi_party_findings.append(
                        {
                            "finding_id": mp.finding_id,
                            "confidence": mp.judge_confidence,
                            "rationale_first_line": ((mp.judge_rationale or "").splitlines()[:1]),
                            "n_docs": len({d for d in mp.doc_ids}),
                        }
                    )
                multi_party_findings.sort(key=lambda f: -(f["confidence"] or 0.0))
        finally:
            store.close()

        return templates.TemplateResponse(
            request,
            "cc_contradictions.html",
            {
                "htmx": _is_htmx(request),
                "active_tab": "contradictions",
                "run": {
                    "run_id": run.run_id,
                    "n_assertions": run.n_assertions,
                    "n_pairs_judged": run.n_pairs_judged,
                }
                if run is not None
                else None,
                "pair_findings": pair_findings,
                "multi_party_findings": multi_party_findings,
            },
        )

    @app.get("/findings/{finding_id}/diff", response_class=HTMLResponse)
    def pair_diff(request: Request, finding_id: str) -> HTMLResponse:
        store, audit = _open_audit()
        try:
            finding = audit.get_finding(finding_id)
            if finding is None:
                raise HTTPException(status_code=404, detail=f"finding {finding_id} not found")
            a = store.get_assertion(finding.assertion_a_id)
            b = store.get_assertion(finding.assertion_b_id)
            if a is None or b is None:
                raise HTTPException(
                    status_code=404, detail="assertion referenced by finding not found"
                )
            doc_a = store.get_document(a.doc_id)
            doc_b = store.get_document(b.doc_id)
            context = {
                "finding": {
                    "finding_id": finding.finding_id,
                    "verdict": finding.judge_verdict,
                    "confidence": finding.judge_confidence,
                    "rationale": finding.judge_rationale,
                    "evidence_spans": finding.evidence_spans,
                    "nli_p_contradiction": finding.nli_p_contradiction,
                    "gate_score": finding.gate_score,
                },
                "side_a": {
                    "assertion_text": a.assertion_text,
                    "doc_label": _document_label(doc_a, a.doc_id),
                },
                "side_b": {
                    "assertion_text": b.assertion_text,
                    "doc_label": _document_label(doc_b, b.doc_id),
                },
            }
        finally:
            store.close()
        return templates.TemplateResponse(request, "cc__pair_diff.html", context)

    @app.get("/multi_party_findings/{finding_id}/diff", response_class=HTMLResponse)
    def multi_party_diff(request: Request, finding_id: str) -> HTMLResponse:
        store, audit = _open_audit()
        try:
            mp = audit.get_multi_party_finding(finding_id)
            if mp is None:
                raise HTTPException(
                    status_code=404, detail=f"multi-party finding {finding_id} not found"
                )
            sides: list[dict[str, Any]] = []
            for label, aid in zip(("A", "B", "C"), mp.assertion_ids[:3], strict=False):
                assertion = store.get_assertion(aid)
                if assertion is None:
                    sides.append({"label": label, "assertion_text": "(missing)", "doc_label": aid})
                    continue
                doc = store.get_document(assertion.doc_id)
                sides.append(
                    {
                        "label": label,
                        "assertion_text": assertion.assertion_text,
                        "doc_label": _document_label(doc, assertion.doc_id),
                    }
                )
            min_edge = (
                min(score for _, _, score in mp.triangle_edge_scores)
                if mp.triangle_edge_scores
                else None
            )
            context = {
                "finding": {
                    "finding_id": mp.finding_id,
                    "verdict": mp.judge_verdict,
                    "confidence": mp.judge_confidence,
                    "rationale": mp.judge_rationale,
                    "evidence_spans": mp.evidence_spans,
                    "min_edge_score": min_edge,
                },
                "sides": sides,
            }
        finally:
            store.close()
        return templates.TemplateResponse(request, "cc__multi_party_diff.html", context)

    PAGE_SIZE = 25

    def _pagination(*, page: int, total: int) -> dict[str, Any]:
        """Compute prev/next page numbers for a paginated table."""
        page = max(1, page)
        n_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        page = min(page, n_pages)
        return {
            "page": page,
            "n_pages": n_pages,
            "has_prev": page > 1,
            "has_next": page < n_pages,
            "prev": page - 1,
            "next": page + 1,
            "total": total,
            "page_size": PAGE_SIZE,
        }

    @app.get("/tabs/documents", response_class=HTMLResponse)
    def tab_documents(request: Request, page: int = 1) -> HTMLResponse:
        store, _audit = _open_audit()
        try:
            total = store.stats()["documents"]
            pag = _pagination(page=page, total=total)
            offset = (pag["page"] - 1) * PAGE_SIZE
            rows = [
                {
                    "doc_id": d.doc_id,
                    "title": d.title or d.source_path,
                    "source_path": d.source_path,
                    "ingested_at": d.ingested_at.isoformat(timespec="seconds")
                    if d.ingested_at
                    else "",
                    "doc_type": d.doc_type or "",
                }
                for d in store.iter_documents(limit=PAGE_SIZE, offset=offset)
            ]
        finally:
            store.close()
        return templates.TemplateResponse(
            request,
            "cc_documents.html",
            {
                "htmx": _is_htmx(request),
                "active_tab": "documents",
                "documents": rows,
                "pagination": pag,
            },
        )

    @app.get("/documents/{doc_id}", response_class=HTMLResponse)
    def document_detail(request: Request, doc_id: str) -> HTMLResponse:
        store, _audit = _open_audit()
        try:
            doc = store.get_document(doc_id)
            if doc is None:
                raise HTTPException(status_code=404, detail=f"document {doc_id} not found")
            n_assertions = sum(1 for _ in store.iter_assertions(doc_id=doc_id))
            preview_assertions = [
                a.assertion_text for a in store.iter_assertions(doc_id=doc_id, limit=5)
            ]
            context = {
                "doc": {
                    "doc_id": doc.doc_id,
                    "title": doc.title or doc.source_path,
                    "source_path": doc.source_path,
                    "doc_type": doc.doc_type or "",
                    "doc_date": doc.doc_date or "",
                    "ingested_at": doc.ingested_at.isoformat(timespec="seconds")
                    if doc.ingested_at
                    else "",
                    "metadata_json": doc.metadata_json,
                },
                "n_assertions": n_assertions,
                "preview_assertions": preview_assertions,
            }
        finally:
            store.close()
        return templates.TemplateResponse(request, "cc__document_detail.html", context)

    @app.get("/tabs/assertions", response_class=HTMLResponse)
    def tab_assertions(request: Request, page: int = 1) -> HTMLResponse:
        store, _audit = _open_audit()
        try:
            total = store.stats()["assertions"]
            pag = _pagination(page=page, total=total)
            offset = (pag["page"] - 1) * PAGE_SIZE
            rows: list[dict[str, Any]] = []
            for a in store.iter_assertions(limit=PAGE_SIZE, offset=offset):
                doc = store.get_document(a.doc_id)
                rows.append(
                    {
                        "assertion_id": a.assertion_id,
                        "doc_label": _document_label(doc, a.doc_id),
                        "text_preview": _truncate(a.assertion_text, 100),
                    }
                )
        finally:
            store.close()
        return templates.TemplateResponse(
            request,
            "cc_assertions.html",
            {
                "htmx": _is_htmx(request),
                "active_tab": "assertions",
                "assertions": rows,
                "pagination": pag,
            },
        )

    @app.get("/assertions/{assertion_id}", response_class=HTMLResponse)
    def assertion_detail(request: Request, assertion_id: str) -> HTMLResponse:
        store, _audit = _open_audit()
        try:
            a = store.get_assertion(assertion_id)
            if a is None:
                raise HTTPException(status_code=404, detail=f"assertion {assertion_id} not found")
            doc = store.get_document(a.doc_id)
            context = {
                "assertion": {
                    "assertion_id": a.assertion_id,
                    "assertion_text": a.assertion_text,
                    "doc_id": a.doc_id,
                    "doc_label": _document_label(doc, a.doc_id),
                    "chunk_id": a.chunk_id,
                    "char_start": a.char_start,
                    "char_end": a.char_end,
                    "char_span": (
                        f"{a.char_start}:{a.char_end}"
                        if a.char_start is not None and a.char_end is not None
                        else None
                    ),
                },
            }
        finally:
            store.close()
        return templates.TemplateResponse(request, "cc__assertion_detail.html", context)

    @app.get("/tabs/stats", response_class=HTMLResponse)
    def tab_stats(request: Request) -> HTMLResponse:
        store, audit = _open_audit()
        try:
            run = audit.most_recent_run()
            status = _infer_run_status(run)
            counters = _live_counters(run) if run is not None else None
        finally:
            store.close()
        return templates.TemplateResponse(
            request,
            "cc_stats.html",
            {
                "htmx": _is_htmx(request),
                "active_tab": "stats",
                "status": status,
                "counters": counters,
            },
        )

    @app.get("/runs/{run_id}/stats", response_class=HTMLResponse)
    def run_stats_fragment(request: Request, run_id: str) -> HTMLResponse:
        """Polled by the Stats tab every 2 s while a run is in flight.

        Returns the live fragment (with self-polling attributes) while
        ``status == "running"``; once ``status == "done"``, returns the
        final-summary fragment which has no polling attributes — so the
        next HTMX swap stops the poll.
        """
        store, audit = _open_audit()
        try:
            run = audit.get_run(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail=f"run {run_id} not found")
            status = _infer_run_status(run)
            counters = _live_counters(run)
        finally:
            store.close()
        template = "cc__stats_final.html" if status == "done" else "cc__stats_live.html"
        return templates.TemplateResponse(
            request,
            template,
            {"status": status, "counters": counters},
        )

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
