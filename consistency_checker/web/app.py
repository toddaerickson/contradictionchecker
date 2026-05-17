"""FastAPI + HTMX web UI for the consistency checker — ADR-0007.

The web layer is a presentation layer over ``pipeline.ingest``,
``pipeline.check``, the assertion store, and the audit logger. No new
business logic — every route delegates to those four surfaces. Templates
under ``templates/`` are prefixed ``cc_`` to avoid filename collisions in
mixed-app deployments; partials use ``cc__<name>.html``.
"""

from __future__ import annotations

import secrets
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
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
    from consistency_checker.check.definition_judge import DefinitionJudge
    from consistency_checker.check.llm_judge import Judge
    from consistency_checker.check.multi_party_judge import MultiPartyJudge
    from consistency_checker.check.nli_checker import NliChecker
    from consistency_checker.extract.atomic_facts import Extractor

_log = get_logger(__name__)

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"


VERDICT_LABELS: dict[str, str] = {
    "confirmed": "Real issue",
    "false_positive": "Not an issue",
    "dismissed": "Dismissed",
}


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
    """``"none"`` when no run exists; otherwise the row's ``run_status``."""
    return "none" if run is None else str(run.run_status)


def _count_total_findings(store: AssertionStore, detector_type: str) -> int:
    """Total findings of the given detector type (across all runs).

    Used by the reviewer-workflow progress count to compute the denominator
    of "X of N reviewed". multi_party_findings lives in its own table.
    """
    if detector_type == "multi_party":
        row = store._conn.execute(
            "SELECT COUNT(*) FROM multi_party_findings "
            "WHERE judge_verdict = 'multi_party_contradiction'"
        ).fetchone()
        return int(row[0])
    row = store._conn.execute(
        "SELECT COUNT(*) FROM findings WHERE detector_type = ?",
        (detector_type,),
    ).fetchone()
    return int(row[0])


def _progress_estimate(run: Any) -> str | None:
    """Human-readable time-remaining estimate; None when not enough data."""
    if not run.started_at or not run.n_pairs_gated or not run.n_pairs_judged:
        return None
    elapsed = (datetime.now() - run.started_at).total_seconds()
    if elapsed <= 0 or run.n_pairs_judged <= 0:
        return None
    rate = run.n_pairs_judged / elapsed
    remaining = (run.n_pairs_gated - run.n_pairs_judged) / rate
    if remaining < 60:
        return f"About {max(1, int(remaining))}s remaining"
    if remaining < 3600:
        return f"About {int(remaining / 60)}m remaining"
    return f"About {int(remaining / 3600)}h remaining"


def _live_counters(run: Any, audit: Any = None) -> dict[str, Any]:
    """Counters in a shape shared by the live and final stats templates."""
    n_multi_party_findings = (
        sum(1 for _ in audit.iter_multi_party_findings(run_id=run.run_id))
        if audit is not None
        else None
    )
    n_definition_pairs_judged = (
        sum(
            1
            for _ in audit.iter_findings(
                run_id=run.run_id, detector_type="definition_inconsistency"
            )
        )
        if audit is not None
        else None
    )
    n_definition_findings = (
        sum(
            1
            for _ in audit.iter_findings(
                run_id=run.run_id,
                verdict="definition_divergent",
                detector_type="definition_inconsistency",
            )
        )
        if audit is not None
        else None
    )
    return {
        "run_id": run.run_id,
        "n_assertions": run.n_assertions,
        "n_pairs_gated": run.n_pairs_gated,
        "n_pairs_judged": run.n_pairs_judged,
        "n_findings": run.n_findings,
        "n_multi_party_findings": n_multi_party_findings,
        "n_definition_pairs_judged": n_definition_pairs_judged,
        "n_definition_findings": n_definition_findings,
        "error_message": run.error_message,
        "progress_estimate": _progress_estimate(run),
        "started_at": run.started_at.isoformat(timespec="seconds") if run.started_at else None,
        "finished_at": run.finished_at.isoformat(timespec="seconds") if run.finished_at else None,
    }


def create_app(
    config: Config,
    *,
    extractor: Extractor | None = None,
    embedder: Embedder | None = None,
    nli_checker: NliChecker | None = None,
    judge: Judge | None = None,
    multi_party_judge: MultiPartyJudge | None = None,
    definition_judge: DefinitionJudge | None = None,
) -> FastAPI:
    """Build the FastAPI app bound to ``config``.

    Tests inject ``extractor`` / ``embedder`` / ``nli_checker`` / ``judge`` /
    ``multi_party_judge`` / ``definition_judge`` to keep ingest and check
    hermetic. In production all default to ``None`` and the app falls back
    to the same factories the CLI uses.
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

    config.data_dir.mkdir(parents=True, exist_ok=True)
    (config.data_dir / "uploads").mkdir(parents=True, exist_ok=True)

    def _make_embedder() -> Embedder:
        if embedder is not None:
            return embedder
        from consistency_checker.pipeline import make_embedder

        return make_embedder(config)

    def _make_extractor() -> Extractor:
        if extractor is not None:
            return extractor
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
    def index(
        request: Request,
        show_reviewed_contradiction: bool = False,
        show_reviewed_multi_party: bool = False,
    ) -> Response:
        """Contradictions tab — the main page (ADR-0007)."""
        store, audit = _open_audit()
        try:
            if store.stats()["documents"] == 0:
                store.close()
                if _is_htmx(request):
                    resp: Response = Response(status_code=200)
                    resp.headers["HX-Redirect"] = "/tabs/ingest"
                    return resp
                return Response(status_code=303, headers={"Location": "/tabs/ingest"})
            run = audit.most_recent_run()
            pair_findings: list[dict[str, Any]] = []
            multi_party_findings: list[dict[str, Any]] = []
            if run is not None:
                raw_pair_findings = [
                    *audit.iter_findings(run_id=run.run_id, verdict="contradiction"),
                    *audit.iter_findings(run_id=run.run_id, verdict="numeric_short_circuit"),
                ]
                assertion_ids = [
                    aid
                    for raw in raw_pair_findings
                    for aid in (raw.assertion_a_id, raw.assertion_b_id)
                ]
                assertions = store.get_assertions_bulk(assertion_ids)
                doc_ids = list({a.doc_id for a in assertions.values()})
                documents = store.get_documents_bulk(doc_ids)

                # Bulk-fetch reviewer verdicts for all pair findings.
                all_pair_keys = [
                    ":".join(sorted([raw.assertion_a_id, raw.assertion_b_id]))
                    for raw in raw_pair_findings
                ]
                pair_reviewer_verdicts = audit.get_reviewer_verdicts_bulk(
                    [(pk, "contradiction") for pk in all_pair_keys]
                )

                for raw in raw_pair_findings:
                    a = assertions.get(raw.assertion_a_id)
                    b = assertions.get(raw.assertion_b_id)
                    if a is None or b is None:
                        continue
                    pair_key = ":".join(sorted([raw.assertion_a_id, raw.assertion_b_id]))
                    rv = pair_reviewer_verdicts.get((pair_key, "contradiction"))
                    if rv is not None and not show_reviewed_contradiction:
                        continue
                    reviewer_verdict = rv.verdict if rv is not None else None
                    pair_findings.append(
                        {
                            "finding_id": raw.finding_id,
                            "verdict": raw.judge_verdict,
                            "confidence": raw.judge_confidence,
                            "doc_a_label": _document_label(documents.get(a.doc_id), a.doc_id),
                            "doc_b_label": _document_label(documents.get(b.doc_id), b.doc_id),
                            "rationale_first_line": (raw.judge_rationale or "").splitlines()[:1],
                            "pair_key": pair_key,
                            "reviewer_verdict": reviewer_verdict,
                            "reviewer_label": VERDICT_LABELS.get(reviewer_verdict)
                            if reviewer_verdict is not None
                            else None,
                        }
                    )
                pair_findings.sort(key=lambda f: -(f["confidence"] or 0.0))

                # Bulk-fetch reviewer verdicts for all multi-party findings.
                raw_mp_findings = list(
                    audit.iter_multi_party_findings(
                        run_id=run.run_id, verdict="multi_party_contradiction"
                    )
                )
                all_mp_keys = [":".join(sorted(mp.assertion_ids)) for mp in raw_mp_findings]
                mp_reviewer_verdicts = audit.get_reviewer_verdicts_bulk(
                    [(pk, "multi_party") for pk in all_mp_keys]
                )

                for mp in raw_mp_findings:
                    mp_key = ":".join(sorted(mp.assertion_ids))
                    rv_mp = mp_reviewer_verdicts.get((mp_key, "multi_party"))
                    if rv_mp is not None and not show_reviewed_multi_party:
                        continue
                    reviewer_verdict_mp = rv_mp.verdict if rv_mp is not None else None
                    multi_party_findings.append(
                        {
                            "finding_id": mp.finding_id,
                            "confidence": mp.judge_confidence,
                            "rationale_first_line": ((mp.judge_rationale or "").splitlines()[:1]),
                            "n_docs": len({d for d in mp.doc_ids}),
                            "pair_key": mp_key,
                            "reviewer_verdict": reviewer_verdict_mp,
                            "reviewer_label": VERDICT_LABELS.get(reviewer_verdict_mp)
                            if reviewer_verdict_mp is not None
                            else None,
                        }
                    )
                multi_party_findings.sort(key=lambda f: -(f["confidence"] or 0.0))

            pair_reviewed = sum(
                audit.count_reviewer_verdicts(detector_type="contradiction").values()
            )
            pair_total = _count_total_findings(store, "contradiction")
            mp_reviewed = sum(audit.count_reviewer_verdicts(detector_type="multi_party").values())
            mp_total = _count_total_findings(store, "multi_party")
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
                "show_reviewed_contradiction": show_reviewed_contradiction,
                "show_reviewed_multi_party": show_reviewed_multi_party,
                "pair_reviewed": pair_reviewed,
                "pair_total": pair_total,
                "mp_reviewed": mp_reviewed,
                "mp_total": mp_total,
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
            page_assertions = list(store.iter_assertions(limit=PAGE_SIZE, offset=offset))
            documents = store.get_documents_bulk([a.doc_id for a in page_assertions])
            rows = [
                {
                    "assertion_id": a.assertion_id,
                    "doc_label": _document_label(documents.get(a.doc_id), a.doc_id),
                    "text_preview": _truncate(a.assertion_text, 100),
                }
                for a in page_assertions
            ]
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

    @app.get("/tabs/definitions", response_class=HTMLResponse)
    def tab_definitions(
        request: Request,
        show_reviewed_definition_inconsistency: bool = False,
    ) -> HTMLResponse:
        """Definition-inconsistencies tab (ADR-0009)."""
        store, audit = _open_audit()
        try:
            run = audit.most_recent_run()
            findings: list[dict[str, Any]] = []
            if run is not None:
                raw = list(
                    audit.iter_findings(
                        run_id=run.run_id,
                        verdict="definition_divergent",
                        detector_type="definition_inconsistency",
                    )
                )
                assertion_ids = [aid for r in raw for aid in (r.assertion_a_id, r.assertion_b_id)]
                assertions = store.get_assertions_bulk(assertion_ids)
                doc_ids = list({a.doc_id for a in assertions.values()})
                documents = store.get_documents_bulk(doc_ids)

                all_pair_keys = [
                    ":".join(sorted([r.assertion_a_id, r.assertion_b_id])) for r in raw
                ]
                reviewer_verdicts = audit.get_reviewer_verdicts_bulk(
                    [(pk, "definition_inconsistency") for pk in all_pair_keys]
                )

                for r in raw:
                    a = assertions.get(r.assertion_a_id)
                    b = assertions.get(r.assertion_b_id)
                    if a is None or b is None:
                        continue
                    pair_key = ":".join(sorted([r.assertion_a_id, r.assertion_b_id]))
                    rv = reviewer_verdicts.get((pair_key, "definition_inconsistency"))
                    if rv is not None and not show_reviewed_definition_inconsistency:
                        continue
                    reviewer_verdict = rv.verdict if rv is not None else None
                    findings.append(
                        {
                            "finding_id": r.finding_id,
                            "term": a.term or "",
                            "confidence": r.judge_confidence,
                            "doc_a_label": _document_label(documents.get(a.doc_id), a.doc_id),
                            "doc_b_label": _document_label(documents.get(b.doc_id), b.doc_id),
                            "def_a_text": a.assertion_text,
                            "def_b_text": b.assertion_text,
                            "rationale": r.judge_rationale or "",
                            "pair_key": pair_key,
                            "reviewer_verdict": reviewer_verdict,
                            "reviewer_label": VERDICT_LABELS.get(reviewer_verdict)
                            if reviewer_verdict is not None
                            else None,
                        }
                    )
                findings.sort(key=lambda f: (f["term"].lower(), -(f["confidence"] or 0.0)))

            reviewed_count = sum(
                audit.count_reviewer_verdicts(detector_type="definition_inconsistency").values()
            )
            total_count = _count_total_findings(store, "definition_inconsistency")
        finally:
            store.close()

        return templates.TemplateResponse(
            request,
            "cc_definitions.html",
            {
                "htmx": _is_htmx(request),
                "active_tab": "definitions",
                "run": {"run_id": run.run_id} if run is not None else None,
                "findings": findings,
                "show_reviewed": show_reviewed_definition_inconsistency,
                "reviewed_count": reviewed_count,
                "total_count": total_count,
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

    def _run_check_in_background(run_id: str, deep: bool) -> None:
        # SQLite handles can't safely be shared across threads, so the
        # background task opens its own store. The embedder is not needed for
        # check (only for ingest), so open faiss directly to avoid the ~800 MB
        # model load that _open_stores() would trigger.
        from consistency_checker.pipeline import check as run_check

        store = AssertionStore(config.db_path)
        store.migrate()
        try:
            faiss_store = FaissStore.open_or_create(
                index_path=config.faiss_path,
                id_map_path=config.faiss_path.with_suffix(".idmap.json"),
            )
            audit_logger = AuditLogger(store)
            try:
                if nli_checker is None:
                    from consistency_checker.check.nli_checker import (
                        TransformerNliChecker,
                    )

                    nli_inst: NliChecker = TransformerNliChecker(model_name=config.nli_model)
                else:
                    nli_inst = nli_checker
                if judge is None:
                    from consistency_checker.pipeline import make_judge

                    judge_inst: Judge = make_judge(config)
                else:
                    judge_inst = judge
                mp_judge = multi_party_judge
                if mp_judge is None and deep:
                    from consistency_checker.pipeline import make_multi_party_judge

                    mp_judge = make_multi_party_judge(config)

                from consistency_checker.check.definition_checker import (
                    DefinitionChecker,
                )

                if definition_judge is not None:
                    def_checker: DefinitionChecker | None = DefinitionChecker(
                        judge=definition_judge
                    )
                else:
                    from consistency_checker.pipeline import make_definition_checker

                    def_checker = make_definition_checker(config)

                run_check(
                    config.model_copy(update={"enable_multi_party": deep}),
                    store=store,
                    faiss_store=faiss_store,
                    nli_checker=nli_inst,
                    judge=judge_inst,
                    audit_logger=audit_logger,
                    multi_party_judge=mp_judge if deep else None,
                    definition_checker=def_checker,
                    run_id=run_id,
                )
            except Exception as exc:
                _log.exception("Run %s failed: %s", run_id, exc)
                audit_logger.update_run_status(run_id, "failed", error_message=str(exc))
        finally:
            store.close()

    @app.post("/runs", response_class=HTMLResponse)
    def post_runs(
        request: Request,
        background_tasks: BackgroundTasks,
        deep: bool = Form(False),
    ) -> Response:
        store, audit = _open_audit()
        try:
            run_id = audit.begin_run(
                config={
                    "deep": deep,
                    "embedder_model": config.embedder_model,
                    "nli_model": config.nli_model,
                    "judge_provider": config.judge_provider,
                    "judge_model": config.judge_model,
                },
                run_status="pending",
            )
        finally:
            store.close()

        background_tasks.add_task(_run_check_in_background, run_id, deep)

        target = f"/tabs/stats?run_id={run_id}"
        if _is_htmx(request):
            response = Response(status_code=202)
            response.headers["HX-Redirect"] = target
            return response
        return Response(
            status_code=303,
            headers={"Location": target},
        )

    @app.get("/tabs/stats", response_class=HTMLResponse)
    def tab_stats(request: Request) -> HTMLResponse:
        store, audit = _open_audit()
        try:
            run = audit.most_recent_run()
            status = _infer_run_status(run)
            counters = _live_counters(run, audit) if run is not None else None
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
        # Returning the final fragment (no polling attrs) is how the
        # self-polling loop stops once the run terminates.
        store, audit = _open_audit()
        try:
            run = audit.get_run(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail=f"run {run_id} not found")
            status = _infer_run_status(run)
            counters = _live_counters(run, audit)
        finally:
            store.close()
        if status == "failed":
            template = "cc__stats_failed.html"
        elif status == "done":
            template = "cc__stats_final.html"
        else:
            template = "cc__stats_live.html"
        return templates.TemplateResponse(
            request,
            template,
            {"status": status, "counters": counters},
        )

    @app.get("/tabs/ingest", response_class=HTMLResponse)
    def tab_ingest(request: Request) -> HTMLResponse:
        store, _audit = _open_audit()
        try:
            conn = store._conn
            rows = conn.execute(
                "SELECT corpus_id, corpus_name FROM corpora ORDER BY created_at DESC"
            ).fetchall()
            corpora = [{"corpus_id": r[0], "corpus_name": r[1]} for r in rows]
        finally:
            store.close()
        return templates.TemplateResponse(
            request,
            "cc_ingest.html",
            {
                "htmx": _is_htmx(request),
                "active_tab": "ingest",
                "corpora": corpora,
            },
        )

    @app.get("/tabs/process", response_class=HTMLResponse)
    def tab_process(request: Request, run_id: str = "") -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "cc_process.html",
            {
                "htmx": _is_htmx(request),
                "active_tab": "process",
                "run_id": run_id,
                "corpus_name": "",
                "judge_provider": "",
            },
        )

    @app.post("/verdicts", response_class=HTMLResponse)
    def post_verdict(
        request: Request,
        pair_key: str = Form(...),
        detector_type: str = Form(...),
        verdict: str = Form(...),
        prior_verdict: str = Form(""),
    ) -> HTMLResponse:
        if detector_type not in {"contradiction", "definition_inconsistency", "multi_party"}:
            raise HTTPException(status_code=400, detail=f"unknown detector_type {detector_type!r}")
        if verdict not in {"confirmed", "false_positive", "dismissed"}:
            raise HTTPException(status_code=400, detail=f"unknown verdict {verdict!r}")

        store, audit = _open_audit()
        try:
            audit.set_reviewer_verdict(
                pair_key=pair_key,
                detector_type=detector_type,  # type: ignore[arg-type]
                verdict=verdict,  # type: ignore[arg-type]
            )
            counts = audit.count_reviewer_verdicts(detector_type=detector_type)  # type: ignore[arg-type]
            reviewed_count = sum(counts.values())
            total_count = _count_total_findings(store, detector_type)
        finally:
            store.close()

        toast = templates.get_template("cc__verdict_toast.html").render(
            verdict_label=VERDICT_LABELS[verdict],
            pair_key=pair_key,
            detector_type=detector_type,
            prior_verdict=prior_verdict,
        )
        progress = templates.get_template("cc__progress_count.html").render(
            detector_type=detector_type,
            reviewed_count=reviewed_count,
            total_count=total_count,
        )
        return HTMLResponse(content=toast + progress)

    @app.post("/verdicts/undo", response_class=HTMLResponse)
    def post_verdict_undo(
        request: Request,
        pair_key: str = Form(...),
        detector_type: str = Form(...),
        prior_verdict: str = Form(""),
    ) -> HTMLResponse:
        if detector_type not in {"contradiction", "definition_inconsistency", "multi_party"}:
            raise HTTPException(status_code=400, detail=f"unknown detector_type {detector_type!r}")
        store, audit = _open_audit()
        try:
            if prior_verdict == "":
                audit.delete_reviewer_verdict(
                    pair_key=pair_key,
                    detector_type=detector_type,  # type: ignore[arg-type]
                )
            else:
                if prior_verdict not in {"confirmed", "false_positive", "dismissed"}:
                    raise HTTPException(
                        status_code=400,
                        detail=f"unknown prior_verdict {prior_verdict!r}",
                    )
                audit.set_reviewer_verdict(
                    pair_key=pair_key,
                    detector_type=detector_type,  # type: ignore[arg-type]
                    verdict=prior_verdict,  # type: ignore[arg-type]
                )
        finally:
            store.close()
        # Caller has hx-target="#cc-tab-content" hx-swap="innerHTML"; we tell HTMX
        # to redirect back to the current tab so it re-fetches fresh content.
        referer = request.headers.get("HX-Current-URL") or request.headers.get("Referer", "/")
        response = HTMLResponse(content="")
        response.headers["HX-Redirect"] = referer
        return response

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
            audit = AuditLogger(store)
            is_first_check = audit.most_recent_run() is None
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
                "is_first_check": is_first_check,
                "error": None,
            },
        )

    # Register API routes
    from consistency_checker.web.api.corpora import (
        findings_router,
    )
    from consistency_checker.web.api.corpora import (
        router as corpora_router,
    )
    from consistency_checker.web.api.runs import (
        corpora_runs_router,
    )
    from consistency_checker.web.api.runs import (
        router as runs_router,
    )

    app.include_router(corpora_router)
    app.include_router(findings_router)
    app.include_router(runs_router)
    app.include_router(corpora_runs_router)

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
