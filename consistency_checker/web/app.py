"""FastAPI + HTMX web UI for the consistency checker — ADR-0007.

The web layer is a presentation layer over ``pipeline.ingest``,
``pipeline.check``, the assertion store, and the audit logger. No new
business logic — every route delegates to those four surfaces. Templates
under ``templates/`` are prefixed ``cc_`` to avoid filename collisions in
mixed-app deployments; partials use ``cc__<name>.html``.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import math
import re
import secrets
import shutil
import sqlite3
import uuid
from collections.abc import AsyncGenerator
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.cli.warnings import (
    render_corpus_warning,
    render_fragmentation_warning,
    render_identification_failure_notice,
    summarize_buckets,
)
from consistency_checker.config import Config, load_local_env
from consistency_checker.corpus.chunker import chunk_document
from consistency_checker.corpus.junk_filter import JunkAudit
from consistency_checker.corpus.loader import load_path
from consistency_checker.corpus.ocr import OcrAudit
from consistency_checker.index.assertion_store import AssertionStore, CrossCorpusDocumentError
from consistency_checker.index.embedder import Embedder, embed_pending
from consistency_checker.index.faiss_store import FaissStore
from consistency_checker.logging_setup import get_logger
from consistency_checker.pipeline import default_per_call_costs

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

# Raw exception text can carry absolute paths and provider/SDK error strings,
# and the run's error_message is rendered verbatim in the failed-stats UI; keep
# the stored value generic and route the detail to the logs instead.
_GENERIC_FAILURE_MESSAGE = "The check failed. See server logs for details."


VERDICT_LABELS: dict[str, str] = {
    "confirmed": "Real issue",
    "false_positive": "Not an issue",
    "dismissed": "Dismissed",
}

# ADR-0017 Phase 3: SSE progress polling cadence + cap. The cap is generous (1h)
# so a long check doesn't kill the stream; tests monkeypatch these to make the
# generator exit in deterministic, sub-second time.
PROGRESS_POLL_SECONDS: float = 1.0
PROGRESS_MAX_ITERATIONS: int = 3600
# After a run transitions to done/failed (or when no run exists), keep streaming
# for a short tail so the row settles cleanly in the sidebar before the SSE
# extension closes. Tests set this to 0.
PROGRESS_DONE_TAIL_SECONDS: float = 5.0


def _generate_upload_id() -> str:
    """Per-upload directory name: timestamp + short random suffix.

    Sortable on disk; the random suffix prevents collisions when two browsers
    submit in the same wall-clock second.
    """
    return f"{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}_{secrets.token_hex(4)}"


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


def _count_total_findings(
    store: AssertionStore, detector_type: str, *, corpus_id: str | None = None
) -> int:
    """Total findings of the given detector type.

    When ``corpus_id`` is None, counts across all corpora (used by the
    verdict-toast global count). When set, scopes to runs on that corpus
    so the per-corpus drawer counter denominator matches the findings
    actually shown. multi_party_findings lives in its own table.
    """
    if detector_type == "multi_party":
        if corpus_id is None:
            row = store._conn.execute(
                "SELECT COUNT(*) FROM multi_party_findings "
                "WHERE judge_verdict = 'multi_party_contradiction'"
            ).fetchone()
        else:
            row = store._conn.execute(
                "SELECT COUNT(*) FROM multi_party_findings mpf "
                "JOIN pipeline_runs pr ON mpf.run_id = pr.run_id "
                "WHERE mpf.judge_verdict = 'multi_party_contradiction' "
                "AND pr.corpus_id = ?",
                (corpus_id,),
            ).fetchone()
        return int(row[0])
    if corpus_id is None:
        row = store._conn.execute(
            "SELECT COUNT(*) FROM findings WHERE detector_type = ?",
            (detector_type,),
        ).fetchone()
    else:
        row = store._conn.execute(
            "SELECT COUNT(*) FROM findings f "
            "JOIN pipeline_runs pr ON f.run_id = pr.run_id "
            "WHERE f.detector_type = ? AND pr.corpus_id = ?",
            (detector_type, corpus_id),
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
        "run_corpus_id": run.corpus_id,
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


def _corpus_banner_context(
    store: AssertionStore,
    config: Config,
    *,
    run_id: str | None,
    corpus_id: str | None = None,
) -> dict[str, Any]:
    """Compute the four corpus-composition warning fields for the Stats banner.

    Mirrors the CLI warnings (see ``cli/warnings.py``) so the web surface emits
    the same triggers/messages as ``consistency-check check``.

    When ``corpus_id`` is provided the banner is scoped to that corpus only so
    that multi-org warnings from other corpora do not bleed into the view.
    """
    docs = list(store.iter_documents(corpus_id=corpus_id))
    rows: list[tuple[str, str | None]] = [(d.doc_id, d.org_label) for d in docs]
    summary = summarize_buckets(rows)
    corpus_warning = render_corpus_warning(
        summary.known,
        summary.unknown_count,
        scope_enabled=config.org_scope_enabled,
    )
    fragmentation_warning = render_fragmentation_warning(summary.known)
    failures = sum(1 for d in docs if d.org_reason in {"llm_error", "truncated"})
    identification_failure_notice = render_identification_failure_notice(
        failures=failures, total=len(rows)
    )
    n_suppressed = 0
    if run_id is not None:
        row = store._conn.execute(
            "SELECT COUNT(*) FROM findings WHERE suppressed = 1 AND run_id = ?",
            (run_id,),
        ).fetchone()
        n_suppressed = row[0] if row else 0
    return {
        "corpus_warning": corpus_warning,
        "fragmentation_warning": fragmentation_warning,
        "identification_failure_notice": identification_failure_notice,
        "n_definition_pairs_suppressed": n_suppressed,
    }


MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB per file
MAX_UPLOAD_FILES = 100  # files per request
MAX_UPLOAD_TOTAL_BYTES = 500 * 1024 * 1024  # 500 MB per request (aggregate)
_ALLOWED_EXTENSIONS = frozenset({".txt", ".md", ".pdf", ".docx"})

# ADR-0017 Phase 6: the legacy 7-tab UI is gone. Its entry points
# (``GET /?legacy=1`` and every ``GET /tabs/*`` route) now return 410 Gone
# pointing at the single-page shell that replaced them.
_LEGACY_GONE_BODY = "This UI was replaced. Visit / for the new interface."

# pair_key is N segments of 16-hex joined by ':' (build_pair_key over 2+ ids).
_PAIR_KEY_RE = re.compile(r"[0-9a-f]{16}(?::[0-9a-f]{16})+")


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
    load_local_env()
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

    # --- ADR-0017 single-page shell helpers ------------------------------
    #
    # The single-page shell (sidebar + findings list + New Corpus, Run
    # Check, drawer, and verdict buttons + cost gauge) is the only UI.

    _PAIR_FINDING_VERDICTS = ("contradiction", "numeric_short_circuit")
    _DEFINITION_VERDICTS = ("definition_divergent",)

    def _list_corpora_with_counts(store: AssertionStore) -> list[dict[str, Any]]:
        """Sidebar rows: corpora + per-corpus (n_assertions, n_runs)."""
        rows: list[dict[str, Any]] = []
        for c in store.list_corpora():
            n_assertions = store.stats(corpus_id=c.corpus_id)["assertions"]
            n_runs_row = store._conn.execute(
                "SELECT COUNT(*) FROM pipeline_runs WHERE corpus_id = ?",
                (c.corpus_id,),
            ).fetchone()
            n_runs = int(n_runs_row[0]) if n_runs_row else 0
            rows.append(
                {
                    "corpus_id": c.corpus_id,
                    "corpus_name": c.corpus_name,
                    "n_assertions": n_assertions,
                    "n_runs": n_runs,
                }
            )
        return rows

    def _pick_active_corpus_id(
        store: AssertionStore, corpora: list[dict[str, Any]], requested: str | None
    ) -> str | None:
        """Honor ``?corpus=<id>`` if it points at a real corpus.

        Otherwise pick the corpus whose most recent run finished most
        recently; ties and run-less corpora fall back to the first row
        (already alphabetical via list_corpora's ORDER BY).
        """
        valid_ids = {c["corpus_id"] for c in corpora}
        if requested and requested in valid_ids:
            return requested
        if not corpora:
            return None
        row = store._conn.execute(
            "SELECT corpus_id FROM pipeline_runs "
            "WHERE corpus_id IS NOT NULL AND finished_at IS NOT NULL "
            "ORDER BY finished_at DESC LIMIT 1"
        ).fetchone()
        if row is not None and row[0] in valid_ids:
            return str(row[0])
        return str(corpora[0]["corpus_id"])

    def _most_recent_run_for_corpus(store: AssertionStore, corpus_id: str) -> dict[str, Any] | None:
        row = store._conn.execute(
            "SELECT run_id, started_at, finished_at, run_status "
            "FROM pipeline_runs WHERE corpus_id = ? "
            "ORDER BY started_at DESC, run_id DESC LIMIT 1",
            (corpus_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "started_at": row["started_at"],
            "completed_at": row["finished_at"] or "in progress",
            "run_status": row["run_status"],
        }

    def _collect_findings_for_run(
        store: AssertionStore,
        audit: AuditLogger,
        run_id: str,
        *,
        filter: str = "all",
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Pull pair findings (contradiction + numeric_short_circuit) and
        definition_divergent findings for the run, hydrate doc labels, and
        flatten to the shape ``cc_findings.html`` and the CSV export consume
        (the latter also reads the assertion-text and rationale keys).

        Returns ``(filtered_findings, counts)`` where ``counts`` has the
        per-filter totals (all/open/confirmed/false_positive/dismissed)
        across the unfiltered finding set so the filter chips can render
        accurate totals regardless of which filter is active.
        """
        raw_pair = [
            f
            for verdict in _PAIR_FINDING_VERDICTS
            for f in audit.iter_findings(run_id=run_id, verdict=verdict)
        ]
        raw_def = [
            f
            for verdict in _DEFINITION_VERDICTS
            for f in audit.iter_findings(
                run_id=run_id,
                verdict=verdict,
                detector_type="definition_inconsistency",
            )
        ]
        # Detector-type per raw row: pair findings (contradiction or
        # numeric_short_circuit) all post against detector_type="contradiction"
        # in the verdicts table; definition_divergent posts against
        # "definition_inconsistency".
        raw_with_dt: list[tuple[Any, str]] = [(r, "contradiction") for r in raw_pair] + [
            (r, "definition_inconsistency") for r in raw_def
        ]
        assertion_ids = [
            aid for raw, _dt in raw_with_dt for aid in (raw.assertion_a_id, raw.assertion_b_id)
        ]
        assertions = store.get_assertions_bulk(assertion_ids)
        doc_ids = list({a.doc_id for a in assertions.values()})
        documents = store.get_documents_bulk(doc_ids)

        keys = [
            (":".join(sorted([raw.assertion_a_id, raw.assertion_b_id])), dt)
            for raw, dt in raw_with_dt
        ]
        reviewer_verdicts = audit.get_reviewer_verdicts_bulk(keys)  # type: ignore[arg-type]

        out: list[dict[str, Any]] = []
        for raw, detector_type in raw_with_dt:
            a = assertions.get(raw.assertion_a_id)
            b = assertions.get(raw.assertion_b_id)
            if a is None or b is None:
                continue
            doc_a_label = _document_label(documents.get(a.doc_id), a.doc_id)
            doc_b_label = _document_label(documents.get(b.doc_id), b.doc_id)
            rationale_first = (raw.judge_rationale or "").splitlines()[:1]
            if rationale_first:
                label = rationale_first[0]
            else:
                label = (
                    f"{doc_a_label}: {_truncate(a.assertion_text, 140)} ↔ "
                    f"{doc_b_label}: {_truncate(b.assertion_text, 140)}"
                )
            pair_key = ":".join(sorted([raw.assertion_a_id, raw.assertion_b_id]))
            rv = reviewer_verdicts.get((pair_key, detector_type))
            reviewer_verdict = rv.verdict if rv is not None else None
            out.append(
                {
                    "finding_id": raw.finding_id,
                    "verdict": raw.judge_verdict or "uncertain",
                    "confidence": raw.judge_confidence,
                    "label_or_quote": label,
                    "pair_key": pair_key,
                    "detector_type": detector_type,
                    "reviewer_verdict": reviewer_verdict,
                    "reviewer_label": VERDICT_LABELS.get(reviewer_verdict)
                    if reviewer_verdict is not None
                    else None,
                    "assertion_a_text": a.assertion_text,
                    "assertion_b_text": b.assertion_text,
                    "doc_a_label": doc_a_label,
                    "doc_b_label": doc_b_label,
                    "judge_rationale": raw.judge_rationale,
                }
            )
        out.sort(key=lambda f: -(f["confidence"] or 0.0))

        # Compute counts BEFORE filtering so chips show real totals.
        counts: dict[str, int] = {
            "all": len(out),
            "open": sum(1 for f in out if f["reviewer_verdict"] is None),
            "confirmed": sum(1 for f in out if f["reviewer_verdict"] == "confirmed"),
            "false_positive": sum(1 for f in out if f["reviewer_verdict"] == "false_positive"),
            "dismissed": sum(1 for f in out if f["reviewer_verdict"] == "dismissed"),
        }

        if filter == "open":
            filtered = [f for f in out if f["reviewer_verdict"] is None]
        elif filter == "confirmed":
            filtered = [f for f in out if f["reviewer_verdict"] == "confirmed"]
        elif filter == "false_positive":
            filtered = [f for f in out if f["reviewer_verdict"] == "false_positive"]
        elif filter == "dismissed":
            filtered = [f for f in out if f["reviewer_verdict"] == "dismissed"]
        else:
            filtered = out

        return filtered, counts

    def _compute_run_cost(run: Any, audit: AuditLogger) -> float:
        """Upper-bound spend estimate for a run.

        Schema has no measured ``spent_usd`` column (Phase 5 caveat in
        ADR-0017), so multiply the count of judge calls (pair + definition)
        by the provider's per-call HIGH bound. Same formula ADR-0016 uses
        for the pre-flight ceiling gate; explicitly labelled an estimate
        in the rendered gauge.

        Definition-judge calls aren't stored on ``pipeline_runs`` directly,
        so they're counted by querying the findings table for this run —
        same source ``_live_counters`` uses.
        """
        _low, per_call_high = default_per_call_costs(config.judge_provider)
        n_pairs = int(getattr(run, "n_pairs_judged", 0) or 0)
        n_def = sum(
            1
            for _ in audit.iter_findings(
                run_id=run.run_id, detector_type="definition_inconsistency"
            )
        )
        return (n_pairs + n_def) * per_call_high

    def _shell_context(
        store: AssertionStore,
        audit: AuditLogger,
        active_corpus_id: str | None,
        *,
        filter: str = "all",
    ) -> dict[str, Any]:
        corpora = _list_corpora_with_counts(store)
        resolved_id = _pick_active_corpus_id(store, corpora, active_corpus_id)
        active_corpus = next((c for c in corpora if c["corpus_id"] == resolved_id), None)
        last_run: dict[str, Any] | None = None
        findings: list[dict[str, Any]] = []
        counts: dict[str, int] = {
            "all": 0,
            "open": 0,
            "confirmed": 0,
            "false_positive": 0,
            "dismissed": 0,
        }
        spent_usd: float | None = None
        if resolved_id is not None:
            last_run = _most_recent_run_for_corpus(store, resolved_id)
            if last_run is not None:
                findings, counts = _collect_findings_for_run(
                    store, audit, last_run["run_id"], filter=filter
                )
                run_obj = audit.get_run(last_run["run_id"])
                if run_obj is not None:
                    spent_usd = _compute_run_cost(run_obj, audit)
        return {
            "corpora": corpora,
            "active_corpus_id": resolved_id,
            "active_corpus": active_corpus,
            "last_run": last_run,
            "findings": findings,
            "filter": filter,
            "counts": counts,
            "spent_usd": spent_usd,
            "max_cost": config.max_cost_usd,
        }

    def _render_single_page_shell(
        request: Request, *, active_corpus_id: str | None, filter: str = "all"
    ) -> HTMLResponse:
        store, audit = _open_audit()
        try:
            ctx = _shell_context(store, audit, active_corpus_id, filter=filter)
        finally:
            store.close()
        return templates.TemplateResponse(request, "cc_single.html", ctx)

    @app.get("/corpora/sidebar", response_class=HTMLResponse)
    def corpora_sidebar(request: Request, active: str | None = Query(default=None)) -> HTMLResponse:
        """Sidebar fragment used by the ``corpus-created`` HTMX trigger.

        Returns the ``cc_sidebar.html`` partial — corpora list + new-corpus
        button — without the surrounding ``<aside>`` so the caller can
        ``hx-swap="innerHTML"`` into the existing sidebar shell. Honors
        ``?active=<corpus_id>`` so a just-created corpus row gets the
        active-row highlight on refresh.
        """
        store, _audit = _open_audit()
        try:
            corpora = _list_corpora_with_counts(store)
            active_corpus_id = (
                active if active and any(c["corpus_id"] == active for c in corpora) else None
            )
        finally:
            store.close()
        return templates.TemplateResponse(
            request,
            "cc_sidebar.html",
            {"corpora": corpora, "active_corpus_id": active_corpus_id},
        )

    @app.get("/corpora/new/modal", response_class=HTMLResponse)
    def corpora_new_modal(request: Request) -> HTMLResponse:
        """Modal fragment for creating a new corpus + ingesting files."""
        return templates.TemplateResponse(
            request,
            "cc_new_corpus_modal.html",
            {
                "allowed_extensions": sorted(_ALLOWED_EXTENSIONS),
                "success": False,
                "error": None,
                "corpus_name": "",
                "judge_provider": "moonshot",
            },
        )

    def _render_new_corpus_modal(
        request: Request,
        *,
        success: bool,
        error: str | None,
        status_code: int,
        corpus_name: str = "",
        judge_provider: str = "moonshot",
        corpus_id: str | None = None,
        n_assertions: int = 0,
        n_files: int = 0,
        hx_trigger: str | None = None,
    ) -> HTMLResponse:
        response = templates.TemplateResponse(
            request,
            "cc_new_corpus_modal.html",
            {
                "allowed_extensions": sorted(_ALLOWED_EXTENSIONS),
                "success": success,
                "error": error,
                "corpus_name": corpus_name,
                "judge_provider": judge_provider,
                "corpus_id": corpus_id,
                "n_assertions": n_assertions,
                "n_files": n_files,
            },
            status_code=status_code,
        )
        if hx_trigger:
            response.headers["HX-Trigger"] = hx_trigger
        return response

    @app.post("/corpora/new", response_class=HTMLResponse)
    async def post_corpora_new(
        request: Request,
        corpus_name: str = Form(...),
        judge_provider: str = Form("moonshot"),
        files: list[UploadFile] | None = None,
    ) -> HTMLResponse:
        """Create a corpus row and optionally ingest uploaded files.

        Wraps the existing corpus-creation + upload-ingest plumbing so the
        ADR-0017 New Corpus modal can do both in one round-trip. Empty file
        list is valid — empty corpora are a supported state.

        On success returns the modal in its success state with an
        ``HX-Trigger: corpus-created`` header so the sidebar refreshes.
        """
        corpus_name = (corpus_name or "").strip()
        provider = (judge_provider or "").strip()

        if not corpus_name or len(corpus_name) > 80:
            return _render_new_corpus_modal(
                request,
                success=False,
                error="Corpus name must be 1-80 characters.",
                status_code=400,
                corpus_name=corpus_name,
                judge_provider=provider or "moonshot",
            )
        invalid_chars = set(r'\/:"*?<>|')
        if any(c in corpus_name for c in invalid_chars):
            return _render_new_corpus_modal(
                request,
                success=False,
                error='Corpus name contains invalid characters: \\ / : " * ? < > |',
                status_code=400,
                corpus_name=corpus_name,
                judge_provider=provider or "moonshot",
            )
        if provider not in {"moonshot", "anthropic"}:
            return _render_new_corpus_modal(
                request,
                success=False,
                error="Judge provider must be 'moonshot' or 'anthropic'.",
                status_code=400,
                corpus_name=corpus_name,
                judge_provider="moonshot",
            )

        # Derive the path from a freshly minted id, never the raw name: the
        # name validator above lets ".." and bare dot-names through, so a
        # name-based path could escape data_dir/corpora.
        corpus_id = uuid.uuid4().hex
        corpora_root = (config.data_dir / "corpora").resolve()
        corpus_path = config.data_dir / "corpora" / corpus_id
        if not corpus_path.resolve().is_relative_to(corpora_root):
            return _render_new_corpus_modal(
                request,
                success=False,
                error="Could not derive a safe corpus path.",
                status_code=400,
                corpus_name=corpus_name,
                judge_provider=provider,
            )

        store = AssertionStore(config.db_path)
        store.migrate()
        duplicate = False
        try:
            existing = store._conn.execute(
                "SELECT 1 FROM corpora WHERE corpus_name = ?", (corpus_name,)
            ).fetchone()
            if existing is not None:
                duplicate = True
            else:
                # The SELECT above is a UX fast-path; the IntegrityError catch is
                # the real guard against a concurrent duplicate name racing past it.
                try:
                    store.create_corpus(corpus_id, corpus_name, str(corpus_path), provider)
                except sqlite3.IntegrityError:
                    duplicate = True
        finally:
            store.close()
        if duplicate:
            return _render_new_corpus_modal(
                request,
                success=False,
                error=f"Corpus '{corpus_name}' already exists.",
                status_code=409,
                corpus_name=corpus_name,
                judge_provider=provider,
            )

        upload_files = [f for f in (files or []) if f and f.filename]

        if not upload_files:
            return _render_new_corpus_modal(
                request,
                success=True,
                error=None,
                status_code=200,
                corpus_name=corpus_name,
                judge_provider=provider,
                corpus_id=corpus_id,
                n_assertions=0,
                n_files=0,
                hx_trigger="corpus-created",
            )

        upload_id = _generate_upload_id()
        upload_dir = config.data_dir / "uploads" / upload_id
        upload_dir.mkdir(parents=True, exist_ok=False)

        saved: list[Path] = []
        try:
            if len(upload_files) > MAX_UPLOAD_FILES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Too many files (max {MAX_UPLOAD_FILES} per request)",
                )
            total_bytes = 0
            for file in upload_files:
                # upload_files is pre-filtered to truthy filenames above; assert for mypy.
                assert file.filename is not None
                ext = Path(file.filename).suffix.lower()
                if not ext:
                    raise HTTPException(
                        status_code=400,
                        detail="File has no extension; expected .txt, .md, .pdf, or .docx",
                    )
                if ext not in _ALLOWED_EXTENSIONS:
                    raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext!r}")
                content = await file.read(MAX_UPLOAD_BYTES + 1)
                if len(content) > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File too large (max 100 MB)")
                total_bytes += len(content)
                if total_bytes > MAX_UPLOAD_TOTAL_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Upload too large (max {MAX_UPLOAD_TOTAL_BYTES // (1024 * 1024)} "
                            "MB total per request)"
                        ),
                    )
                target = upload_dir / Path(file.filename).name
                target.write_bytes(content)
                saved.append(target)
        except HTTPException as exc:
            shutil.rmtree(upload_dir, ignore_errors=True)
            # Roll back the corpus row created above so the user can retry the
            # same name; otherwise this empty ghost row would 409 every future
            # attempt. No store is open in this scope, so open a fresh one.
            rollback_store = AssertionStore(config.db_path)
            rollback_store.migrate()
            try:
                rollback_store.delete_corpus(corpus_id)
            finally:
                rollback_store.close()
            return _render_new_corpus_modal(
                request,
                success=False,
                error=f"Upload rejected: {exc.detail}",
                status_code=exc.status_code,
                corpus_name=corpus_name,
                judge_provider=provider,
            )

        store, faiss_store, embedder_inst = _open_stores()
        ingest_error: tuple[int, str] | None = None
        try:
            extractor = _make_extractor()
            try:
                n_assertions = _ingest_uploaded_paths(
                    saved,
                    store=store,
                    faiss_store=faiss_store,
                    embedder=embedder_inst,
                    extractor=extractor,
                    config=config,
                    corpus_id=corpus_id,
                )
            except CrossCorpusDocumentError as err:
                names_by_id = {c.corpus_id: c.corpus_name for c in store.list_corpora()}
                existing_name = names_by_id.get(err.existing_corpus_id, err.existing_corpus_id)
                requested_name = names_by_id.get(err.requested_corpus_id, err.requested_corpus_id)
                ingest_error = (
                    409,
                    (
                        f"Document {err.doc_id} already exists under corpus "
                        f"'{existing_name}' and cannot be re-uploaded into corpus "
                        f"'{requested_name}'."
                    ),
                )
                shutil.rmtree(upload_dir, ignore_errors=True)
                store.delete_corpus(corpus_id)
            except Exception as exc:
                # PR #73 pattern: render a user-facing error modal instead of
                # letting FastAPI propagate a 500 traceback into the HTMX swap
                # target. Roll back the corpus row created above so the user
                # can retry the same name; otherwise this empty ghost row would
                # 409 every future attempt.
                _log.exception("New-corpus ingest failed: %s", exc)
                shutil.rmtree(upload_dir, ignore_errors=True)
                store.delete_corpus(corpus_id)
                ingest_error = (
                    500,
                    "Ingest failed unexpectedly. The corpus has been rolled back; you can retry.",
                )
        finally:
            store.close()
        if ingest_error is not None:
            status_code, message = ingest_error
            return _render_new_corpus_modal(
                request,
                success=False,
                error=message,
                status_code=status_code,
                corpus_name=corpus_name,
                judge_provider=provider,
            )

        _log.info(
            "New corpus '%s' (%s): %d files saved, %d assertions extracted",
            corpus_name,
            corpus_id,
            len(saved),
            n_assertions,
        )
        return _render_new_corpus_modal(
            request,
            success=True,
            error=None,
            status_code=200,
            corpus_name=corpus_name,
            judge_provider=provider,
            corpus_id=corpus_id,
            n_assertions=n_assertions,
            n_files=len(saved),
            hx_trigger="corpus-created",
        )

    @app.get("/corpora/{corpus_id}/findings", response_class=HTMLResponse)
    def corpora_findings(
        request: Request,
        corpus_id: str,
        filter: str = "all",
    ) -> HTMLResponse:
        """Main-pane fragment for an HTMX swap from the sidebar.

        Returns only the ``cc_findings.html`` body — no ``<html>`` / ``<head>``
        wrapper — because the caller targets ``#cc-main`` with
        ``hx-swap="innerHTML"``.

        404s on unknown ``corpus_id``: the path parameter is the resource
        identity, so silently falling through to another corpus would hand
        callers (stale HTMX links after a corpus deletion) a 200 with the
        wrong data. The shell route ``GET /?corpus=<id>`` is more
        forgiving — there ``corpus`` is a query hint, not the identity.
        """
        if filter not in {"all", "open", "confirmed", "false_positive", "dismissed"}:
            filter = "all"
        store, audit = _open_audit()
        try:
            row = store._conn.execute(
                "SELECT 1 FROM corpora WHERE corpus_id = ?", (corpus_id,)
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"corpus_id {corpus_id!r} not found")
            ctx = _shell_context(store, audit, corpus_id, filter=filter)
        finally:
            store.close()
        return templates.TemplateResponse(request, "cc_findings.html", ctx)

    @app.get("/corpora/{corpus_id}/findings.csv")
    def corpora_findings_csv(
        request: Request,
        corpus_id: str,
        filter: str = "all",
    ) -> Response:
        """Download the on-screen findings as CSV (closes the ADR-0017 deferred CSV-export gap).

        Exports the active corpus's LATEST run honoring the active filter
        chip, so the file matches exactly what the findings pane shows. A
        corpus with no run / no findings yields a valid header-only CSV.
        """
        if filter not in {"all", "open", "confirmed", "false_positive", "dismissed"}:
            filter = "all"
        store, audit = _open_audit()
        try:
            row = store._conn.execute(
                "SELECT 1 FROM corpora WHERE corpus_id = ?", (corpus_id,)
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"corpus_id {corpus_id!r} not found")
            last_run = _most_recent_run_for_corpus(store, corpus_id)
            findings: list[dict[str, Any]] = []
            if last_run is not None:
                findings, _counts = _collect_findings_for_run(
                    store, audit, last_run["run_id"], filter=filter
                )
        finally:
            store.close()

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "finding_type",
                "judge_verdict",
                "confidence",
                "reviewer_verdict",
                "doc_a",
                "assertion_a",
                "doc_b",
                "assertion_b",
                "rationale",
            ]
        )
        for f in findings:
            writer.writerow(
                [
                    f["detector_type"],
                    f["verdict"],
                    f["confidence"],
                    f["reviewer_verdict"] or "",
                    f["doc_a_label"],
                    f["assertion_a_text"],
                    f["doc_b_label"],
                    f["assertion_b_text"],
                    f["judge_rationale"] or "",
                ]
            )
        return Response(
            content=buf.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="findings-{corpus_id}-{filter}.csv"'
            },
        )

    # --- ADR-0017 Phase 3: Run Check modal + per-corpus SSE progress ------

    def _corpus_name_or_404(store: AssertionStore, corpus_id: str) -> str:
        row = store._conn.execute(
            "SELECT corpus_name FROM corpora WHERE corpus_id = ?", (corpus_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"corpus_id {corpus_id!r} not found")
        return str(row[0])

    def _render_new_run_modal(
        request: Request,
        *,
        corpus_id: str,
        corpus_name: str,
        success: bool,
        error: str | None,
        status_code: int,
        pairwise: str = "",
        no_definitions: bool = False,
        deep: bool = False,
        max_cost: float | None = None,
        run_id: str | None = None,
        hx_trigger: str | None = None,
    ) -> HTMLResponse:
        response = templates.TemplateResponse(
            request,
            "cc_new_run_modal.html",
            {
                "corpus_id": corpus_id,
                "corpus_name": corpus_name,
                "config_pairwise_enabled": config.pairwise_enabled,
                "success": success,
                "error": error,
                "pairwise": pairwise,
                "no_definitions": no_definitions,
                "deep": deep,
                "max_cost": max_cost,
                "run_id_short": (run_id[:8] if run_id else None),
            },
            status_code=status_code,
        )
        if hx_trigger:
            response.headers["HX-Trigger"] = hx_trigger
        return response

    @app.get("/corpora/{corpus_id}/run/modal", response_class=HTMLResponse)
    def corpora_run_modal(request: Request, corpus_id: str) -> HTMLResponse:
        """Render the per-corpus Run Check modal pre-filled with defaults."""
        store, _audit = _open_audit()
        try:
            corpus_name = _corpus_name_or_404(store, corpus_id)
        finally:
            store.close()
        return _render_new_run_modal(
            request,
            corpus_id=corpus_id,
            corpus_name=corpus_name,
            success=False,
            error=None,
            status_code=200,
        )

    @app.post("/corpora/{corpus_id}/run", response_class=HTMLResponse)
    def post_corpora_run(
        request: Request,
        corpus_id: str,
        background_tasks: BackgroundTasks,
        pairwise: str = Form(""),
        no_definitions: bool = Form(False),
        deep: bool = Form(False),
        max_cost: float | None = Form(None),
    ) -> HTMLResponse:
        """Begin a check run for ``corpus_id``.

        Mirrors ``POST /runs`` but scoped to the path-parameter corpus and
        with the ADR-0017 modal toggles (pairwise tri-state, no_definitions,
        max_cost) threaded through ``_run_check_in_background``.
        """
        pairwise = (pairwise or "").strip().lower()
        if pairwise not in {"", "true", "false"}:
            raise HTTPException(status_code=400, detail=f"unknown pairwise value {pairwise!r}")
        # Validate max_cost: CLI uses typer.Option(min=0); the web layer must
        # match. Negative or non-finite floats bypass the cost ceiling because
        # `model_copy(update=...)` skips Pydantic field validators on
        # frozen models, so the guard has to live here.
        if max_cost is not None and (not math.isfinite(max_cost) or max_cost < 0):
            raise HTTPException(
                status_code=422,
                detail="max_cost must be a finite non-negative number",
            )

        store, audit = _open_audit()
        try:
            corpus_name = _corpus_name_or_404(store, corpus_id)

            # Reject double-submit: if a run is already pending/running on
            # this corpus, return 409 before begin_run so we don't spawn
            # parallel background tasks racing on the same SQLite + FAISS.
            existing_active = store._conn.execute(
                "SELECT run_id FROM pipeline_runs "
                "WHERE corpus_id = ? AND run_status IN ('pending', 'running') LIMIT 1",
                (corpus_id,),
            ).fetchone()
            if existing_active is not None:
                return _render_new_run_modal(
                    request,
                    corpus_id=corpus_id,
                    corpus_name=corpus_name,
                    success=False,
                    error="A run is already in progress on this corpus. Wait for it to finish.",
                    status_code=409,
                    pairwise=pairwise,
                    no_definitions=no_definitions,
                    deep=deep,
                    max_cost=max_cost,
                )

            pairwise_override: bool | None = None if pairwise == "" else (pairwise == "true")
            effective_pairwise = (
                pairwise_override if pairwise_override is not None else config.pairwise_enabled
            )
            if deep and not effective_pairwise:
                return _render_new_run_modal(
                    request,
                    corpus_id=corpus_id,
                    corpus_name=corpus_name,
                    success=False,
                    error=(
                        "'deep' (multi-party) requires the pairwise gate output. "
                        "Enable pairwise_enabled in the server config or do not "
                        "select 'deep'."
                    ),
                    status_code=400,
                    pairwise=pairwise,
                    no_definitions=no_definitions,
                    deep=deep,
                    max_cost=max_cost,
                )

            # ADR-0017 review: mirror the CLI's begin_run config dict
            # exactly so audit-log replay sees one shape regardless of
            # whether the run was started from `consistency-check check`
            # or the web Run Check modal. Aligned with the CLI `begin_run`
            # config in `cli/main.py`.
            run_id = audit.begin_run(
                config={
                    "embedder_model": config.embedder_model,
                    "nli_model": config.nli_model,
                    "judge_provider": config.judge_provider,
                    "judge_model": config.judge_model,
                    "nli_contradiction_threshold": config.nli_contradiction_threshold,
                    "gate_top_k": config.gate_top_k,
                    "gate_similarity_threshold": config.gate_similarity_threshold,
                    "enable_multi_party": deep,
                    "max_triangles_per_run": config.max_triangles_per_run,
                    "definitions_enabled": not no_definitions,
                    "pairwise_enabled": effective_pairwise,
                    "max_cost_usd": (max_cost if max_cost is not None else config.max_cost_usd),
                },
                run_status="pending",
                corpus_id=corpus_id,
            )
        finally:
            store.close()

        background_tasks.add_task(
            _run_check_in_background,
            run_id,
            deep,
            corpus_id,
            pairwise_override=pairwise_override,
            no_definitions=no_definitions,
            max_cost_override=max_cost,
        )

        return _render_new_run_modal(
            request,
            corpus_id=corpus_id,
            corpus_name=corpus_name,
            success=True,
            error=None,
            status_code=200,
            pairwise=pairwise,
            no_definitions=no_definitions,
            deep=deep,
            max_cost=max_cost,
            run_id=run_id,
            hx_trigger="run-started",
        )

    def _render_cost_gauge(snapshot: dict[str, Any] | None) -> str:
        """Render the inline cost gauge HTML for one SSE snapshot.

        Mirrors the cc__cost_gauge.html fragment but inline here so the SSE
        path doesn't have to round-trip through Jinja per tick. ``spent_usd``
        is the upper-bound estimate (judge calls * provider per-call high),
        explicitly labelled an estimate per ADR-0017 Phase 5.
        """
        max_cost = config.max_cost_usd
        if snapshot is None:
            return (
                'Cost: <span class="cc-cost-spent">--</span> / '
                '<span class="cc-cost-ceiling">--</span>'
            )
        _low, per_call_high = default_per_call_costs(config.judge_provider)
        n_pairs = int(snapshot.get("n_pairs_judged") or 0)
        n_def = int(snapshot.get("n_definition_pairs_judged") or 0)
        spent = (n_pairs + n_def) * per_call_high
        ceiling = (
            f'<span class="cc-cost-ceiling">${max_cost:.4f}</span>'
            if max_cost is not None
            else '<span class="cc-cost-ceiling">no limit</span>'
        )
        return f'Est. spent: <span class="cc-cost-spent">${spent:.4f}</span> / {ceiling}'

    def _render_progress_snapshot(snapshot: dict[str, Any]) -> str:
        """Render the inline progress bar HTML for one SSE snapshot.

        Server-side rendering keeps the sidebar JS-free: the htmx-sse
        extension just innerHTML-swaps whatever we emit.
        """
        status = snapshot.get("status") or "pending"
        gated = snapshot.get("n_pairs_gated") or 0
        judged = snapshot.get("n_pairs_judged") or 0
        findings = snapshot.get("n_findings") or 0
        pct = 0
        if gated and judged:
            pct = max(0, min(100, round(100 * judged / gated)))
        elif status == "done":
            pct = 100
        # Fall back to a safe literal — `status` flows from the SQLite
        # run_status column and is innerHTML'd by the sidebar; an unknown
        # value (future migration, manual DB write) must not become an
        # XSS sink. The autoescape applies to template renders, not to
        # raw HTML strings concatenated here.
        status_label = {
            "pending": "Queued",
            "running": "Running",
            "done": "Done",
            "failed": "Failed",
        }.get(status, "Unknown")
        modifier = ""
        if status == "done":
            modifier = " cc-progress-bar--done"
        elif status == "failed":
            modifier = " cc-progress-bar--failed"
        return (
            '<div class="cc-progress-label">'
            f"{status_label} · {judged}/{gated} judged · {findings} findings"
            "</div>"
            f'<div class="cc-progress-bar{modifier}">'
            f'<div class="cc-progress-bar__fill" style="width: {pct}%"></div>'
            "</div>"
        )

    def _read_latest_run_snapshot(corpus_id: str) -> dict[str, Any] | None:
        """Open a short-lived store and return the most recent run snapshot.

        Each call uses its own connection so we don't hold a SQLite handle
        open inside the async generator across ``await asyncio.sleep``.
        Returns ``None`` when the corpus has no runs at all.
        """
        store, audit = _open_audit()
        try:
            row = store._conn.execute(
                "SELECT run_id FROM pipeline_runs WHERE corpus_id = ? "
                "ORDER BY started_at DESC, run_id DESC LIMIT 1",
                (corpus_id,),
            ).fetchone()
            if row is None:
                return None
            run = audit.get_run(str(row[0]))
            if run is None:
                return None
            counters = _live_counters(run, audit)
            return {
                "type": "snapshot",
                "run_id": run.run_id,
                "status": run.run_status,
                "n_pairs_gated": counters["n_pairs_gated"],
                "n_pairs_judged": counters["n_pairs_judged"],
                "n_findings": counters["n_findings"],
                "n_definition_pairs_judged": counters["n_definition_pairs_judged"],
                "started_at": counters["started_at"],
                "finished_at": counters["finished_at"],
            }
        finally:
            store.close()

    async def _progress_sse_generator(corpus_id: str) -> AsyncGenerator[str, None]:
        """Emit one snapshot per poll until the active run terminates.

        Wraps the body in try/except so a SQLite error mid-stream (lock
        contention, disk full) doesn't drop the connection silently and
        trigger an infinite EventSource reconnect loop on the client.
        """
        max_iter = PROGRESS_MAX_ITERATIONS
        poll_seconds = PROGRESS_POLL_SECONDS
        tail_seconds = PROGRESS_DONE_TAIL_SECONDS
        terminal_emits = 0
        # Cap the post-terminal tail by integer ticks of the poll interval so
        # the tests' poll_seconds=0 setting doesn't spin a busy loop.
        max_terminal_emits = max(1, round(tail_seconds / poll_seconds)) if poll_seconds > 0 else 1

        try:
            for _ in range(max_iter):
                snapshot = _read_latest_run_snapshot(corpus_id)
                if snapshot is None:
                    # No run yet, or corpus has none. Tell the client we're
                    # done so it doesn't hold the connection open.
                    yield "data: " + _render_progress_snapshot({"status": "pending"}) + "\n\n"
                    yield 'event: done\ndata: {"type":"done"}\n\n'
                    return

                payload = json.dumps(snapshot)
                html = _render_progress_snapshot(snapshot)
                yield f"event: snapshot\ndata: {html}\n\n"
                cost_html = _render_cost_gauge(snapshot)
                yield f"event: cost_update\ndata: {cost_html}\n\n"
                # Also expose the raw JSON payload as a debug-friendly default
                # event for downstream consumers (tests, future native JS).
                yield f"data: {payload}\n\n"

                if snapshot["status"] in {"done", "failed"}:
                    terminal_emits += 1
                    if terminal_emits >= max_terminal_emits:
                        yield 'event: done\ndata: {"type":"done"}\n\n'
                        return

                await asyncio.sleep(poll_seconds)
        except Exception as exc:
            _log.exception("SSE progress generator failed for corpus %s: %s", corpus_id, exc)
            # Emit an `error` event so the client knows the stream died for
            # a server-side reason, then a `done` so the EventSource closes
            # cleanly instead of looping.
            yield 'event: error\ndata: {"type":"error","message":"server error"}\n\n'

        # Exhausted the iteration cap (or hit the except above) — close cleanly.
        yield 'event: done\ndata: {"type":"done"}\n\n'

    @app.get("/corpora/{corpus_id}/progress")
    async def corpora_progress(corpus_id: str) -> StreamingResponse:
        """Per-corpus SSE channel for run progress (ADR-0017 Phase 3).

        404s on unknown ``corpus_id`` rather than streaming an empty channel,
        so stale sidebar rows surface their bug instead of hanging silently.

        Declared ``async def`` so the inner ``asyncio.sleep`` inside the
        generator yields to the event loop instead of blocking a thread-pool
        worker on each poll tick.
        """
        store, _audit = _open_audit()
        try:
            _corpus_name_or_404(store, corpus_id)
        finally:
            store.close()
        return StreamingResponse(
            _progress_sse_generator(corpus_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # --- ADR-0017 Phase 4: per-corpus slide-over drawers --------------------
    #
    # The drawer routes render the body templates (cc_assertions.html,
    # cc_definitions.html, cc_stats.html) by wrapping them in cc_drawer.html
    # via {% include body_template %}. Since Phase 6 deleted the legacy tab
    # shell, these templates are drawer-body fragments only.

    DRAWER_PAGE_SIZE = 25

    @app.get("/corpora/{corpus_id}/drawer/assertions", response_class=HTMLResponse)
    def drawer_assertions(request: Request, corpus_id: str, page: int = 1) -> HTMLResponse:
        store, _audit = _open_audit()
        try:
            _corpus_name_or_404(store, corpus_id)
            total = store.stats(corpus_id=corpus_id)["assertions"]
            pag = _pagination(page=page, total=total)
            offset = (pag["page"] - 1) * DRAWER_PAGE_SIZE
            page_assertions = list(
                store.iter_assertions(limit=DRAWER_PAGE_SIZE, offset=offset, corpus_id=corpus_id)
            )
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
        # PR #77 review fix: cc__pagination.html hardcodes
        # hx-target="#cc-tab-content", which is the legacy tab DOM and does
        # not exist inside the drawer. Without these overrides the Prev/Next
        # buttons silently no-op as soon as a corpus has > DRAWER_PAGE_SIZE
        # assertions. Pass the drawer-scoped URL + target so pagination
        # stays in-drawer.
        pagination_url = f"/corpora/{corpus_id}/drawer/assertions"
        return templates.TemplateResponse(
            request,
            "cc_drawer.html",
            {
                "title": "Assertions",
                "body_template": "cc_assertions.html",
                "htmx": True,
                "active_tab": "assertions",
                "assertions": rows,
                "pagination": pag,
                "pagination_url": pagination_url,
                "pagination_target": "#cc-drawer-region",
                "pagination_swap": "innerHTML",
            },
        )

    @app.get("/corpora/{corpus_id}/drawer/definitions", response_class=HTMLResponse)
    def drawer_definitions(
        request: Request,
        corpus_id: str,
        show_reviewed_definition_inconsistency: bool = False,
    ) -> HTMLResponse:
        store, audit = _open_audit()
        try:
            _corpus_name_or_404(store, corpus_id)
            # Scope to the most recent run for this corpus, not the global
            # most_recent_run() — otherwise the drawer for corpus A could show
            # definitions from a run on corpus B.
            row = store._conn.execute(
                "SELECT run_id FROM pipeline_runs WHERE corpus_id = ? "
                "ORDER BY started_at DESC, run_id DESC LIMIT 1",
                (corpus_id,),
            ).fetchone()
            run = audit.get_run(str(row[0])) if row is not None else None
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
            # PR #77 review fix: the counter must agree with the per-corpus
            # findings list, but `count_reviewer_verdicts` is content-addressed
            # (one verdict spans corpora by design — see migration 0009) so a
            # corpus-scoped numerator would require joining via the findings
            # table. Until that join is added, suppress the counter inside the
            # drawer rather than show a mismatched ratio. The global definition
            # count is computed here via `_count_total_findings`.
            reviewed_count = None
            total_count = _count_total_findings(
                store, "definition_inconsistency", corpus_id=corpus_id
            )
        finally:
            store.close()
        # Override the Show-reviewed toggle's hx-get/hx-target so the
        # checkbox refreshes the drawer in-place instead of trying to
        # swap a #cc-tab-content element that only exists in the legacy
        # tab shell.
        drawer_url = f"/corpora/{corpus_id}/drawer/definitions"
        return templates.TemplateResponse(
            request,
            "cc_drawer.html",
            {
                "title": "Definitions",
                "body_template": "cc_definitions.html",
                "htmx": True,
                "active_tab": "definitions",
                "run": {"run_id": run.run_id} if run is not None else None,
                "findings": findings,
                "show_reviewed": show_reviewed_definition_inconsistency,
                "reviewed_count": reviewed_count,
                "total_count": total_count,
                "drawer_refresh_url": drawer_url,
                "drawer_refresh_target": "#cc-drawer-region",
                "drawer_refresh_swap": "innerHTML",
            },
        )

    @app.get("/corpora/{corpus_id}/drawer/stats", response_class=HTMLResponse)
    def drawer_stats(request: Request, corpus_id: str) -> HTMLResponse:
        store, audit = _open_audit()
        try:
            _corpus_name_or_404(store, corpus_id)
            # Per-corpus most-recent run, mirroring the query in
            # ``_read_latest_run_snapshot`` (the SSE generator's helper) so the
            # drawer and the sidebar agree on which run they're describing.
            row = store._conn.execute(
                "SELECT run_id FROM pipeline_runs WHERE corpus_id = ? "
                "ORDER BY started_at DESC, run_id DESC LIMIT 1",
                (corpus_id,),
            ).fetchone()
            run = audit.get_run(str(row[0])) if row is not None else None
            status = _infer_run_status(run)
            counters = _live_counters(run, audit) if run is not None else None
            banner = _corpus_banner_context(
                store,
                config,
                run_id=run.run_id if run is not None else None,
                corpus_id=corpus_id,
            )
        finally:
            store.close()
        return templates.TemplateResponse(
            request,
            "cc_drawer.html",
            {
                "title": "Stats",
                "body_template": "cc_stats.html",
                "htmx": True,
                "active_tab": "stats",
                "status": status,
                "counters": counters,
                **banner,
            },
        )

    @app.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        legacy: bool = False,
        corpus: str | None = None,
        filter: str = "all",
    ) -> Response:
        """Main page — the ADR-0017 single-page shell (now the default).

        ``?corpus=<id>`` selects which corpus the shell loads with and
        ``?filter=open|confirmed|...`` narrows the findings list. ``?legacy=1``
        is a tombstone for the deleted 7-tab UI and returns 410 Gone.
        """
        if legacy:
            return PlainTextResponse(_LEGACY_GONE_BODY, status_code=410)
        if filter not in {"all", "open", "confirmed", "false_positive", "dismissed"}:
            filter = "all"
        return _render_single_page_shell(request, active_corpus_id=corpus, filter=filter)

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

    def _run_check_in_background(
        run_id: str,
        deep: bool,
        corpus_id: str,
        *,
        pairwise_override: bool | None = None,
        no_definitions: bool = False,
        max_cost_override: float | None = None,
    ) -> None:
        # SQLite handles can't safely be shared across threads, so the
        # background task opens its own store. The embedder is not needed for
        # check (only for ingest), so open faiss directly to avoid the ~800 MB
        # model load that _open_stores() would trigger.
        #
        # ADR-0017 Phase 3 overrides: when the Run Check modal posts an explicit
        # pairwise / no_definitions / max_cost value, derive a per-run config so
        # the modal-driven flow respects those flags without mutating the
        # process-wide ``config``. None overrides leave the app-level value in
        # effect; the sole entry point is ``POST /corpora/{id}/run``.
        from consistency_checker.pipeline import CostCeilingExceeded
        from consistency_checker.pipeline import check as run_check

        store = AssertionStore(config.db_path)
        store.migrate()
        audit_logger = AuditLogger(store)
        try:
            # Belt-and-suspenders: the POST route rejects non-finite /
            # negative max_cost, but Pydantic's model_copy(update=...) skips
            # field validators on frozen models, so a stray legacy caller
            # could otherwise smuggle a bad value past Config.max_cost_usd's
            # ge=0 constraint. Re-run validators via model_validate on the
            # merged values and treat any failure as a clean "failed" run
            # (not a silent process-level crash).
            try:
                updates: dict[str, Any] = {"enable_multi_party": deep}
                if pairwise_override is not None:
                    updates["pairwise_enabled"] = pairwise_override
                if max_cost_override is not None:
                    if not math.isfinite(max_cost_override) or max_cost_override < 0:
                        raise ValueError(
                            f"max_cost_override {max_cost_override} is invalid "
                            "(must be a finite non-negative number)."
                        )
                    updates["max_cost_usd"] = max_cost_override
                effective_cfg = config.model_validate(
                    {**config.model_dump(), **updates},
                )
            except Exception as exc:
                _log.warning("Run %s aborted at config validation: %s", run_id, exc)
                audit_logger.update_run_status(
                    run_id,
                    "failed",
                    error_message="Invalid run configuration. See server logs for details.",
                )
                return

            faiss_store = FaissStore.open_or_create(
                index_path=effective_cfg.faiss_path,
                id_map_path=effective_cfg.faiss_path.with_suffix(".idmap.json"),
            )
            try:
                # ADR-0015: only construct the NLI checker when the pairwise
                # pass is enabled. Skipping it avoids the ~800 MB model
                # download / RSS hit for operators who only want the
                # definition pass.
                nli_inst: NliChecker | None = None
                if effective_cfg.pairwise_enabled:
                    if nli_checker is None:
                        from consistency_checker.check.nli_checker import (
                            TransformerNliChecker,
                        )

                        nli_inst = TransformerNliChecker(model_name=effective_cfg.nli_model)
                    else:
                        nli_inst = nli_checker
                if judge is None:
                    from consistency_checker.pipeline import make_judge

                    judge_inst: Judge = make_judge(effective_cfg)
                else:
                    judge_inst = judge
                mp_judge = multi_party_judge
                if mp_judge is None and deep:
                    from consistency_checker.pipeline import make_multi_party_judge

                    mp_judge = make_multi_party_judge(effective_cfg)

                from consistency_checker.check.definition_checker import (
                    DefinitionChecker,
                )

                def_checker: DefinitionChecker | None
                if no_definitions:
                    # Mirror CLI's `--no-definitions`: skip the definition pass
                    # entirely (the pipeline treats `definition_checker=None`
                    # as "definitions disabled").
                    def_checker = None
                elif definition_judge is not None:
                    def_checker = DefinitionChecker(
                        judge=definition_judge,
                        org_scope_enabled=effective_cfg.org_scope_enabled,
                    )
                else:
                    from consistency_checker.pipeline import make_definition_checker

                    def_checker = make_definition_checker(effective_cfg)

                run_check(
                    effective_cfg,
                    store=store,
                    faiss_store=faiss_store,
                    nli_checker=nli_inst,
                    judge=judge_inst,
                    audit_logger=audit_logger,
                    multi_party_judge=mp_judge if deep else None,
                    definition_checker=def_checker,
                    run_id=run_id,
                    corpus_id=corpus_id,
                )
            except CostCeilingExceeded as exc:
                # ADR-0016: surface the budget-guardrail outcome with a clean
                # diagnostic instead of dumping a generic traceback into the
                # UI's failure field.
                _log.info("Run %s aborted by max_cost_usd ceiling: %s", run_id, exc)
                audit_logger.update_run_status(
                    run_id,
                    "failed",
                    error_message=(
                        f"Estimated cost ${exc.estimated_high:.4f} exceeds "
                        f"max_cost_usd ${exc.ceiling:.4f}. Raise the ceiling, "
                        f"disable pairwise/definitions, or narrow the corpus."
                    ),
                )
            except Exception as exc:
                _log.exception("Run %s failed: %s", run_id, exc)
                audit_logger.update_run_status(
                    run_id, "failed", error_message=_GENERIC_FAILURE_MESSAGE
                )
        finally:
            store.close()

    @app.get("/runs/{run_id}/stats", response_class=HTMLResponse)
    def run_stats_fragment(
        request: Request,
        run_id: str,
        corpus: str | None = Query(default=None),
    ) -> HTMLResponse:
        # Returning the final fragment (no polling attrs) is how the
        # self-polling loop stops once the run terminates.
        store, audit = _open_audit()
        try:
            run = audit.get_run(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail=f"run {run_id} not found")
            status = _infer_run_status(run)
            counters = _live_counters(run, audit)
            banner = _corpus_banner_context(store, config, run_id=run.run_id, corpus_id=corpus)
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
            {"status": status, "counters": counters, **banner},
        )

    @app.get("/tabs/{tab}", response_class=PlainTextResponse)
    def legacy_tab_gone(tab: str) -> PlainTextResponse:
        """ADR-0017 Phase 6 tombstone for the deleted 7-tab UI.

        Every former tab (documents, assertions, definitions, stats, ingest,
        action_items, process) now lives in the single-page shell or its
        drawers. The live per-corpus routes are ``/corpora/{id}/drawer/*`` and
        ``/runs/{id}/stats`` — none of which match this ``/tabs/{tab}`` path.
        """
        return PlainTextResponse(_LEGACY_GONE_BODY, status_code=410)

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
        if _PAIR_KEY_RE.fullmatch(pair_key) is None:
            raise HTTPException(status_code=400, detail="invalid pair_key format")

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
        if _PAIR_KEY_RE.fullmatch(pair_key) is None:
            raise HTTPException(status_code=400, detail="invalid pair_key format")
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
        # Callers (cc__verdict_toast.html, cc_findings.html) have
        # hx-target="#cc-toast-region" hx-swap="afterbegin"; we tell HTMX to
        # redirect back to the current URL so it re-fetches fresh content.
        referer = request.headers.get("HX-Current-URL") or request.headers.get("Referer", "/")
        response = HTMLResponse(content="")
        response.headers["HX-Redirect"] = referer
        return response

    return app


def _ingest_uploaded_paths(
    paths: list[Path],
    *,
    store: AssertionStore,
    faiss_store: FaissStore,
    embedder: Embedder,
    extractor: Extractor,
    config: Config,
    corpus_id: str,
) -> int:
    """Run loader → chunker → extractor → embedder on a list of file paths.

    Mirrors :func:`pipeline.ingest` but takes explicit paths instead of walking
    a directory — the web upload doesn't have a single corpus_dir, just a
    handful of just-saved files. Returns the number of assertions added.
    """
    junk_audit = (
        JunkAudit(config.data_dir / "junk_drops.jsonl") if config.junk_filter_enabled else None
    )
    ocr_audit = OcrAudit(config.data_dir / "ocr_events.jsonl") if config.ocr_enabled else None
    n_assertions = 0
    for path in paths:
        loaded = load_path(
            path,
            junk_filter_enabled=config.junk_filter_enabled,
            junk_audit=junk_audit,
            ocr_enabled=config.ocr_enabled,
            ocr_audit=ocr_audit,
        )
        doc = loaded.document
        if config.org_grouping_enabled:
            res = extractor.identify_org(title=doc.title, text=loaded.text)
            doc = replace(doc, org_label=res.label, org_reason=res.reason)
        store.add_document(doc, corpus_id=corpus_id)
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
    if junk_audit is not None and junk_audit.counts:
        _log.info("Junk filter dropped (text stage): %s", junk_audit.counts)
    if ocr_audit is not None and ocr_audit.counts:
        _log.info("OCR fallback: %s", ocr_audit.counts)
    return n_assertions
