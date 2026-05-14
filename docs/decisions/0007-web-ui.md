# ADR 0007 — Web UI: FastAPI + HTMX, single-page-ish, SQLite-backed

**Status**: Accepted

## Context

v0.2 shipped the full pair-and-triangle contradiction pipeline behind a Typer CLI. The user-facing workflow — "edit `config.yml`, run `consistency-check ingest`, run `consistency-check check`, run `consistency-check report --out X.md`, open `X.md` in a renderer" — works but has friction: there's no way to drop files into the system, no live view of a long-running scan, and the report is a one-shot artefact rather than something browsable. A web UI replaces the loop with a single localhost app: drop docs, watch the run progress, click into each contradiction.

Options considered:

- **Streamlit.** Smallest line count, native Python. Trade-off: locks the UI to Streamlit's idioms, full reruns on every interaction, harder to test the rendered HTML against golden fixtures.
- **FastAPI + Jinja2 SSR (no JS framework).** Clean separation, pure server-side rendering, hermetic to test via `TestClient`. Trade-off: no live progress updates without polling-via-page-reload; tab navigation costs a full reload.
- **FastAPI + HTMX (chosen).** Same backend as Jinja2 SSR. HTMX adds ~14 KB of vendored JS that turns `hx-get` attributes into XHR-and-swap behaviour. Tab navigation becomes a partial swap; live stats become `hx-trigger="every 2s"` against a JSON-returning endpoint. No build step, no node, no SPA framework — the templates remain testable HTML strings.
- **FastAPI + React.** Modern but adds a node toolchain, a separate build, and a JSON API layer the SSR/HTMX option avoids. Over-engineered for a single-user localhost tool.

## Decision

FastAPI + HTMX + Jinja2 partials. Server-rendered, server-stateful, HTMX-progressive. No SPA, no node.

### Information architecture

- `GET /` → **Contradictions tab** (the main page). Always shows findings from `AuditLogger.most_recent_run()`.
- HTMX-driven tab strip: Contradictions / Documents / Assertions / Stats / Ingest.
- Each tab swap: `hx-get="/tabs/<name>" hx-target="#tab-content"`.
- Each finding row carries a Diff button: `hx-get="/findings/{id}/diff"` (pair) or `hx-get="/multi_party_findings/{id}/diff"` (triangle). The partial returned is a side-by-side card view (2 cards for pair, 3 for triangle) showing assertion text, parent document, NLI p_contradiction (pair only), judge rationale, and evidence spans.
- Stats tab live-updates via `hx-trigger="every 2s" hx-get="/runs/{id}/stats.json"` against the in-progress run; replaced by a final-summary card when `run_status = "done"`.

### Template namespacing

All Jinja2 templates carry a `cc_` prefix (consistency-checker) so a multi-app deployment can never collide:

- Full pages: `cc_base.html`, `cc_contradictions.html`, `cc_documents.html`, `cc_assertions.html`, `cc_stats.html`, `cc_ingest.html`.
- HTMX partials: `cc__pair_diff.html`, `cc__multi_party_diff.html`, `cc__stats_live.html`, `cc__stats_final.html`, `cc__upload_success.html` (double underscore = partial marker after the namespace prefix).
- Static assets: `cc_style.css`. The vendored `htmx.min.js` keeps its upstream name.

URLs stay clean (`/`, `/tabs/documents`) — only filesystem names are namespaced.

### Output filenames

Every artefact written through the web UI (and the CLI in parallel — see v0.3 plan §"Output naming") gets a unique descriptive filename so two runs against the same corpus never overwrite each other. The shared helper `consistency_checker/audit/naming.py` produces:

- Reports: `cc_report_{run_started_at:%Y-%m-%dT%H-%M-%S}_{run_id_short}.md` — e.g. `cc_report_2026-05-14T10-30-00_a1b2c3d4.md`.
- CSV exports: `cc_assertions_{timestamp:%Y-%m-%dT%H-%M-%S}.csv`.
- JSONL exports: `cc_assertions_{timestamp:%Y-%m-%dT%H-%M-%S}.jsonl`.
- Web-UI run downloads inherit the same names; the browser-suggested filename comes via `Content-Disposition: attachment; filename="..."`.

Default destination directory is `data_dir / "reports"` (created on demand). `--out` on the CLI stays accepted for explicit paths.

### Persistence

The web app does **not** add a new database. SQLite + FAISS stay canonical. Uploads land in `data_dir / "uploads" / <upload_id>` (timestamp + 8-char hash); no automatic GC in v0.3.

### Background tasks

`pipeline.check` becomes background-task-aware via a new `run_status` column on `pipeline_runs` (migration `0004_run_status.sql`). FastAPI `BackgroundTasks` is enough for a single-user local app; if we ever bind beyond localhost we'll switch to `arq` or RQ.

`pipeline.ingest` stays synchronous in v0.3 — real upload sizes are small enough that browser timeouts aren't a concern. Revisit if real corpora cause problems.

### Auth / threat model

No auth. Default bind is `127.0.0.1:8000` via the CLI's `consistency-check serve` command. Threat model is local-only; if anyone proposes binding `0.0.0.0` they must add auth first (this ADR is the precommitment).

## Consequences

- New `consistency_checker/web/` package (`app.py`, `templates/`, `static/`). Additive — `cli/`, `pipeline.py`, and the audit logger only grow new methods.
- Runtime deps grow: `fastapi`, `python-multipart`, `jinja2`, `uvicorn`. Dev dep adds `httpx` (FastAPI's `TestClient` already pulls it).
- One new migration (`0004_run_status.sql`). Additive; existing rows backfill with `'done'`.
- The CLI surface gains `consistency-check serve --host 127.0.0.1 --port 8000`.
- The report renderer's markdown output stays canonical. The web UI renders the same markdown to HTML via `mistune`; we don't fork a parallel HTML report.
- Out of scope: token-level diff highlighting, SSE for live stats, run-picker UI, authentication, upload GC, async ingest. All deferred to v0.4 with rationale recorded in `docs/plans/v0.3-block-g.md`.
