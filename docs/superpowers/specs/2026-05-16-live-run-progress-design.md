# Live Run Progress (Stats Tab Refactor) — Design Spec

**Date:** 2026-05-16
**Item:** New UX work, no prior roadmap number
**Origin:** Operator pain — uploaded ~2,200 assertions, sat blind for ~20 min with no progress feedback; existing Stats tab counters only update at run-end.

## Goal

Replace the current "left-to-right tile" stats view with a top-to-bottom **table of stages** that updates live every 2 s during both **ingest** and **check**. Each row shows the stage name, its status (waiting / running / done / skipped / failed), and an actual progress number — not just a "running" flag.

## Why the current view fails

1. The polled counters (`n_assertions`, `n_pairs_gated`, `n_pairs_judged`, `n_findings`) are written only **once** — at the end of `pipeline.check()`. Mid-run, they're zero. The "live" view is barely live.
2. Ingest happens **synchronously** inside `POST /uploads`. There's no run id, no counters, no progress events. The handler returns only when the LLM extraction finishes — minutes later, with no feedback in between.
3. Failures inside extraction (e.g., the 2026-05-16 11:06 Pydantic-validation incident where the LLM returned `definitions` as a JSON string) disappear silently. A user has no way to know a chunk failed.

## Out of scope

- Historical run browser ("show me runs from yesterday"). One-run-at-a-time view; the most-recent run wins.
- Retrying individual failed chunks from the UI.
- Multi-tenant concurrency. Solo-operator assumption holds.
- Cancelling an in-flight run from the UI.
- Streaming logs to the browser. The table summarises; the log file is the source of detail.

## Design

### Data model: extend `runs` table

Add three columns to `runs` (migration `0010_run_stages.sql`):

```sql
ALTER TABLE runs ADD COLUMN run_kind TEXT NOT NULL DEFAULT 'check'
  CHECK (run_kind IN ('ingest', 'check'));
ALTER TABLE runs ADD COLUMN stages_json TEXT;  -- per-stage progress, serialised
ALTER TABLE runs ADD COLUMN current_stage TEXT;
```

`stages_json` holds a list of `{name, status, done, total, detail}` entries. Stored as JSON because the shape is short, write-mostly, read-as-a-blob, and we don't query inside it. SQLite's JSON1 functions are available if we ever need to, but we won't on day one.

### Stage catalog

**Ingest run** (one row per file uploaded; aggregated when N > 10):

| name | done semantics | total semantics |
|------|----------------|------------------|
| `files_received` | files saved to disk | files submitted |
| `bytes_processed` | bytes read | total bytes of all files |
| `chunks_extracted` | chunks where extraction finished (success OR failure) | total chunks |
| `assertions_extracted` | successfully extracted assertions written to store | — (running total only) |
| `embeddings_indexed` | vectors written to FAISS | total assertions to embed |
| `extraction_failures` | failed chunks count | — (only appears as a row if `> 0`, with link to log) |

**Check run** (one logical run):

| name | done semantics | total semantics |
|------|----------------|------------------|
| `gate_candidates` | candidate pairs found | — (running total only; emitted at end of gate) |
| `nli_screen` | pairs scored | total candidate pairs |
| `statement_judge` | LLM calls completed | pairs that survived NLI |
| `definition_checker` | definition-pair LLM calls completed | total definition pairs |
| `cross_document` | triangles enumerated | total triangles (only if `deep=True`) |

### Backgrounding ingest

`POST /uploads` becomes asynchronous:

1. Save uploaded files to `data/uploads/<upload_id>/` synchronously (fast, deterministic).
2. Insert a `runs` row with `run_kind='ingest'`, `run_status='pending'`, `stages_json=<initial stage list>`, return immediately with a 202 + `HX-Redirect: /tabs/stats?run_id=<id>`.
3. Background task runs `pipeline.ingest()` with a new `progress_callback` parameter that the pipeline invokes at each stage transition and (for long stages) every N chunks.
4. Callback writes `stages_json` + `current_stage` via `AuditLogger.update_run_stages()`.
5. On completion, `run_status='done'` (or `'failed'` with `error_message` set if any stage raised).

Same shape as today's `POST /runs` for check, so the two paths converge.

### Pipeline instrumentation

`pipeline.ingest()` and `pipeline.check()` accept an optional `progress: ProgressReporter | None = None`. Where there's a tight loop (NLI scoring, judge calls, chunk extraction), invoke `progress.advance(stage, done, total)` either every chunk (small loops) or every 25 (judge loop). The reporter is responsible for throttling DB writes to ≤ once per 2 s so we don't thrash SQLite during a 6,000-pair NLI run.

```python
class ProgressReporter(Protocol):
    def advance(self, stage: str, done: int, total: int | None = None) -> None: ...
    def stage_status(self, stage: str, status: Literal["running", "done", "skipped", "failed"]) -> None: ...
```

For library callers (CLI, tests) the default is a no-op reporter — the pipeline keeps working with no DB writes.

For the web layer, a `DbProgressReporter` wraps `AuditLogger` and a 2 s throttle.

### UI

Replace `cc__stats_live.html` and `cc__stats_final.html` with a single table partial `cc__stats_stages.html`. The "done" and "running" states share the layout; only the polling and the status pills differ.

```
┌────────────────────────────┬─────────┬────────────────────────────────┐
│ Stage                      │ Status  │ Progress                       │
├────────────────────────────┼─────────┼────────────────────────────────┤
│ Files received             │ done    │ 3 of 3                         │
│ Bytes processed            │ done    │ 4.2 MB of 4.2 MB               │
│ Atomic-fact extraction     │ running │ 1,847 of ~2,200 assertions     │
│ Embedding into FAISS       │ waiting │ —                              │
├────────────────────────────┼─────────┼────────────────────────────────┤
│ Candidate gate (FAISS)     │ waiting │ —                              │
│ NLI screen (DeBERTa)       │ waiting │ —                              │
│ Statement judge (LLM)      │ waiting │ —                              │
│ Definition checker (LLM)   │ waiting │ —                              │
│ Cross-document triangles   │ skipped │ deep=false                     │
└────────────────────────────┴─────────┴────────────────────────────────┘
```

- Status pill colors: waiting=muted, running=blue + spinner, done=green check, skipped=gray, failed=red.
- Polling: `hx-get="/runs/{run_id}/stats" hx-trigger="every 2s"` while any row is `running` or `pending`. When all rows are terminal (`done`/`skipped`/`failed`), drop the trigger.
- Mid-run, the section header reads "Run in progress" with a spinner; on completion it reads "Run complete" or "Run failed" with the elapsed time. Below the table: a CTA — "View contradictions →" when done.
- For multi-file uploads, the ingest stages aggregate. A small "3 files" badge in the section header.

### Run trigger flow

Keep the existing two-button shape:

- **Upload form** (Ingest tab) → `POST /uploads` → backgrounds ingest → redirects to Stats tab → table shows ingest stages live → on completion, table includes a `Check for contradictions →` CTA (with the same `hx-confirm` API-spend gate as today).
- **Check button** (Contradictions tab, or post-upload CTA) → `POST /runs` → backgrounds check → same Stats tab → table now shows check stages live below the (still-visible) "Ingest" header section that completed earlier.

This preserves the API-spend confirmation gate. Two runs back-to-back appear as two table sections in the Stats tab. After 24 h, the older one is replaced by whatever's most recent.

### Concurrency

While any run (ingest or check) is in `pending` or `running` state, disable the upload form and the "Check for contradictions" button. Display a small banner: "A run is in progress — please wait." This is enforced server-side: the trigger endpoints return `409 Conflict` if any incomplete run exists, and client-side they grey out via an `hx-get="/runs/active"` poll on the relevant pages.

## Open decisions to confirm

1. **Per-file vs aggregate during multi-file uploads.** Default: aggregate (one row each for "Bytes / Chunks / Extracted"). Per-file rows only if you specifically want them — they get long fast on 10+ file batches.
2. **Surface extraction failures inline?** Default: yes — a `extraction_failures` row appears only when `> 0`, with the count and a "see log" link. The actual chunk content goes to `data/logs/execution.log` only (not the DB).
3. **Throttle: 2 s DB write floor.** This means a 6,000-pair NLI run writes the DB ~30-60 times total instead of 6,000. Acceptable trade-off — display still updates within 2 s of any change.
4. **Cancel button.** Not in this spec. If you want it later, it's a separate small addition (set `run_status='cancelled'`, pipeline checks the flag in its loops).

## Test plan

- Migration round-trip: `pytest tests/test_migrations.py` — new column non-destructive, default values applied.
- `ProgressReporter` Protocol satisfies a no-op implementation and a DB-backed implementation. Unit tests for both.
- `DbProgressReporter` throttles writes correctly (asserts ≤ 1 write per 2 s under a tight loop).
- `pipeline.ingest(progress=reporter)` and `pipeline.check(progress=reporter)` call the reporter at expected stage transitions. Use `FixtureExtractor` / `FixtureNliChecker` / `FixtureJudge` to keep hermetic.
- Web layer: backgrounded `POST /uploads` returns 202 + `HX-Redirect`. `GET /runs/{id}/stats` renders the table for each `run_kind`.
- E2E: a fixture-mode end-to-end test that uploads a small payload, polls the stats endpoint, and asserts the table eventually shows all-done.
- Concurrency: second `POST /runs` while one is running returns 409.

## Build phasing

Two PRs:

**PR 1 — Instrumentation + table layout (no ingest backgrounding yet).**

- Migration 0010.
- `ProgressReporter` Protocol, no-op + DB implementations, throttle test.
- Instrument `pipeline.check()` (gate, NLI, statement judge, definition checker, cross-doc).
- Replace `cc__stats_live.html`, `cc__stats_final.html` with `cc__stats_stages.html`. Wire `_live_counters` → stage list.
- Concurrency guard: disable check button while a run is active.
- Ingest rows show post-hoc done state (sourced from store stats), no live updates yet.

PR 1 gets shipped and tested before PR 2 starts.

**PR 2 — Backgrounded ingest.**

- Background task wiring for `POST /uploads`.
- Instrument `pipeline.ingest()` with progress callbacks (files / bytes / chunks / extracted / embeddings).
- `extraction_failures` row appears when failures occur.
- E2E test for ingest progress polling.

## Effort estimate

- PR 1: ~3 hours of focused coding + tests
- PR 2: ~3-4 hours of focused coding + tests
- Total: ~6-7 hours
