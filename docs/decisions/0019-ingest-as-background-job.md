# ADR 0019 — Corpus ingestion runs as a background job reusing the check-run progress channel

**Status**: Accepted

## Context

Corpus ingestion (`POST /corpora/new`) ran the loader → chunker → extractor → embedder pipeline **synchronously inside the `async def` request handler**. Because FastAPI runs `async def` endpoints on the event loop, that blocking work starved every other request: while a large or scanned upload ingested, the whole single-page UI froze — the sidebar, other corpora, and the **Run check** button all stopped responding. An operator uploading a scanned PDF (empty fast-text → ~100 s OCR attempt, then a Tesseract-not-installed failure) saw a dead modal and concluded the app was broken.

The check run had already solved this exact problem the right way (ADR-0017 Phase 3): `POST /corpora/{id}/run` returns instantly, queues a `BackgroundTasks` worker that opens its own thread-local store, and streams `pending → running → done/failed` over a per-corpus SSE channel that the sidebar renders as a live progress bar. Ingestion was the one heavy operation **not** using that machinery.

## Decision

Model ingestion as a first-class **background job that reuses the check-run infrastructure** rather than a parallel system.

- **Schema (migration 0016):** add `run_kind TEXT NOT NULL DEFAULT 'check'` plus `n_files_total` / `n_files_done` to `pipeline_runs`. One table holds both job kinds, so the existing `GET /corpora/{id}/progress` SSE endpoint and the sidebar progress element drive ingest progress with no new plumbing. All columns are additive with safe defaults; existing rows read back as check runs.
- **Handler:** `POST /corpora/new` saves the uploads (fast I/O, stays on the loop), creates the corpus row, `begin_run(run_kind="ingest", n_files_total=…)`, queues `_ingest_in_background`, and returns the modal **immediately** in an "Ingest started — watch the sidebar" state.
- **Worker:** `_ingest_in_background` mirrors `_run_check_in_background` — own thread-local stores, marks `running`, ingests file-by-file bumping `n_files_done` / `n_assertions` (so the bar advances live), then finalises `done` or `failed` with a user-facing `error_message`.
- **Failure keeps the corpus.** A failed ingest no longer rolls the corpus away; it stays with a visible `failed` run so the user can see *why*. This is only safe because it ships **with** a **Delete corpus** control (`POST /corpora/{id}/delete` → `store.delete_corpus`, surfaced as a confirm-guarded button in the findings header): keeping a failed corpus without a delete path would re-open the orphan-corpus 409-on-retry trap that commit `6d00250` closed, since the duplicate-name guard rejects re-creating the same name. Keep-on-failure and Delete are coupled and ship together.
- **Empty-text surfacing:** files that load with no extractable text (scanned images when OCR is unavailable) are recorded in the run `notes` JSON and shown in the finished progress label — never a silent 0-assertion "success".

### Why reuse `pipeline_runs` instead of a new `ingest_jobs` table

A dedicated table would have duplicated the SSE endpoint, the sidebar element, and the "which job is active" selection logic, and forced a merge of two progress sources. Reusing `pipeline_runs` via `run_kind` overloads three numeric columns but inherits the entire progress/sidebar/double-submit-guard surface unchanged — strictly less net code and one place to reason about run lifecycle. The per-corpus double-submit guard (`run_status IN ('pending','running')`) is kind-agnostic, so it already blocks a check run while an ingest is in flight (they share SQLite + FAISS).

## Consequences

- Uploads never freeze the UI; the sidebar shows a live `files done / total` bar.
- A crashed worker can leave a run stuck `running` (no cancel/clear yet) — the same pre-existing limitation as check runs; a "clear stuck run" control is queued with the CRUD essentials.
- The modal can no longer report a final assertion count (ingest hasn't run yet); it reports files queued and points at the sidebar.
