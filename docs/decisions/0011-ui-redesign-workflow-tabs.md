# ADR-0011: Workflow-Ordered Tab Navigation and Persistent Corpora

**Status:** Accepted  
**Date:** 2026-05-17  
**Relates to:** ADR-0007 (Web UI)

---

## Context

The original six-tab UI (Contradictions, Definitions, Documents, Statements, Stats, Ingest) presented tabs in no meaningful order. Users had no visual cue for which tab to use first, and the Ingest tab appeared last despite being the entry point. Documents and Statements tabs showed raw data without context for what to do with it.

Two structural problems drove the redesign:

1. **Navigation order**: Tabs were ordered by feature area, not by workflow. New users had to read documentation to understand the intended sequence.
2. **Corpus ephemerality**: Each pipeline run operated on a flat uploads directory. There was no way to name a collection of documents, add to it over time, or distinguish between runs on different document sets.

---

## Decision

Reorganize the UI into seven workflow-ordered tabs that follow the natural user progression, and introduce persistent named corpora backed by both the filesystem and the database.

### Tab Order (left to right)

| # | Tab | Purpose |
|---|-----|---------|
| 01 | Ingest | Select or create corpus, choose judge provider, upload files |
| 02 | Process | Real-time streaming log of the analysis pipeline |
| 03 | Assertions | Read-only exploration of extracted assertions |
| 04 | Contradictions | Read-only exploration of detected contradiction pairs |
| 05 | Definitions | Read-only exploration of definition inconsistencies |
| 06 | Action Items | Unified verdict management — the single source of truth |
| 07 | Stats | Corpus health summary and run metrics |

### Persistent Corpora

A corpus is a named, filesystem-backed collection of documents stored under `data/corpora/{corpus_id}/documents/`. Corpora are permanent — adding new files to an existing corpus allows incremental analysis without losing prior verdicts.

The database tracks corpora and runs in two new tables (`corpora`, `runs`) added in migration 0011. Each run links back to its corpus and stores a newline-delimited JSON message log that the Process tab streams via SSE.

### Action Items as Single Source of Truth

User verdicts (Confirmed / False Positive / Dismissed / Pending) are stored on `findings` and `multi_party_findings` rows via the `user_verdict` column. The Action Items tab is the only place where verdicts can be set — exploration tabs (Contradictions, Definitions) are read-only and link users to Action Items to take action.

---

## Backend Changes

- `POST /api/corpora` — create corpus (generates id slug, creates filesystem directory)
- `GET /api/corpora` — list all corpora
- `GET /api/corpora/{corpus_id}` — get corpus with document count
- `GET /api/corpora/{corpus_id}/documents` — list documents
- `POST /api/corpora/{corpus_id}/runs` — start a run
- `GET /api/runs/{run_id}/progress` — SSE stream of pipeline messages
- `POST /api/runs/{run_id}/messages` — append message (used by pipeline)
- `PATCH /api/runs/{run_id}` — update run status
- `POST /api/findings/{finding_id}/verdict` — set user verdict (pairwise or multi-party)

---

## Aesthetic Direction

The redesign adopts a **Precision Instrument** aesthetic:
- IBM Plex Mono (headers, numbers, tabs) + IBM Plex Sans (body)
- Near-black header (#111111) with numbered, monospace tab labels
- Orange-red accent (#c9450a) for contradiction findings (semantically meaningful)
- Warm off-white body (#f4f3f0) with white panels
- Terminal-style dark log view for the Process tab

---

## Consequences

**Positive:**
- New users immediately understand the workflow sequence from the tab order
- Named corpora allow incremental analysis on growing document sets
- Action Items tab eliminates scattered verdict controls
- SSE streaming gives users visibility into long-running pipeline operations

**Negative:**
- The `/uploads` endpoint (existing) remains for backward compatibility but is now subordinate to the corpus-based flow; a future migration should unify these
- The Action Items tab currently renders an empty state (findings query not yet wired to the new corpus model); this is a known gap for the next sprint
- Document count in corpus detail endpoint does two filesystem `stat()` calls per file on large corpora (tracked for optimization)

---

## References

- Spec: `docs/superpowers/specs/2026-05-16-ui-redesign.md`
- Migration: `consistency_checker/index/migrations/0011_ui_redesign_schema.sql`
- Data models: `consistency_checker/models/ui.py`
- API: `consistency_checker/web/api/corpora.py`, `consistency_checker/web/api/runs.py`
