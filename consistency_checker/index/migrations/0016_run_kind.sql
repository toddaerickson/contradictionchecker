-- Migration 0016: model corpus ingestion as a first-class background job in
-- pipeline_runs (ADR-0019). `run_kind` distinguishes a check run ('check', the
-- historical default) from an ingest job ('ingest'); the two file counters let
-- the existing per-corpus SSE progress stream render a live "files done / total"
-- bar for ingest without a parallel table. All columns are additive with safe
-- defaults so existing rows read back as check runs.
ALTER TABLE pipeline_runs ADD COLUMN run_kind TEXT NOT NULL DEFAULT 'check';
ALTER TABLE pipeline_runs ADD COLUMN n_files_total INTEGER NOT NULL DEFAULT 0;
ALTER TABLE pipeline_runs ADD COLUMN n_files_done INTEGER NOT NULL DEFAULT 0;
