-- Per-run lifecycle status: pending | running | done | failed.
-- Backfill: pre-existing rows with finished_at set are marked 'done'.

ALTER TABLE pipeline_runs ADD COLUMN run_status TEXT NOT NULL DEFAULT 'pending';

UPDATE pipeline_runs SET run_status = 'done' WHERE finished_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status ON pipeline_runs(run_status);
