-- v0.3 Block G step G5: per-run lifecycle status column.
--
-- pipeline.check now transitions a run through these states:
--   pending  → row created by POST /runs, not yet started
--   running  → background task has started; live stats panel polls
--   done     → finished_at is set; final stats card is rendered
--   failed   → unhandled exception in background task; surfaced in UI
--
-- Backfill rule: every row that already has finished_at gets 'done';
-- everything else gets 'pending'. The web UI tolerates either value.

ALTER TABLE pipeline_runs ADD COLUMN run_status TEXT NOT NULL DEFAULT 'pending';

UPDATE pipeline_runs SET run_status = 'done' WHERE finished_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status ON pipeline_runs(run_status);
