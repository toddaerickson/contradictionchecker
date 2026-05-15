-- Migration 0006: add error_message column to pipeline_runs for U2 (surface
-- failure reason in Stats tab instead of a silent "Run failed" message).
ALTER TABLE pipeline_runs ADD COLUMN error_message TEXT;
