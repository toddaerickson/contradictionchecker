-- Add user_verdict column to findings table with enum enforcement
-- Values: 'confirmed', 'false_positive', 'dismissed', 'pending', NULL
-- Pattern: established in migration 0009 for verdict tracking
ALTER TABLE findings ADD COLUMN user_verdict TEXT DEFAULT NULL
  CHECK (user_verdict IN ('confirmed', 'false_positive', 'dismissed', 'pending', NULL));

-- Add user_verdict column to multi_party_findings table with same enum
ALTER TABLE multi_party_findings ADD COLUMN user_verdict TEXT DEFAULT NULL
  CHECK (user_verdict IN ('confirmed', 'false_positive', 'dismissed', 'pending', NULL));

-- Create corpora table: persistent document collections for the UI redesign
-- Each corpus groups documents that users want to analyze together
-- Users can add new files to an existing corpus over time
CREATE TABLE IF NOT EXISTS corpora (
  corpus_id TEXT PRIMARY KEY,  -- e.g., 'financial-audit-q1'
  corpus_name TEXT NOT NULL UNIQUE,
  corpus_path TEXT NOT NULL,  -- e.g., 'data/corpora/financial-audit-q1'
  judge_provider TEXT DEFAULT 'moonshot'
    CHECK (judge_provider IN ('moonshot', 'anthropic')),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create runs table: one processing run per corpus
-- Links to corpora; cascade-deletes if corpus is removed (it's a derived run record)
-- message_log: JSON array of progress messages, e.g., ["Parsing doc 1/10", "Extracted 42 assertions"]
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,  -- e.g., 'run_20260516_140530'
  corpus_id TEXT NOT NULL REFERENCES corpora(corpus_id) ON DELETE CASCADE,
  started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMP DEFAULT NULL,
  status TEXT DEFAULT 'in_progress',  -- 'in_progress', 'completed', 'failed'
  message_log TEXT DEFAULT NULL  -- JSON array of progress messages
);

-- Create indices for fast filtering in web UI
CREATE INDEX IF NOT EXISTS idx_runs_corpus_id ON runs(corpus_id);
CREATE INDEX IF NOT EXISTS idx_findings_user_verdict ON findings(user_verdict);
CREATE INDEX IF NOT EXISTS idx_multi_party_findings_user_verdict ON multi_party_findings(user_verdict);
