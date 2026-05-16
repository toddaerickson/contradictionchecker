-- Add user_verdict column to findings table
ALTER TABLE findings ADD COLUMN user_verdict TEXT DEFAULT NULL;
  -- Values: 'confirmed', 'false_positive', 'dismissed', 'pending', NULL

-- Add user_verdict column to multi_party_findings table
ALTER TABLE multi_party_findings ADD COLUMN user_verdict TEXT DEFAULT NULL;

-- Create corpora table: persistent document collections
CREATE TABLE IF NOT EXISTS corpora (
  corpus_id TEXT PRIMARY KEY,  -- e.g., 'financial-audit-q1'
  corpus_name TEXT NOT NULL UNIQUE,
  corpus_path TEXT NOT NULL,  -- e.g., 'data/corpora/financial-audit-q1'
  judge_provider TEXT DEFAULT 'moonshot',  -- 'moonshot' or 'anthropic'
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create runs table: one per processing run on a corpus
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,  -- e.g., 'run_20260516_140530'
  corpus_id TEXT NOT NULL REFERENCES corpora(corpus_id),
  started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMP DEFAULT NULL,
  status TEXT DEFAULT 'in_progress',  -- 'in_progress', 'completed', 'failed'
  message_log TEXT DEFAULT NULL  -- JSON array of progress messages
);

-- Create indices
CREATE INDEX IF NOT EXISTS idx_runs_corpus_id ON runs(corpus_id);
CREATE INDEX IF NOT EXISTS idx_findings_user_verdict ON findings(user_verdict);
CREATE INDEX IF NOT EXISTS idx_multi_party_findings_user_verdict ON multi_party_findings(user_verdict);
