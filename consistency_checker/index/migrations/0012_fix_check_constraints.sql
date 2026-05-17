-- Migration 0012: Fix broken CHECK constraints from 0011
--
-- In SQLite, `val IN ('a', 'b', NULL)` evaluates to NULL (not FALSE) when val
-- is not in the list, so the CHECK is silently skipped for any value.
-- Fix strategy:
--   * corpora/runs: drop and recreate (no production data).
--   * findings/multi_party_findings: add BEFORE INSERT/UPDATE triggers because
--     SQLite cannot ALTER an existing CHECK constraint.

DROP TABLE IF EXISTS runs;
DROP TABLE IF EXISTS corpora;

CREATE TABLE corpora (
  corpus_id   TEXT PRIMARY KEY,
  corpus_name TEXT NOT NULL UNIQUE,
  corpus_path TEXT NOT NULL,
  judge_provider TEXT DEFAULT 'moonshot'
    CHECK (judge_provider IN ('moonshot', 'anthropic')),
  created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE runs (
  run_id       TEXT PRIMARY KEY,
  corpus_id    TEXT NOT NULL REFERENCES corpora(corpus_id) ON DELETE CASCADE,
  started_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMP DEFAULT NULL,
  status       TEXT DEFAULT 'in_progress'
    CHECK (status IN ('in_progress', 'completed', 'failed')),
  message_log  TEXT DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_corpus_id ON runs(corpus_id);

-- Fix user_verdict CHECK on findings via triggers (cannot ALTER CHECK in SQLite)
CREATE TRIGGER IF NOT EXISTS trg_findings_verdict_insert
BEFORE INSERT ON findings
WHEN NEW.user_verdict IS NOT NULL
BEGIN
  SELECT RAISE(ABORT, 'user_verdict must be confirmed, false_positive, dismissed, or pending')
  WHERE NEW.user_verdict NOT IN ('confirmed', 'false_positive', 'dismissed', 'pending');
END;

CREATE TRIGGER IF NOT EXISTS trg_findings_verdict_update
BEFORE UPDATE OF user_verdict ON findings
WHEN NEW.user_verdict IS NOT NULL
BEGIN
  SELECT RAISE(ABORT, 'user_verdict must be confirmed, false_positive, dismissed, or pending')
  WHERE NEW.user_verdict NOT IN ('confirmed', 'false_positive', 'dismissed', 'pending');
END;

CREATE TRIGGER IF NOT EXISTS trg_mp_findings_verdict_insert
BEFORE INSERT ON multi_party_findings
WHEN NEW.user_verdict IS NOT NULL
BEGIN
  SELECT RAISE(ABORT, 'user_verdict must be confirmed, false_positive, dismissed, or pending')
  WHERE NEW.user_verdict NOT IN ('confirmed', 'false_positive', 'dismissed', 'pending');
END;

CREATE TRIGGER IF NOT EXISTS trg_mp_findings_verdict_update
BEFORE UPDATE OF user_verdict ON multi_party_findings
WHEN NEW.user_verdict IS NOT NULL
BEGIN
  SELECT RAISE(ABORT, 'user_verdict must be confirmed, false_positive, dismissed, or pending')
  WHERE NEW.user_verdict NOT IN ('confirmed', 'false_positive', 'dismissed', 'pending');
END;
