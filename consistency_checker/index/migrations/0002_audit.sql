-- Audit tables: pipeline_runs records one row per scan; findings records every
-- judge verdict (contradiction, not_contradiction, uncertain) so a run is fully
-- reproducible from logged inputs alone.

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY,
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMP,
    config_json TEXT,
    n_assertions INTEGER NOT NULL DEFAULT 0,
    n_pairs_gated INTEGER NOT NULL DEFAULT 0,
    n_pairs_judged INTEGER NOT NULL DEFAULT 0,
    n_findings INTEGER NOT NULL DEFAULT 0,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS findings (
    finding_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    assertion_a_id TEXT NOT NULL REFERENCES assertions(assertion_id),
    assertion_b_id TEXT NOT NULL REFERENCES assertions(assertion_id),
    gate_score REAL,
    nli_label TEXT,
    nli_p_contradiction REAL,
    nli_p_entailment REAL,
    nli_p_neutral REAL,
    judge_verdict TEXT,
    judge_confidence REAL,
    judge_rationale TEXT,
    evidence_spans_json TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_findings_run ON findings(run_id);
CREATE INDEX IF NOT EXISTS idx_findings_verdict ON findings(judge_verdict);
CREATE INDEX IF NOT EXISTS idx_findings_pair ON findings(assertion_a_id, assertion_b_id);
