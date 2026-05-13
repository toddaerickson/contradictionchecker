-- Multi-document conditional contradiction findings (ADR-0006, v0.2 F1).
--
-- Sits alongside `findings` rather than overloading it: `findings` stays
-- pair-shaped (two assertion columns); multi_party_findings carries N-ary
-- assertion sets as JSON arrays.

CREATE TABLE IF NOT EXISTS multi_party_findings (
    finding_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    assertion_ids_json TEXT NOT NULL,        -- JSON array, length >= 3
    doc_ids_json TEXT NOT NULL,              -- distinct doc ids, length >= 2
    triangle_edge_scores_json TEXT,          -- JSON list of [a_id, b_id, similarity]
    judge_verdict TEXT,                      -- multi_party_contradiction | not_contradiction | uncertain
    judge_confidence REAL,
    judge_rationale TEXT,
    evidence_spans_json TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_mpf_run ON multi_party_findings(run_id);
CREATE INDEX IF NOT EXISTS idx_mpf_verdict ON multi_party_findings(judge_verdict);
