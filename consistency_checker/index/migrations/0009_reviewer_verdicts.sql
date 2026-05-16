-- 0009_reviewer_verdicts.sql
--
-- Verdicts are content-addressed (not run-scoped) so re-checking the same
-- corpus does not wipe review work. The pair_key construction MUST match the
-- Python builder in `consistency_checker.audit.reviewer.build_pair_key`:
--   pair findings:     min(a_id, b_id) || ':' || max(a_id, b_id)
--   triangle findings: ':'.join(sorted(assertion_ids))
-- If this formula diverges, render-time joins go silently empty.

CREATE TABLE reviewer_verdicts (
  pair_key TEXT NOT NULL,
  detector_type TEXT NOT NULL
    CHECK (detector_type IN ('contradiction', 'definition_inconsistency', 'multi_party')),
  verdict TEXT NOT NULL
    CHECK (verdict IN ('confirmed', 'false_positive', 'dismissed')),
  set_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  note TEXT,
  PRIMARY KEY (pair_key, detector_type)
);
