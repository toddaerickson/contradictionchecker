-- Document-level org grouping + suppressed-pair audit trail.
-- Additive; nullable columns and DEFAULT 0 keep existing rows valid.
ALTER TABLE documents ADD COLUMN org_label TEXT;
ALTER TABLE documents ADD COLUMN org_reason TEXT;
ALTER TABLE findings  ADD COLUMN suppressed INTEGER NOT NULL DEFAULT 0;
