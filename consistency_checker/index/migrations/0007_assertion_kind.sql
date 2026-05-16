-- Adds the `kind` discriminator to assertions and the two definition-only
-- columns. Backwards-compatible: existing rows default to `kind='claim'` so
-- the contradiction detector continues to see them unchanged.
ALTER TABLE assertions ADD COLUMN kind TEXT NOT NULL DEFAULT 'claim';
ALTER TABLE assertions ADD COLUMN term TEXT;
ALTER TABLE assertions ADD COLUMN definition_text TEXT;
CREATE INDEX IF NOT EXISTS idx_assertions_kind ON assertions(kind);
CREATE INDEX IF NOT EXISTS idx_assertions_term ON assertions(term) WHERE kind = 'definition';
