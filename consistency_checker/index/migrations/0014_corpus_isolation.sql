-- Logical corpus isolation: every document and run belongs to one corpus.
-- corpora.corpus_id is TEXT (UUID), so the new FK columns mirror that type.
-- runs (web table) already has corpus_id from migration 0011/0012; no change needed there.

ALTER TABLE documents     ADD COLUMN corpus_id TEXT REFERENCES corpora(corpus_id) ON DELETE CASCADE;
ALTER TABLE pipeline_runs ADD COLUMN corpus_id TEXT REFERENCES corpora(corpus_id) ON DELETE CASCADE;

-- Auto-create "legacy" corpus when pre-isolation rows exist.
INSERT INTO corpora (corpus_id, corpus_name, corpus_path, judge_provider, created_at, updated_at)
SELECT lower(hex(randomblob(16))), 'legacy', '(pre-isolation)', 'moonshot',
       datetime('now'), datetime('now')
WHERE EXISTS (
    SELECT 1 FROM documents     WHERE corpus_id IS NULL
    UNION ALL
    SELECT 1 FROM pipeline_runs WHERE corpus_id IS NULL
)
AND NOT EXISTS (SELECT 1 FROM corpora WHERE corpus_name='legacy');

UPDATE documents
SET corpus_id = (SELECT corpus_id FROM corpora WHERE corpus_name='legacy')
WHERE corpus_id IS NULL;

UPDATE pipeline_runs
SET corpus_id = (SELECT corpus_id FROM corpora WHERE corpus_name='legacy')
WHERE corpus_id IS NULL;
