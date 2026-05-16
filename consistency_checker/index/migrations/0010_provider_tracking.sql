-- Add provider column to findings and multi_party_findings tables
-- to track which judge provider (anthropic/openai/moonshot) generated each verdict.

ALTER TABLE findings ADD COLUMN provider TEXT DEFAULT 'anthropic';
ALTER TABLE multi_party_findings ADD COLUMN provider TEXT DEFAULT 'anthropic';

-- Create index for provider filtering
CREATE INDEX IF NOT EXISTS idx_findings_provider ON findings(provider);
CREATE INDEX IF NOT EXISTS idx_multi_party_findings_provider ON multi_party_findings(provider);
