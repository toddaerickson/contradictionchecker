-- Adds the detector_type discriminator to findings so the same table can hold
-- pair-shaped findings produced by multiple detectors (contradiction, then
-- definition_inconsistency in this build, then others later).
ALTER TABLE findings ADD COLUMN detector_type TEXT NOT NULL DEFAULT 'contradiction';
CREATE INDEX IF NOT EXISTS idx_findings_detector ON findings(detector_type);
