-- Remove the LLM judge's self-reported confidence score.
--
-- judge_confidence was the model's subjective certainty token (see the judge
-- prompts), not a calibrated probability. It was surfaced as a 0-1 figure that
-- implied a precision it did not have, so it is removed from the product
-- entirely. The calibration tooling that consumed it is removed alongside.
--
-- SQLite >= 3.35 supports ALTER TABLE ... DROP COLUMN. The column is not
-- referenced by any index, view, or trigger, so the drop is a clean rewrite.
ALTER TABLE findings DROP COLUMN judge_confidence;
ALTER TABLE multi_party_findings DROP COLUMN judge_confidence;
