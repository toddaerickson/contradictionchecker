# ADR 0018 — Remove the LLM judge confidence score and calibration tooling

**Status**: Accepted

## Context

Stage B asked the LLM judge to return a `confidence` float in `[0.0, 1.0]`
alongside its verdict. The judge prompt defined it as "your subjective
certainty in the verdict" — i.e. a self-reported token, not a probability with
any measured relationship to correctness. This figure was:

- persisted to `findings.judge_confidence` / `multi_party_findings.judge_confidence`,
- surfaced in the web findings card ("Confidence 0.81"), the definitions
  drawer, the findings CSV export, and the markdown report,
- used as the default sort key for findings (descending), and
- the entire input to the `eval` command's calibration table and the
  `compute_calibration` / `CalibrationBin` machinery in `audit/eval.py`.

The number created a false pretense of precision. Two decimal places imply 1%
resolution that a subjective self-rating does not carry, and a colour/severity
treatment on top of it would have implied a calibrated risk score the system
cannot produce. The existence of the calibration tooling was itself the tell:
it existed to check whether stated confidence tracked reviewer-confirmed
precision — confirming the team already knew the raw number was not a
probability. Until a finding's confidence is calibrated against labels, the
figure is noise dressed as signal, and showing it makes the product less
trustworthy, not more.

There is no honest per-finding accuracy number available without labels. The
only defensible accuracy measurement is system-level: run the detector against
a fixed labelled benchmark and publish precision/recall (tracked separately;
see `docs/benchmarks.md`). The reviewer-verdict **precision** signal
(`confirmed / (confirmed + false_positive)`) is also honest — it is computed
from human labels, not from the model's self-report — and is kept.

## Decision

Remove the judge confidence score and the calibration subsystem entirely.

- `confidence` is dropped from `JudgePayload`, `MultiPartyJudgePayload`,
  `DefinitionJudgePayload`, and from the `JudgeVerdict`,
  `MultiPartyJudgeVerdict`, `DefinitionJudgeVerdict` dataclasses. The judge
  prompts no longer ask for it.
- `judge_confidence` is dropped from the `Finding` / `MultiPartyFinding`
  dataclasses and from the database. Migration
  `0015_drop_judge_confidence.sql` drops the columns from `findings` and
  `multi_party_findings` (SQLite ≥ 3.35 `ALTER TABLE ... DROP COLUMN`). The
  original `0002` / `0003` migration files are left untouched per the
  append-only migration rule; the column simply no longer exists after `0015`.
- `audit/eval.py` loses `compute_calibration`, `CalibrationBin`,
  `format_calibration_table`, `write_calibration_csv`, and
  `DEFAULT_CALIBRATION_BINS`. The reviewer-verdict precision path
  (`compute_detector_precision`, `format_precision_table`, `write_precision_csv`)
  is kept — it rests on human labels, not a model self-score.
- CLI: `report` loses `--min-confidence`; `eval` loses `--detector` and no
  longer prints or writes a calibration table (precision only).
- Findings are no longer sorted by confidence. The report orders by
  `finding_id` within document-pair groups; the web findings list orders by
  `(doc_a, doc_b, finding_id)`. Order is stable and score-free.
- The web findings card, definitions drawer, and findings CSV no longer show a
  confidence column/field. The NLI contradiction probability
  (`nli_p_contradiction`) is a separate classifier output and is retained in
  the audit trail and report — it is not the LLM self-score this ADR removes.

Historical ADRs and dated specs/plans that mention confidence are left as-is;
they are a record of past decisions.

## Consequences

- The product no longer presents any per-finding numeric score. A finding is
  surfaced with its verdict, rationale, and verbatim evidence spans for a human
  to verify — which is what the system can honestly support.
- Existing databases lose the `judge_confidence` column on next `migrate()`.
  The stored values were self-reported and uncalibrated; nothing downstream
  depends on them after this change.
- The `eval` command still answers "how often were confirmed findings real?"
  from reviewer verdicts. The calibration question ("does stated confidence
  track correctness?") is moot once confidence is gone.
- Re-introducing a per-finding score later should be calibration-gated against
  labels (benchmark or accumulated reviewer verdicts), not a model self-rating.
  This ADR deliberately removes the self-rating rather than restyling it.
