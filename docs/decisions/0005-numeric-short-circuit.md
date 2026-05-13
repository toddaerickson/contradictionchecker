# ADR 0005 — Numeric short-circuit before the LLM judge

**Status**: Accepted

## Context

v0.2 Block E adds a numeric / quantitative extractor (Step E1). When two candidate assertions share the same metric and scope but disagree in sign (e.g. "Revenue grew 12% in fiscal 2025" vs. "Revenue declined 5% in fiscal 2025"), the answer is deterministic — no LLM judgement is needed.

Two designs were considered:

- **A: Pre-Stage-B short-circuit.** Run the numeric extractor on both pairs before calling the judge. If both yield same-metric/same-scope/same-unit tuples with opposite polarity, emit a contradiction finding directly with `confidence=1.0` and skip the LLM call. Introduces a new verdict label (`numeric_short_circuit`) that the audit logger, report renderer, and report-filtering logic learn about.
- **B: Side-channel to the judge.** Always run the judge, but pass the extracted numeric tuples as additional structured context in the prompt. No new verdict label; the judge owns the final call. Wastes API budget on cases the regex resolves deterministically and grows the prompt without an obvious off-ramp.

## Decision

**Design A.** Short-circuit deterministic sign-flip cases before the LLM judge. The new verdict label `numeric_short_circuit` joins `JudgeVerdictLabel` in `consistency_checker/check/providers/base.py`. Confidence is fixed at `1.0`; rationale is a deterministic string of the form `"Numeric short-circuit: metric={metric}, A={value_a}{unit}, B={value_b}{unit}, polarity mismatch."`. `evidence_spans` contains the raw verbatim source text for each side of the mismatch.

Downstream consumers treat `numeric_short_circuit` the same as `contradiction` for reporting purposes (it *is* a contradiction, just deterministically derived). Filtering logic in `audit/report.py` accepts both labels.

Range overlaps that aren't sign-flips — e.g. "Revenue grew 12%" vs. "Revenue grew 8%" with the same scope — get handled in Step E3 as a *side-channel* prompt hint to the judge (uncertain verdict expected). The two paths are complementary, not exclusive: deterministic cases short-circuit, ambiguous cases go to the LLM with extra structured context.

## Non-goals (v0.2)

- **No LLM metric canonicalisation.** "Revenue" and "Top-line revenue" stay distinct; "FY25" and "fiscal year 2025" stay distinct unless our scope heuristic happens to normalise them. Entity-NER-driven scope resolution is v0.3.
- **No unit conversion across systems** ("12 million dollars" vs. "$12 million" yes; "12 million dollars" vs. "€11 million" no — currency conversion is out of scope).
- **No multi-metric reasoning.** "Revenue grew 12% and margin expanded" decomposes via atomic-fact extraction (Step 7 from v0.1); the numeric extractor sees the atomic outputs, not the original prose.

## Consequences

- `JudgeVerdictLabel` Literal gains one value. `findings.judge_verdict` is free `TEXT` so no SQL migration is required.
- Report renderer's grouping logic widens: short-circuit findings appear alongside LLM-derived contradictions in the summary table and the per-pair detail sections.
- Audit logger doesn't change shape — same `record_finding` call path with a different verdict value and a fixed `confidence=1.0`.
- Cost win: on a financial-corpus benchmark, sign-flip pairs are a meaningful fraction of all candidate pairs surviving Stage A. Short-circuiting them is a direct LLM-spend reduction, with the additional benefit that deterministic reasoning is reproducible across re-runs (no temperature noise on the verdict).
- If the numeric extractor produces a false positive (incorrectly flags a non-contradiction as sign-flip), there's no LLM safety net for that case. Mitigation: keep the extractor conservative — require *strict* metric+scope+unit match before short-circuiting. Anything ambiguous falls through to the LLM judge as before.
- Future v0.3+ extension: when entity-NER (futureplans #7) lands, the numeric extractor's scope heuristic can be replaced with NER-derived scope canonicalisation, raising recall without changing the short-circuit interface.
