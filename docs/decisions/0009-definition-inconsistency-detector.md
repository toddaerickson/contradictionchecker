# ADR 0009 — Definition-inconsistency detector

**Status**: Accepted

## Context

CrossCheck's contradiction judge structurally cannot find issues whose shape isn't a pair of opposing claims. A reviewer reading a loan package, an underwriting policy, or an HR handbook also cares about another class of issue: **the same defined term used inconsistently across the corpus**. Term-sheet `"Material Adverse Effect"` scoped to business condition, credit-agreement `"Material Adverse Effect"` scoped to performance ability — neither is "wrong"; together they are inconsistent.

This is the first new member of the detector family flagged in [`futureplans.md`](../../futureplans.md) item #20 (and previewed in [ADR-0008](0008-persona-aware-analysis.md), which scoped *which detectors to run* per persona but punted on building them). It must reuse the existing ingest, embedding, judge, and audit infrastructure without leaking definition-specific concerns into the contradiction surface.

Two cardinality shapes were considered for v1:

- **Flavor A (definition ↔ definition).** Two `assertions` rows with `kind='definition'` for the same canonical term diverge in meaning. Pair-shaped — fits the existing `findings` schema natively.
- **Flavor B (definition ↔ usage).** One `kind='definition'` row vs an occurrence of the term in a non-definition assertion. Usage detection is a much larger candidate set; verdict interpretation is more subjective.

A third class — *usage of a term that is never defined* — is structurally a **gap** (presence vs absence), not a definition inconsistency, and is deferred to the planned gap detector.

## Decision

Build **flavor A only** in v1, sharing the existing `findings` table with a new `detector_type` discriminator column. Defer flavor B; route flavor C to the gap detector.

Four specific decisions:

1. **Narrow reuse, not supertype.** `findings` gains a `detector_type` column (default `'contradiction'`) rather than introducing a separate `issues` table. The supertype refactor is deferred to when the gap/ambiguity detectors arrive and force richer cardinality.

2. **LLM-based definition extraction, folded into the existing atomic-fact extractor.** The target corpora — underwriting memos, organic policy memos, HR handbooks — express definitions in free-form prose (`"by 'stabilized' we mean ..."`, `"an eligible employee is ..."`, `"for purposes of this policy, ..."`). Syntactic patterns would miss 30-60% of those. Definitions ride alongside atomic facts in one tool call (`record_extraction`) — no new API call per chunk; output-token cost only. A `slow`+`live` A/B regression test gates the "should we split into two LLM calls" decision if recall regresses.

3. **NLI gate bypassed; term grouping replaces it.** The DeBERTa NLI gate is contradiction-tuned and unhelpful for comparing two definitions of the same concept. The `DefinitionChecker`'s gate is the canonical-term grouping step (case-fold, strip `the `, drop trailing plural `s`, strip surrounding quotes). Pairs only reach the judge when their canonical terms match. Cheaper and more precise than the contradiction gate for this question.

4. **Judge verdict surface narrow.** A separate `DefinitionJudgePayload` Pydantic model and `DefinitionJudgeProvider` Protocol carry the three-valued vocabulary (`definition_consistent` / `definition_divergent` / `uncertain`). This mirrors the multi-party pattern in [ADR-0006](0006-three-doc-conditional.md) and prevents an LLM from claiming a `contradiction` verdict in the definition path or vice versa.

## Consequences

- **Schema gains two migrations.** `0007_assertion_kind.sql` adds `kind` / `term` / `definition_text` to `assertions`; `0008_finding_detector_type.sql` adds `detector_type` to `findings`. Both default to current-behavior values so existing audit DBs migrate without data changes.

- **Pipeline ordering.** `pipeline.check()` runs the definition stage after the multi-party pass in the same `run_id`. Both stages contribute to `n_findings`. The CLI flag `--no-definitions` opts out; default is enabled.

- **Report and web UI extend, don't fork.** The markdown report gains a `## Definition inconsistencies` section appended after the contradictions and multi-party sections. The web UI adds a `Definitions` tab and two stats counters. Existing surfaces are byte-stable on prior-shape runs because both render-paths emit only when there are findings of the new kind.

- **Persona mapping is automatic.** Per ADR-0008, personas map onto *which detectors run*. Once persona config lands, an "auditor's-counsel" persona enables `detector_type='definition_inconsistency'` while a "summary-reader" persona disables it. No detector-side work required for that integration.

- **What this unlocks for the family.** `assertions.kind` and `findings.detector_type` are the two discriminator columns the next two detectors (gap, ambiguity) will need. Gap will likely force a richer "issue" shape (one assertion + a document, not a pair); the supertype refactor will happen there, not here.

## Future work captured in `futureplans.md`

- **Flavor B (definition ↔ usage drift).** A definition + an occurrence of the canonical term in a non-definition assertion. Requires usage extraction; expanded candidate set; new finding shape (definition assertion + usage assertion, both already pair-shaped — likely no schema change). Tracked separately.

- **Cross-doc-only mode.** Flavor A today emits findings for divergent definitions within a single document as well as across documents. If false-positive prone in production, restrict to cross-doc pairs.

- **Persona-aware ranking of definition findings** is out of scope for this ADR; tracked by ADR-0008.
