# ADR 0008 — Persona-aware analysis: a view layer, not forked judge agents

**Status**: Proposed

## Context

CrossCheck's judge asks one perspective-neutral question: "do these assertions contradict?" But the consumers of a document set don't read it neutrally. An HR handbook is read differently by an employee, a manager, and an HR professional. A loan package — term sheet, credit agreement, security agreement — is read differently by a lender, a credit analyst, the borrower, and the borrower's counsel.

Three things vary by persona; one thing does not:

- **Invariant — hard contradictions.** `X` and `¬X` at the same scope is a contradiction for everyone. The detector must stay objective.
- **Variable — materiality.** A parking-policy conflict matters to an employee; it's noise to outside counsel.
- **Variable — scope priors.** "All employees get four weeks" vs "engineers get two weeks" only contradicts if engineers ⊆ employees and the scopes overlap. Personas bring different default assumptions about scope, which can flip a borderline `uncertain` to `contradiction`.
- **Variable — the question itself.** A borrower's counsel isn't only hunting contradictions — they hunt undefined terms, silent gaps, and ambiguous provisions. Those are *absences and vaguenesses*, not contradictions; the current pair judge can only return verdicts about two assertions that both exist, so it returns nothing for them.

## Decision

Persona-aware analysis is a **scoring and presentation layer over the existing perspective-neutral detector**, plus one optional detection hook — **not** a set of forked judge agents.

A `Persona` config object (name, interests, scope assumptions, materiality weights) feeds:

1. An **impact scorer** that re-ranks and filters the shared `findings` per persona. Pure presentation; reuses the existing audit DB; selectable in the report renderer and the v0.3 web UI.
2. An optional `persona_context` block spliced into the judge prompt — the same mechanism E3's `numeric_context` already uses (ADR-0005). Additive; only nudges borderline scope calls.

The core detector, the audit trail, and the cache stay single and shared. The persona is a *view*, not a pipeline.

Forked per-persona judge agents are rejected: N personas × every pair multiplies LLM spend, defeats the cache, and invites the model to conflate "matters to this persona" with "is a contradiction."

## Consequences

- Personas live in config plus the report/web layer, not the pipeline core. v0.3 Block G's renderer already separates detection from presentation — personas extend that seam.
- The "different question" cases (gaps, undefined terms, ambiguity) are **out of scope for this ADR**. They need *new detectors* — a gap detector (`X` vs silence), a definition-consistency detector, an ambiguity detector — each its own future ADR, sharing the assertion store but asking a different question of the assertion graph. CrossCheck's contradiction detector is the first of a planned family; personas then map to *which detectors run* as well as how findings rank.
- Persona authoring is a prompt-engineering and eval problem as much as a code problem. This ADR is **Proposed**, not Accepted: build is gated on labeled eval data showing borderline cases actually flip per persona. Ship the report/UI persona *filter* first (cheap, reuses everything); add the prompt-context hook only once eval justifies it.
