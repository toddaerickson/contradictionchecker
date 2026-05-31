# ADR 0016 — Pre-flight cost ceiling for `check`

**Status**: Accepted

## Context

Today `consistency-check check` can spend an unbounded amount of money. The `estimate-cost` command exists but is opt-in and uses a fixed `$0.003-$0.010` per-call range — calibrated for Anthropic and OpenAI, but ~10-100× too high for Moonshot/Kimi (the project's default since PR #60). Two failure modes follow from that:

- Users who flip `judge_provider: moonshot` see scary cost projections from `estimate-cost` that are off by 1-2 orders of magnitude. The number is so far from reality that operators learn to ignore it, which defeats the point of having a cost preview at all.
- There is no automatic ceiling on a real run. A 50,000-assertion corpus with the definition detector engaged can rack up $50-$500 of judge calls before anyone notices. The OOM pre-flight (`max_memory_mb`) gates RAM but not money — a corpus that fits in RAM can still blow the budget by an order of magnitude.

The pre-flight memory check (ADR-0015's `max_memory_mb`) is the closest analog: opt-in, fires before the expensive load, and aborts cleanly. A budget gate wants the same shape.

## Decision

- Add `Config.max_cost_usd: float | None = Field(default=None, ge=0)`. When set, `pipeline.check()` runs `estimate_cost()` as a pre-flight; aborts with `CostCeilingExceeded` BEFORE judge or NLI bootstrap if `est_cost_high > max_cost_usd`.
- Add `--max-cost <USD>` CLI flag on `check`. No flag on `estimate-cost` (`estimate-cost` is itself the estimator; gating the estimator on its own output would be circular).
- Add `pipeline.default_per_call_costs(judge_provider) -> tuple[float, float]`. Anthropic/OpenAI: `(0.003, 0.010)`. Moonshot: `(0.0001, 0.001)`. Fixture: `(0.0, 0.0)`.
- `estimate_cost()` and `check`'s internal pre-flight default `per_call_low/high` from the provider when callers don't pass values explicitly. User overrides win.
- Conservative gate: uses `est_cost_high` (NOT `est_cost_low`). False positives (safe runs rejected) but no false negatives (over-budget runs let through). Correct asymmetry for a budget guardrail — when the gate is wrong, it's wrong in the safe direction.
- Hard-coded prices live in `pipeline.py` next to `default_per_call_costs`. ADR documents the source; price drift is a one-line update, no schema migration, no config churn.

## Alternatives considered

- **Live per-call cost tracking with mid-run abort.** Rejected for v1 — requires every judge provider to return usage information from each call (token counts, model-specific pricing) and a running tally with cooperative cancellation. That's a much bigger surface change and earns its own PR. The pre-flight ceiling captures ~80% of the value (catches obviously-too-expensive runs before any spend) at ~20% of the cost.
- **Soft warning instead of abort.** Rejected — a budget gate that warns but doesn't enforce is not a budget gate. Operators who want a warning can run `estimate-cost` themselves before `check`; the ceiling exists for the case where the run starts unattended and the user wants a hard stop.
- **Per-provider config sections (separate `anthropic_per_call_low`, `moonshot_per_call_low`, etc.).** Rejected — bloats the config for a value users rarely need to override. The `default_per_call_costs` function plus CLI overrides handles the common case; advanced users with a custom rate sheet pass `--per-call-low`/`--per-call-high` explicitly.

## Consequences

- Users with `judge_provider: moonshot` see realistic sub-cent projections from `estimate-cost`, which makes the preview a number worth looking at instead of a number worth ignoring.
- A `--max-cost $5` flag is a safety net for first-time runs against unfamiliar corpora — the operator can cap the blast radius without having to read the eventual `estimate-cost` output and decide.
- Edge case: `--max-cost 0.0` with any pairs to judge aborts. That's intentional — "I want to spend exactly nothing" is a valid signal, and the gate honoring it is the correct behavior.
- `CostEstimate.per_call_low`/`per_call_high` still record what was used for the projection. Replay and audit are unchanged; the only change is whether a run was permitted to start.
- README + CORPORATE_SETUP gain a one-line mention of the field; `config.example.yml` shows a commented `max_cost_usd: null` so operators can see the knob exists without it being on.

Reference: [`docs/superpowers/plans/2026-05-31-max-cost-ceiling.md`](../superpowers/plans/2026-05-31-max-cost-ceiling.md).
