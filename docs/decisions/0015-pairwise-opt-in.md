# ADR 0015 — Pairwise contradiction detector becomes opt-in

**Status**: Accepted

## Context

The pairwise contradiction detector — a DeBERTa NLI gate that proposes candidate pairs, followed by an LLM judge that adjudicates each one — has been default-on since v0.1. It was the original premise of the project: surface contradictions that two passes of close reading would catch, by mechanically considering every pair of assertions a reviewer would otherwise eyeball.

The 2026-05-21 real-corpus run falsified that premise for the dominant corpus shape this tool is used on. Running against a legal-prose corpus, the NLI gate flagged 254 candidate pairs; the judge returned **1 contradiction, 23 uncertain, 230 not_contra**. The single contradiction was a borderline phrasing nit, not the kind of operational finding the tool is meant to surface. The signal-to-cost ratio on prose is effectively zero: legal language is dense with rhetorical negations, hedged clauses, and conditional language that NLI consistently mistakes for contradiction.

Meanwhile the **definition detector** (ADR-0009) — which finds the same term defined two different ways within or across documents — carries the headline value on every corpus shape we've tried. It's the detector that finds the real-world findings the user actually acts on.

The pairwise detector is not cheap to keep default-on:

- First-time `check` runs download a ~800 MB DeBERTa-v3 NLI model.
- Each `check` run holds the model in memory at ~600 MB RSS.
- Every NLI-flagged pair fires a paid LLM judge call, even on corpora where 99%+ are not_contra.
- The pre-flight memory check (`max_memory_mb`) exists specifically because the NLI load is the dominant memory cost.

The cost is paid on every run, regardless of whether the corpus is the kind where pairwise produces signal.

Alternatives considered:

- **Remove the pairwise code entirely.** Rejected. The 2026-05-21 eval is corpus-specific — legal prose is the worst case, but the same detector against a numeric- or spec-heavy corpus (engineering specifications, financial filings, lab reports) is where outright sign flips and quantity disagreements live, and that's exactly what NLI + judge is good at. Deleting the code throws away the future use case to optimize the present one.
- **Lower the NLI contradiction threshold.** Rejected. Tuning a known-low-signal detector is the wrong lever; on legal prose the issue is that NLI's notion of "contradiction" doesn't match the operational notion, regardless of threshold. Lowering threshold just adds noise, raising it just hides whatever signal remains.
- **Auto-enable pairwise under `--deep`.** Rejected as too implicit. `--deep` already means "do the expensive multi-party triangle pass"; coupling NLI download/RSS to it would surprise users who expected `--deep` to just spend more LLM tokens, not change the model footprint. Explicit `--pairwise` is clearer about what cost is being incurred.

## Decision

Flip the `pairwise_enabled` default to **False**. Pairwise becomes an opt-in detector behind an explicit `--pairwise` flag. Implementation notes:

- `Config.pairwise_enabled: bool = Field(default=False, ...)` lands alongside the other detector toggles (`junk_filter_enabled`, `ocr_enabled`, `org_grouping_enabled`, `org_scope_enabled`). YAML configs that don't mention the field automatically pick up the new default; users who want pairwise must say so.
- A tri-state CLI override `--pairwise / --no-pairwise` (default: respect config) lets operators flip the knob per-run without editing config.yml. The tri-state matters because explicit `--no-pairwise` on a corpus whose config opts pairwise in still has to win.
- When pairwise is off, the CLI and web entrypoints do **not** load the NLI model at all — no download on first run, no RSS at steady state, no pre-flight memory check. The cost we're eliminating is paid at import/load time, not at use time, so the gate has to skip the load entirely, not just skip evaluation.
- `pipeline.check()` accepts `nli_checker: NliChecker | None = None`. Passing `None` is the off-path. Tests can still inject a `FixtureNliChecker` for the on-path; the production wiring in `cli/main.py` constructs the real `NliChecker` only when `cfg.pairwise_enabled` (post CLI override) is True.
- `pipeline.estimate_cost()` returns `n_candidate_pairs=0` when pairwise is off. The cost preview that runs before a real `check` should reflect what the run will actually do — overstating it by including a detector that won't fire is misleading.
- `--deep` (the multi-party triangle pass) shares the strong NLI gate, so `--deep --no-pairwise` is rejected as a config error with a clear message. `--deep --pairwise` is the supported way to opt into both at once.
- The pairwise code path is **not removed**. Modules, prompts, providers, tests — all stay. Re-enabling it for the right corpus shape is one flag, not a revert. If the eval reverses on a numeric/spec corpus down the road, the only change is the default.

The implementation plan that produced this decision is at [`docs/superpowers/archive/plans/2026-05-31-pairwise-opt-in.md`](../superpowers/plans/2026-05-31-pairwise-opt-in.md).

## Consequences

- First-time `check` runs no longer download the ~800 MB DeBERTa NLI model by default. Users who want pairwise pay the download on first `--pairwise` run instead, with the same one-line warning that already gates the OCR model download.
- Peak RSS during `check` drops by ~600 MB on default runs. The `max_memory_mb` pre-flight check is skipped entirely when pairwise is off, since the dominant memory consumer doesn't load.
- Users who previously relied on pairwise findings must add `--pairwise` to their `check` invocations. Existing `config.yml` files automatically pick up the new default because they don't mention the field; users who want pairwise default-on for a given corpus add `pairwise_enabled: true` to that corpus's config.
- The web UI silently inherits the new default. A future follow-up may surface a "run pairwise detector" checkbox; that's out of scope for this ADR.
- `estimate-cost` numbers shrink on default runs because `n_candidate_pairs` reports 0 unless `--pairwise` is also passed. The numbers are now an accurate preview of what the corresponding `check` invocation will do; the previous behavior overstated default-run cost by including a detector that produced ~0.4% useful yield in the worst case.
- The definition detector remains the headline value on every corpus shape; nothing about its wiring or defaults changes. This ADR is strictly about rebalancing which detectors are on-by-default, not about which detector is canonical.
- The pairwise code, prompts, providers, and tests stay in the tree. Flipping the default back is a one-line change if the eval reverses on a future corpus.
