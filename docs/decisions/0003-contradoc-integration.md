# ADR 0003 — CONTRADOC benchmark integration timing

**Status**: Accepted

## Context

CONTRADOC is a published dataset for evaluating cross-document contradiction detection, with gold labels suitable for computing precision/recall/F1. Two integration timings were considered:

- **In MVP**: Build `benchmarks/contradoc_harness.py` alongside the pipeline (Step 16 in the build plan). Gives us hard numbers from day one, lets us tune the NLI threshold (Step 10) against real data, and surfaces precision regressions immediately when prompts or models change.
- **Deferred to v0.2**: Ship MVP with only the hand-crafted end-to-end fixture (Step 15); add CONTRADOC once the pipeline is stable.

The two-stage NLI + LLM design is justified largely by an external precision claim (~89% hybrid vs. 16% NLI-alone). Without an internal benchmark, we have no way to verify that claim holds in our own implementation.

## Decision

**Integrate CONTRADOC in MVP scope.** Step 16 of the build plan is in-scope, not optional. The benchmark is not part of CI (too slow, requires the dataset on disk) but is documented and runnable via `python -m benchmarks.contradoc_harness`.

Initial pass-bar: precision ≥ 0.7 on a 50-sample slice. Failing that bar triggers threshold re-tuning in Step 10 or prompt iteration in Step 11.

## Consequences

- Roughly two extra days of work over the deferral path.
- Adds a `benchmarks/` directory and a small runbook in `docs/benchmarks.md`.
- The CONTRADOC dataset is not redistributed; loaders fetch from the original source or expect a local path.
- Once the harness is in place, future PRs that touch the gate / NLI / judge are expected to include before/after metrics from a sample run.
