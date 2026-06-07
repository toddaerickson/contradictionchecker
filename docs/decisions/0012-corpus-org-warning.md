# ADR-0012 — Corpus-composition warning + opt-in org grouping

**Date:** 2026-05-24
**Status:** Accepted
**Spec:** [`docs/superpowers/archive/specs/2026-05-24-corpus-org-warning-design.md`](../superpowers/specs/2026-05-24-corpus-org-warning-design.md)
**Plan:** [`docs/superpowers/archive/plans/2026-05-24-corpus-org-warning.md`](../superpowers/plans/2026-05-24-corpus-org-warning.md)

## Context

The 2026-05-21 real-corpus eval identified cross-organization corpus
composition as the dominant residual driver of false-positive
"definition_divergent" findings on the bylaws corpus. PR #62 closed the
identical-text subcase; this build addresses the composition driver.

## Decision

- Identify each document's single primary issuing org via the existing
  LLM extractor surface (Anthropic, Moonshot), not a new parallel provider
  tree. `identify_org` ships on the `Extractor` Protocol.
- Default behavior is **advisory-only**: warn when a corpus spans more
  than one org bucket, but continue judging all pairs. Cross-org
  suppression is **opt-in** via `--org-scope` (config
  `org_scope_enabled: bool = False`).
- When suppression is on, would-be-suppressed pairs are still written to
  `findings` with `suppressed=1` (no judge call) so the precision-
  measurement surface survives.
- `normalize_org` is precision-safe (no single-token collapses; entity
  types Trust/Foundation are NOT in the strip list) and lives beside
  `canonicalize_term` in `check/definition_terms.py`.
- The fragmentation guard and the >20% identification-failure notice
  surface as additional CLI / web warnings.

## Rejected alternatives

- **Cheap-signal identifier (title/filename heuristics).** Considered.
  Rejected because two code paths add maintenance burden that the per-doc
  Moonshot call cost (~$0.001/doc) does not justify. Can be added later
  behind the same `identify_org` interface without changing callers.
- **Default-on suppression.** Rejected by the parked-spec multi-agent
  review: silently regresses recall on single-org corpora that fragment
  into multiple keys, and erases the item #1 eval signal.
- **Sibling `suppressed_pairs` table.** Adds a new query path; the
  `findings.suppressed` column preserves one query surface for "all pairs
  the system considered."

## Consequences

- Every new ingest does one extra LLM call (~$0.001/doc on Moonshot).
- Pre-feature documents stay `org_label IS NULL` until
  `consistency-check store reidentify-orgs` runs.
- `iter_definitions()` now returns `(Assertion, org_key)` tuples;
  downstream consumers destructure.
- Adds an `index → check` import (`AssertionStore` reads `normalize_org`
  from `check.definition_terms`). Minor layering smell; acceptable
  because `definition_terms` has no internal dependencies. If this
  pattern repeats, consider moving normalization helpers to a leaf
  module.
- The §9 post-ship measurement (divergent-rate delta on the bylaws
  corpus) decides whether the build paid off; recorded in
  `futureplans.md` once collected.
