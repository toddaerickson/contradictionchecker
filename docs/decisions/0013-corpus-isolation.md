# ADR-0013 — Corpus isolation (logical, single DB)

**Date:** 2026-05-25
**Status:** Accepted
**Spec:** [`do../superpowers/archive/specs/2026-05-25-corpus-isolation-design.md`](../superpowers/archive/specs/2026-05-25-corpus-isolation-design.md)
**Plan:** [`docs/superpowers/archive/plans/2026-05-25-corpus-isolation.md`](../superpowers/plans/2026-05-25-corpus-isolation.md)

## Context

A single-DB pool with no CLI-level scoping forced a choice between
polluting prior runs or deleting them. Surfaced mid-Atkins shake-down
test 2026-05-25 — the operator's DB held 5 unrelated prior-session
documents alongside an in-flight Atkins ingest, and the natural
mitigation ("delete the old docs") would destroy prior compute
(extracted assertions, definitions, reviewer verdicts).

## Decision

- **Logical isolation.** Every `documents` and `pipeline_runs` row carries
  a `TEXT corpus_id` FK to `corpora`. FAISS index stays shared.
- **`--corpus` is required** on every mutating/judging CLI command
  (ingest, check, estimate-cost, export, store reidentify-orgs).
  Interactive picker on TTY; hard error in scripts.
- **Pre-isolation rows backfill** to a `legacy` corpus via migration
  0014. No data loss; reviewer verdicts preserved (they're not corpus-
  scoped at the schema level — see Consequences).
- **FAISS gate post-filter** drops any candidate pair whose endpoints
  aren't both in the corpus's assertion-id set. Mandatory, silent —
  not surfaced to the user since it's a hard isolation boundary, not a
  meaningful suppression.
- **`report --run <id>` infers corpus** from `pipeline_runs.corpus_id`;
  explicit `--corpus` mismatch errors out.
- **Web UI mirrors** the CLI: uploads require a `corpus_id` form field;
  read routes filter by `?corpus=<id>` query param; the check-run
  trigger requires `corpus_id` too.

## Rejected

- **Per-corpus DB files / per-corpus directories** (Q3-B/C from
  brainstorming). Cleaner physical isolation, but every CLI command
  would need refactoring and the `corpora` table would move to a
  top-level registry. Bigger change for marginal isolation gain given
  FAISS still shared.
- **Default-on suppression / optional --corpus.** Would re-introduce
  the same accidental-pollution risk this spec exists to eliminate.
- **Denormalized `corpus_id` column on `assertions` / `findings`.**
  Derivable via FK chain (`finding → assertion → document → corpus_id`).
  Add only if real-corpus query times demonstrate need.

## Consequences

- One extra mandatory CLI flag on five commands; interactive picker
  reduces friction on TTY.
- `reviewer_verdicts` has no FK to assertions (its `pair_key` is a
  derived string). Corpus delete leaves orphan verdicts. Accepted in
  v1; cleanup is a follow-up.
- Adds an `index → check` import (`AssertionStore` reads `normalize_org`
  from `check.definition_terms`, established in PR #64 — preserved here).
- Companion **archive spec** (deferred) builds on this: a corpus can
  be exported to a portable artifact (tarball, cloud upload) once
  it's a real isolation unit. Tracked in `futureplans.md`.
- The `corpora.judge_provider` CHECK constraint allows only
  `('moonshot', 'anthropic')`. The CLI clamps `config.judge_provider`
  to one of these when creating a corpus row. Future spec: widen the
  CHECK or drop it.

## Migration notes (operational)

- Migration 0014 is additive; no data loss.
- Pre-isolation rows land in `legacy`. The `consistency-check corpus
  reassign --from legacy --to <new> --where "<safe-listed clause>"`
  helper moves them in bulk (e.g. `--where "org_label LIKE 'ATKINS%'"`
  for the Atkins-mid-flight case).
- `corpus list` shows what's in the DB; `corpus delete --yes-i-mean-it`
  cascades a corpus and its documents/assertions/findings.
