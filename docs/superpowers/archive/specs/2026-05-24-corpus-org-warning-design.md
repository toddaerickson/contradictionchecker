# Corpus-composition warning + document-level org grouping

**Date:** 2026-05-24
**Status:** Approved by user; pending implementation plan
**Tracks:** [`futureplans.md`](../../../futureplans.md) item #2 (corpus composition)
**Supersedes:** `2026-05-21-org-grouping-corpus-warning-design.md` on the parked
branch `feat/org-grouping-corpus-warning`. This document folds in the
2026-05-21 multi-agent review's binding revisions, locks the org-identifier
strategy at LLM-only, and reflects the now-shipped item #1 short-circuit
(PR #62) which removed the original sequencing dependency.

## Problem

The real-corpus eval (2026-05-21) showed that the dominant residual driver of
"definition-divergent" false positives on the bylaws corpus is **corpus
composition**: comparing the bylaws of three *unrelated* organizations makes
every shared term ("Director", "Quorum", "the Corporation") "diverge" by
construction, because each org legitimately defines the term for itself. The
detector has no notion of which organization a document belongs to, pairs Org
A's "Director" against Org B's, and flags a non-meaningful contradiction.

The tool also never tells the user that comparing unrelated entities is the
root cause. That is the usability half of the problem.

The eval did not quantify the corpus-composition share of the residual.
Restated honestly: this build addresses a known driver of noise without
claiming a precise percentage. Section §8 adds a cheap measurement step to
size the effect post-ship.

## Goals

1. Identify the primary organization each document belongs to (document-level,
   one org per document).
2. Group documents into org buckets and **warn** the user when a corpus spans
   more than one bucket — or when an expected single-org corpus fragments
   into multiple buckets (inverse failure).
3. Default behavior is **advisory-only**: warn but still compare globally.
   Cross-org suppression is **opt-in** via `--org-scope` (config
   `org_scope_enabled: bool = False`).
4. When suppression is enabled, every would-be-suppressed pair is still
   written to the audit DB so the precision-measurement surface survives.
5. Distinguish identification failure from a genuine "no organization
   identifiable" result; surface a loud notice when the failure rate exceeds
   a threshold so a wrong API key never presents as "single-org corpus."

## Non-goals (explicitly deferred)

- **Entity-level coreference** ("the Borrower" = "ABC Corp" within one
  document). Stays under item #7's original framing.
- **Fuzzy / edit-distance org matching.** v1 groups by exact match on a
  normalized key. Add fuzzy matching only when real corpora demonstrate the
  need.
- **Re-scoping the pairwise / triangle detectors.** Deprioritized on legal
  prose per the 2026-05-21 eval; this build touches only the definition
  detector.
- **Manual org assignment in the UI.** Org labels are content-derived. The
  user-assigned `corpora` table (migration 0011) is left untouched.
- **Cheap-signal (filename / title heuristic) identifier.** Considered and
  rejected — see §10.

## Architecture

### §1. Org identification (ingest-time)

Org identification rides on the existing extractor surface, not a new
parallel provider tree. Each LLM-backed extractor gains an `identify_org`
method; `Extractor` (in `consistency_checker/extract/atomic_facts.py`) gains
a corresponding method in its Protocol.

```python
@dataclass(frozen=True, slots=True)
class OrgIdentification:
    label: str | None
    reason: Literal["org_found", "no_org", "llm_error", "truncated"]
```

- `AnthropicExtractor.identify_org(text, title) -> OrgIdentification`
- `MoonshotExtractor.identify_org(text, title) -> OrgIdentification`
- `FixtureExtractor.identify_org(text, title) -> OrgIdentification` — keyed
  by content for hermetic tests.

The method sends only the title plus the first ~2000 characters of the
document, framed as **inert data** in clearly delimited blocks so an
attacker-controlled document body cannot redirect the prompt. The structured
output schema constrains the response to one string field plus the reason
enum; nothing the document body says can produce verdict-shaped output that
gates findings display.

The identifier returns the **single primary issuing org** (the entity whose
bylaws / agreement / policy this *is*). If genuinely ambiguous — e.g. a
joint-venture agreement between two parties — return `reason="no_org"`.

The identifier runs once per document, at ingest time only. The resolved
label persists on the document row. Check-time cost is unchanged.

Idempotency: org identification piggybacks on the existing `INSERT OR
IGNORE` ingest path. Re-ingesting identical content does not re-call the
LLM for documents already present.

### §2. Storage

Migration `consistency_checker/index/migrations/0013_document_org.sql`:

```sql
ALTER TABLE documents ADD COLUMN org_label TEXT;
ALTER TABLE documents ADD COLUMN org_reason TEXT;
ALTER TABLE findings  ADD COLUMN suppressed INTEGER NOT NULL DEFAULT 0;
```

(SQLite stores booleans as INTEGER; see §6 for the `findings.suppressed`
contract.)

- `org_label`: raw display name returned by the identifier (nullable).
- `org_reason`: one of `org_found | no_org | llm_error | truncated | NULL`
  (`NULL` = "this row pre-dates the feature"; see backfill in §7).
- The normalized grouping key (`org_key`) is computed on the fly in §3, never
  stored — so normalization can improve later without a migration.

`Document` dataclass (`extract/schema.py`) gains:

```python
org_label: str | None = None
org_reason: str | None = None
```

`AssertionStore.add_document` and row mapping carry both fields through.

`AssertionStore.iter_definitions()` is extended to yield `(Assertion,
org_key)` tuples so the checker stays pure over one iterable — no `doc_id
-> org_label` side map threaded through `find_inconsistencies`,
`_group_by_canonical_term`, `_enumerate_pairs`, or `estimate_cost`.
Callers that don't need the org key destructure and discard.

### §3. Normalization & grouping

```python
def normalize_org(label: str) -> str: ...
```

Lives in `check/definition_terms.py` (alongside `canonicalize_term`). Rules:

1. casefold; collapse internal whitespace and punctuation to single spaces
2. strip a single leading article (`the`)
3. strip a trailing legal-form suffix from the set
   `{inc, llc, l.p., lp, corporation, corp, company, co, ltd, limited}` **only
   when at least one significant token remains**. Entity-type words
   ("Trust", "Foundation") are intentionally NOT in this set — they
   distinguish organizations (a Trust is legally distinct from a Foundation)
   and must stay in the key.
4. trim

`org_key := normalize_org(org_label)`. Idempotent:
`normalize_org(normalize_org(x)) == normalize_org(x)`.

A **bucket** is the set of documents sharing one `org_key`. Documents with
`org_label is None` (whatever the reason) live in the **unknown bucket**.

### §4. Detection scoping (the opt-in payoff)

`definition_checker._group_by_canonical_term` becomes org-aware. Grouping
key is `(org_key, canonical_term)` when `org_scope_enabled` is True,
falling back to `canonical_term` only when False (pre-feature behavior).

Unknown bucket: definitions with `org_label is None` pair against
**every** org's definitions for the term. A failed extraction must never
silently hide a real comparison.

When `org_scope_enabled` is True, every cross-org pair that *would have
formed* without suppression is still recorded — see §6.

`pipeline.estimate_cost` calls the same grouping path so the cost preview
matches the actual run.

### §5. Corpus-composition warning (the default-on payoff)

Fires whenever the corpus resolves to more than one **non-unknown** bucket,
OR when an inverse-fragmentation guard fires (defined below).

**CLI**, printed after `ingest` and at the start of `check`:

Scope-off (default):

```text
⚠ Corpus spans 3 organizations: Acme Foundation, Inc. (2 docs),
  Beta Trust (1), Gamma Inc. (1). Plus 1 doc with no identified
  organization. Cross-org definition pairs are still compared;
  pass --org-scope to suppress them. Best results come from one
  organization's documents at a time.
```

Scope-on:

```text
⚠ Corpus spans 3 organizations: Acme Foundation, Inc. (2 docs),
  Beta Trust (1), Gamma Inc. (1). Plus 1 doc with no identified
  organization. Cross-org definition pairs are suppressed
  (--org-scope); pass --no-org-scope to compare across orgs.
```

**Bucket display rule:** show the **first-seen raw `org_label`** among the
labels that share an `org_key`. Surface the unknown-bucket size as a
distinct sentence; never fold it into the org list.

**Web:** a `.cc-banner` on the Stats tab carries the same message, shown
only when the corpus spans >1 non-unknown bucket or fragmentation fires.

**Inverse fragmentation guard:** for every pair of distinct buckets,
compute `pre_suffix_key`: the result of `normalize_org` with rule 3
(legal-suffix stripping) skipped. If two buckets' `pre_suffix_key`s are
equal, OR if their `org_key`s share their first whitespace-delimited token
and one is a strict prefix of the other, emit:

```text
⚠ Possible fragmentation: 'Acme Foundation, Inc.' and 'The Acme
  Foundation' resolved to different org keys. If they are the same
  entity, file a normalize_org issue.
```

**Identifier-failure notice:** if `>20%` of documents have
`org_reason in {"llm_error", "truncated"}`, emit:

```text
⚠ Organization identification failed on 4 of 7 documents (57%).
  Check your provider/API key. Org warnings below may be
  incomplete.
```

### §6. Audit trail when suppression is enabled

When `org_scope_enabled` is True, every cross-org pair that would have been
judged without suppression is written to the audit DB so item #1's
precision-measurement surface is preserved.

Implementation: reuse `findings` with
`detector_type='definition_inconsistency'` and add a new column
`suppressed BOOLEAN DEFAULT 0` in migration 0013 (alongside the
`documents` columns). The judge is **not called** for suppressed rows;
`judge_verdict` stays NULL and `confidence` is 0. Existing read paths
filter `suppressed=0` by default; the audit/eval surface opts in.

Chosen over a sibling `suppressed_pairs` table because (1) it preserves
one query path for "all definition pairs the system considered," (2)
existing finding-id provenance round-trips into reviewer workflows
unchanged, and (3) no new ORM/row-mapping code is needed.

`CheckResult` gains `n_definition_pairs_suppressed: int`. The run-summary
log includes it. The web Stats tab shows this count when nonzero.

### §7. Config / switches

Two Pydantic fields on `Config` (frozen — use `model_copy(update={...})`):

```python
org_grouping_enabled: bool = True
org_scope_enabled:    bool = False
```

- `org_grouping_enabled`: master switch. When False, the identifier is
  never constructed, no warning is emitted, no scoping occurs, and
  `org_label` stays `NULL`.
- `org_scope_enabled`: when True, the definition detector suppresses
  cross-org pairs (and writes them to the audit trail per §6). When False
  (default), pairs are still judged; only the warning surfaces.
- `org_scope_enabled=True` is a no-op when `org_grouping_enabled=False`
  (documented + asserted by config test).

Both honor the `CC_<FIELD>` env-override convention in `config.py`.

CLI flags on `check` and `ingest`: `--org-scope` / `--no-org-scope`.

### §8. Backfill for NULL labels

`INSERT OR IGNORE` makes plain re-ingest a no-op, so pre-feature documents
stay `NULL` forever without an explicit path. Ship a new CLI subcommand:

```text
consistency-check store reidentify-orgs [--all | --null-only]
```

`--null-only` (default) walks documents where `org_label IS NULL` and
runs the identifier on them; `--all` re-runs identification on every
document. Writes results back via UPDATE. Idempotent. Used for the
initial post-migration backfill *and* as a measurement tool for §9.

### §9. Measurement (post-ship, before any v2)

Two cheap checks the operator runs after the feature ships:

1. **Identifier hit rate on the real bylaws corpus.** Run
   `reidentify-orgs --all` and tally `org_reason` distribution. If
   `llm_error + truncated` exceeds 5%, file a follow-up; if `no_org`
   dominates a single-org corpus, file an identifier-prompt issue.
2. **Suppression impact.** On the same corpus, run `check --org-scope`
   and compare the definition-divergent count to the pre-feature 75%
   number from the 2026-05-21 eval. Record the delta in a Completed-section
   note in `futureplans.md`. **This is the number that decides whether the
   build was worth it.**

## Data flow

```text
ingest:
  load doc -> (if org_grouping_enabled) extractor.identify_org(text, title)
           -> store documents.org_label, documents.org_reason
           -> if >1 non-unknown bucket OR fragmentation: print corpus warning
           -> if identifier-failure rate >20%: print failure notice

check (definition pass):
  load (assertion, org_key) tuples
  (if org_scope_enabled)
      group by (org_key, canonical_term)
      record cross-org would-be-pairs in suppressed audit trail
  (else)
      group by canonical_term
  enumerate pairs within groups -> definition judge -> findings
  emit corpus warning + (if nonzero) suppression count
```

## Error handling

- Org identifier LLM error → `OrgIdentification(label=None, reason="llm_error")`.
  Ingest does not fail; the doc enters the unknown bucket and compares
  globally.
- Empty / garbage / refusal response → `reason="no_org"`.
- Document shorter than the truncation cap with no org named →
  `reason="no_org"`. Document longer with no org in the first 2000 chars →
  `reason="truncated"` (the safe direction is "compare globally").
- `org_grouping_enabled=False` is a clean no-op: every existing test that
  does not opt into org grouping behaves exactly as before.

## §10. Why not a cheap-signal identifier

Multi-agent review's PM lens recommended evaluating title/filename
heuristics or the existing `corpora` table before committing to a per-doc
LLM call. We considered three implementations during brainstorming
(2026-05-24):

- **A — LLM-only** (this design)
- **B — Cheap signal first, LLM fallback** (regex on title + first 500
  chars, fall through to LLM when signal is unconfident)
- **C — Measure cheap-signal hit rate first, then pick A or B**

**Picked A.** Rationale for a single-developer-maintained tool: B is two
code paths, two failure modes, and two test matrices forever, in exchange
for a per-corpus cost reduction that Moonshot already makes negligible
(~$0.001/doc). C delays shipping for a measurement whose only output is
"build B or A" — if the answer is A anyway, the measurement was wasted.

If a future user reports the per-doc LLM cost is unacceptable, the cheap
signal can be added behind the same `identify_org` interface without
changing callers. The `OrgIdentification` return type already encodes the
reason, so a cheap signal can return `reason="org_found"` directly.

## Testing (hermetic, CI-safe)

All new tests are default-mark (hermetic) — no model downloads, no live API.

- `normalize_org`: each rule (article, every legal suffix, whitespace/punct
  collapse), idempotence, and **non-merge** cases ("Acme Trust" vs "Acme
  Foundation"; "Acme Corp" vs "Acme Trust" must NOT share a key; "Trust"
  alone does not collapse to empty).
- `FixtureExtractor.identify_org` returns canned `OrgIdentification`s; round-trip
  through `pipeline.ingest()` persists `org_label` and `org_reason`,
  retrievable via `get_document`.
- Scoping: two fixture orgs each defining "Director". With
  `org_scope_enabled=True`, **0** cross-org definition pairs reach the
  judge; with `False`, the cross-org pair forms and is judged.
- Suppression audit: with `org_scope_enabled=True`, every would-be cross-org
  pair appears in the suppressed audit surface and
  `n_definition_pairs_suppressed` is correct.
- Unknown bucket: one doc with `org_label=None` defining "Director" pairs
  against every known org's "Director" even when scoping is on.
- Warning trigger: a >1-bucket corpus surfaces the warning string; a
  single-org corpus does not.
- Fragmentation guard: "Acme Foundation, Inc." and "The Acme Foundation"
  resolved to different keys triggers the fragmentation notice.
- Identifier-failure notice fires above 20%; does not fire at or below.
- Master switch: `org_grouping_enabled=False` skips identification, leaves
  `org_label` NULL, emits no warning, no scoping. Config test asserts
  `org_scope_enabled=True` is a no-op under this switch.
- `estimate_cost` uses the org-aware grouping path; counts match a real run.

A live test (marked `live`) exercises one real `identify_org` call against
Moonshot to keep the prompt contract honest. Skipped in CI.

## Migration / compatibility

- `0013_document_org.sql` is additive (two nullable columns); existing DBs
  upgrade in place via the filename-ordered migration loader. Never edit
  prior migrations.
- Pre-feature documents land in the unknown bucket on first read after
  upgrade; the `reidentify-orgs` command (§8) populates them when run.
- Release note: document the new ingest cost (one LLM call per new doc),
  the `--org-scope` flag, and the `reidentify-orgs` backfill command.

## Roadmap bookkeeping

On completion:

- Move item #2's open status out of "Eval findings & next levers" into the
  Completed section of `futureplans.md`, with the §9 measurement result
  recorded inline.
- Note that the entity-coreference half of item #7 remains deferred.
- A fresh ADR records the org-identifier provider surface (consistent
  with the "new judge / new provider → ADR" convention).
- Delete the parked branch `feat/org-grouping-corpus-warning` (its only
  unique content was the predecessor spec, now superseded by this one).
