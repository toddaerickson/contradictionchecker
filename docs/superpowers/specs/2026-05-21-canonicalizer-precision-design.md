# Definition-judge precision: identical-text short-circuit + prompt tightening

**Date:** 2026-05-21
**Status:** Approved (design, revised after two review rounds)
**Roadmap item:** futureplans.md item #1 — definition-judge / canonicalizer precision
(the eval-named highest-leverage lever). Sequenced ahead of item #2 (org grouping); the
item-#2 design is parked on branch `feat/org-grouping-corpus-warning` (not on `main`).

> **Revision note.** An earlier draft of this spec also rewrote `canonicalize_term`
> (drop plural-strip, case-sensitive grouping). A multi-agent review showed that change
> was *speculative* (no observed over-merge in the eval) and could only *reduce* recall,
> while the one observed failure is fixed entirely by the short-circuit below. Applying
> the no-net-complexity guardrail, **the canonicalizer rewrite is deferred** (see
> "Deferred"). This spec is now the minimal, evidence-backed change.

## Problem

The 2026-05-21 real-corpus eval found one concrete, reproducible definition-detector
false positive: the judge flagged **textually identical** definitions as
`definition_divergent` (cited example: *"the authorized number of directors of the
Corporation"* compared against the same string). Two documents in the corpus defined the
same term with the same words, the detector grouped them (correctly — same canonical
term), and the LLM judge nonetheless returned `definition_divergent`.

That is a judge-precision failure on a deterministically-decidable case. We should never
spend an LLM call — let alone risk a wrong verdict — on a pair whose two definition texts
are equal after trivial normalization.

## Scope

In scope:

1. A deterministic **identical/near-identical text short-circuit** at the checker layer:
   a formed definition pair whose texts are equivalent (whitespace + case + surrounding
   punctuation) resolves to a machine verdict **without** an LLM call.
2. **Tighten** `definition_judge_system.txt` with an explicit equivalence rule plus two
   schema-stable illustrative examples, as a backstop for *near*-identical cases the
   normalization will not catch (reordered clauses, trivial synonyms).
3. A small **labeled eval set + harness**, used as a hermetic regression guard — with the
   real-corpus signal as the primary acceptance gate (see Measurement).

Out of scope / deferred:

- **`canonicalize_term` rewrite** (plural-strip removal, case handling). Deferred to a
  follow-up gated on the eval/telemetry actually showing distinct-term over-merge. See
  "Deferred".
- **Alias-aware grouping** (`Board of Directors` ↔ `the "Board"`). The 2026-05-15 panel's
  first-listed canonicalizer defect; needs an extraction change. Remains open and is
  named here so the gap is explicit.
- **Org/corpus scoping** — item #2, parked.
- Any change to the strict LLM payload schema, extraction, or the contradiction path.

## Architecture

Mirrors the established deterministic-short-circuit precedent: `_try_numeric_short_circuit`
(`pipeline.py`), which runs **before** the judge call and emits a fixed-confidence machine
verdict with its own label (ADR-0005). Architect trace confirmed every detail below
against the code.

### 1. `definitions_equivalent` — pure predicate (`check/definition_terms.py`)

```python
def definitions_equivalent(a_text: str, b_text: str) -> bool:
    """True if two definition texts are equal after normalization."""
```

Normalization, applied to each text, in this exact order:

1. `casefold()`.
2. Collapse every run of Unicode whitespace to a single space (`" ".join(s.split())`).
3. Strip leading/trailing characters in the set `string.punctuation` ∪ `_QUOTE_CHARS`
   (the smart-quote tuple already defined in this module). **Internal** punctuation is
   left untouched — a mid-string comma that changes scope must keep the texts unequal.

Equivalence is exact string equality after (1)–(3). Pure, dependency-light (`string`
stdlib only), no I/O. Lives beside `canonicalize_term`; its docstring cross-references
that function and states the deliberate difference (this is **case-insensitive** body
comparison; `canonicalize_term` is the term-grouping key and is unchanged here).

### 2. Short-circuit at the checker layer (`check/definition_checker.py`)

`DefinitionChecker.find_inconsistencies` is the single site every formed pair flows
through, immediately before `self._judge.judge(a, b)`. Insert: if
`definitions_equivalent(a.assertion_text, b.assertion_text)`, yield a machine-built
`DefinitionJudgeVerdict` and **skip the judge call**:

```python
DefinitionJudgeVerdict(
    assertion_a_id=a.assertion_id,
    assertion_b_id=b.assertion_id,
    verdict=DEFINITION_CONSISTENT_AUTO,          # distinguishable, see §3
    confidence=1.0,
    rationale="Definitions textually identical after normalization (machine-resolved, no LLM call).",
    evidence_spans=[],
)
```

Because the short-circuit sits **above** the judge, it applies uniformly to both
`LLMDefinitionJudge` and `FixtureDefinitionJudge` — no test/prod divergence. Deferring the
canonicalizer leaves this fully correct: grouping decides *which* pairs exist; the
short-circuit only decides *how an already-formed pair resolves*. The two are orthogonal,
and the short-circuit can only *remove* LLM calls on equivalent-text pairs — it can never
manufacture a pair or suppress a genuinely `definition_divergent` verdict (divergent texts
fail the equality test).

### 3. Distinguishable machine verdict (`check/providers/definition_base.py`)

Add a standalone module constant — **not** folded into the shared `DefinitionVerdictLabel`
Literal that `DefinitionJudgePayload` reuses, so the strict LLM-payload vocabulary stays
narrow (the module's stated goal):

```python
DEFINITION_CONSISTENT_AUTO = "definition_consistent_auto"
```

Type the dataclass field as `DefinitionVerdictLabel | Literal["definition_consistent_auto"]`.
Persistence is free: `record_definition_finding` already writes every verdict verbatim to
`findings.judge_verdict`, a free `TEXT` column with **no CHECK constraint** — so the auto
label is greppable in the audit DB with **no migration**. It auto-excludes from reports
because `_append_definition_section` filters on `verdict="definition_divergent"`, and it is
**not** added to `DEFINITION_INCONSISTENCY_VERDICTS`, so it never counts as a finding.

### 4. Run visibility (`pipeline.py`)

Add an `n_definition_short_circuited` counter to `_run_definition_pass`'s return and to
`CheckResult`, incremented when the short-circuit fires. Surfaced in the run-summary log —
the cheapest way to show how many LLM calls were saved and how often the deterministic gate
applied. (No analog to "pairs suppressed" is needed: grouping is unchanged, so no pair that
formed before stops forming.)

### 5. Judge prompt tightening (`check/prompts/definition_judge_system.txt`)

Append, in the verdict-rules region, this rule verbatim:

> If the two definitions are identical or differ only in whitespace, punctuation,
> capitalization, or cross-reference numbering, the verdict is `definition_consistent`.

Then append two illustrative examples (the *identical* case never reaches the LLM — the
short-circuit catches it — so the consistent example illustrates **rewording**, the real
judgment the model must make). Exact text to add:

> Examples:
>
> - Term "Quorum". A: "a majority of the directors then in office". B: "more than half
>   of the directors currently serving." → `definition_consistent` (same threshold,
>   reworded).
> - Term "Affiliate". A: "any entity controlling, controlled by, or under common control
>   with the Company". B: "any entity that owns more than 50% of the Company's voting
>   stock." → `definition_divergent` (B narrows "control" to a >50%-ownership test; A is
>   broader).

These illustrate the existing verdict vocabulary only. **No change** to
`DefinitionJudgePayload`, `DefinitionVerdictLabel`, or the user-prompt template.

### 6. ADR

Add a 1–2 sentence "definition extension (consistent-polarity)" note to
`docs/decisions/0005-numeric-short-circuit.md` Consequences: the same deterministic-gate
pattern now also resolves *consistent* definition pairs (inverse polarity to the numeric
contradiction case), via the `definition_consistent_auto` verdict string. No standalone ADR
— no new provider, table, or seam.

## Data flow

```text
definition pass (grouping UNCHANGED):
  group by canonicalize_term(term)          # unchanged
  enumerate pairs within groups             # unchanged
  for each pair, in DefinitionChecker.find_inconsistencies:
      if definitions_equivalent(a.text, b.text):
          yield definition_consistent_auto  # NEW — no LLM call, n_short_circuited++
      else:
          judge.judge(a, b)                 # system prompt now tightened (§5)
  record_definition_finding(...)            # every verdict persisted (existing)
  report renders definition_divergent only  # auto + consistent + uncertain filtered out
```

## Testing (hermetic, CI-safe — default mark)

- `definitions_equivalent`: identical → True; whitespace-only / case-only /
  surrounding-punct-only differences → True; a genuine wording difference → False; a
  **mid-string comma** that changes scope → False (guards the lossy-normalization risk).
- Short-circuit at checker layer: build a `DefinitionChecker` with a **stub judge whose
  `judge()` raises**; a corpus with two identical same-term definitions yields one
  `definition_consistent_auto` verdict and the stub is never called; two same-term
  *divergent* definitions DO call the stub.
- `n_definition_short_circuited` increments correctly.
- Report/finding filtering: a `definition_consistent_auto` verdict is persisted, is not
  rendered in the report, and is not counted in `n_findings`.
- Hermetic slice of the eval set: the `identical` rows are asserted deterministically
  (resolved by the short-circuit, no LLM).
- `canonicalize_term` and its existing test table are **untouched** (the canonicalizer is
  deferred), so no regression sweep there.

## Labeled eval set + harness (`benchmarks/definition_eval/`)

Mirrors `benchmarks/targeted_eval/`. Used as a **regression guard**, not the primary
precision gate (see Measurement).

- `pairs.jsonl` — ~32 pairs, one JSON object per line:
  `{"term", "def_a", "def_b", "label": "consistent|divergent", "category"}`.
  Categories: `identical` (~6, deterministic/hermetic), `reworded_consistent` (~8, LLM →
  consistent), `scope_divergent` (~8), `threshold_divergent` (~5),
  `inclusion_exclusion_divergent` (~5). **Divergent rows are seeded from REAL flagged
  pairs** mined out of the bylaws / credit-agreement corpus findings (not freshly drafted
  prose), so the set reflects messy real divergences rather than the prompt's own idiom.
  The operator reviews/corrects all labels before they are treated as ground truth.
- `harness.py` (+ `__init__.py`) — groups inputs, applies the short-circuit, runs
  `LLMDefinitionJudge` on the rest, reports accuracy + precision/recall **by category** and
  prints each miss with its texts. **`live`-marked**; run manually with an API key.
- **Baseline capture:** before any code change, run the harness on `main` and save
  per-category numbers to `benchmarks/definition_eval/baseline.json`. "No regression" is
  defined against that file (see Measurement). The `identical` rows are also added as the
  hermetic deterministic assertion above.

## Measurement / acceptance

Trust order (per the eval note: no ground truth exists; the real corpus is the only
non-synthetic signal):

- **Primary gate — real corpus.** Re-run the bylaws corpus before/after; report the
  `definition_divergent` rate move, and **human spot-check a sample of surviving
  `definition_divergent` findings** to confirm the change removed false positives without
  hiding real divergences. Acceptance: the identical-text false positives disappear and no
  spot-checked real divergence flips to consistent.
- **Production signal — reviewer telemetry.** Wire `audit/eval.py::compute_detector_precision`
  (which already mines `reviewer_verdicts` for per-detector precision) into the measurement
  story: capture the current `definition_inconsistency` precision as a baseline, re-read it
  after analysts review a batch on a real corpus. This is the only non-circular *production*
  precision number available.
- **Regression guard — labeled set.** `harness.py` vs `baseline.json`: recall on each
  divergent category must be `>=` baseline; `identical` + `reworded_consistent` accuracy
  must improve or hold. This guards against future regressions; it is explicitly **not** the
  primary precision claim, because the set is synthetic and LLM-graded.

## Deferred (gated on evidence from the above)

- **`canonicalize_term` rewrite.** If the eval/telemetry shows real distinct-term
  over-merge (e.g. `"Loan"`/`"Loans"` producing false divergent findings), revisit:
  drop the plural-strip, and choose the case policy deliberately — the review showed
  *case-sensitive* grouping is a silent recall regression, so a recall-safe `casefold()`
  key (preserving the display term) is the likely correct form, **not** the case-sensitive
  version. Add the observability counter (singleton-group / suppressed-pair) at that time.
- **Alias-aware grouping** — the panel's primary canonicalizer defect, needs an extraction
  change; its own spec when scheduled.

## Risks / known limitations

- **`definitions_equivalent` is lossy by design** (casefold + surrounding-punct strip):
  it catches whitespace/case/edge-punct equivalence only. Semantically-equivalent-but-
  reworded definitions correctly fall through to the LLM (handled by §5, measured by the
  eval set). The mid-string-comma test guards against over-broad equivalence.
- **Eval-set label quality** is the regression-guard ceiling — hence the operator review
  gate and the real-corpus primary gate.
- This change **only fixes the identical-text symptom**; it does not address corpus
  composition (item #2) or alias coreference. Those remain the larger precision levers and
  are explicitly out of scope.

## Roadmap bookkeeping

On completion, move item #1 from "Eval findings & next levers" to the Completed section of
`futureplans.md`, noting that the canonicalizer rewrite and alias-aware grouping remain
deferred (with the evidence gate for revisiting the canonicalizer). After this lands, the
parked item-#2 spec resumes.
