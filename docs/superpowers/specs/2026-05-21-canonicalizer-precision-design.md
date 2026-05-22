# Definition canonicalizer + judge precision

**Date:** 2026-05-21
**Status:** Approved (design)
**Roadmap item:** futureplans.md item #1 — definition-judge / canonicalizer precision
(the eval-named highest-leverage lever). Sequenced ahead of item #2 (org grouping),
which is parked at `docs/superpowers/specs/2026-05-21-org-grouping-corpus-warning-design.md`.

## Problem

The 2026-05-21 real-corpus eval found that after the PDF junk filter (#61) removed
extraction noise, the residual "divergent definition" false positives are genuine
precision failures in two places:

1. **Grouping.** `canonicalize_term` (`check/definition_terms.py`) lowercases, strips a
   trailing plural `s`, and strips a leading `the `. The plural-strip merges distinct
   legal defined terms (`"Loan"` vs `"Loans"`), and lowercasing discards the
   capitalization signal that distinguishes a defined term from generic prose. Both
   manufacture spurious pairs.
2. **Judging.** The definition judge has been observed flagging *textually identical*
   definitions as `definition_divergent` (e.g. *"the authorized number of directors of
   the Corporation"* compared against the same string). The system prompt's "be
   conservative" instruction is not preventing this.

This spec raises precision on the definition detector by fixing both, without touching
extraction, the payload schema, or any other detector.

## Scope

In scope (operator-approved):
- Rewrite `canonicalize_term` — drop plural-strip, preserve case.
- Add a deterministic identical/near-identical short-circuit to the definition judge.
- Tighten `definition_judge_system.txt` with an explicit equivalence rule + 1–2
  schema-stable few-shot examples.
- Build a small labeled definition-pair eval set + harness to validate the change.

Out of scope (explicitly deferred):
- **Alias-aware grouping** (`Board of Directors` ↔ `the "Board"`). Needs an extraction
  prompt/schema change; deferred to a follow-up.
- **Org/corpus scoping** — that is item #2, parked.
- Any change to the contradiction (pairwise) path, the payload schema, or extraction.

## Architecture

### 1. Canonicalizer rewrite (`check/definition_terms.py`)

`canonicalize_term(raw: str) -> str` changes in the **precision-safe direction** (group
*less*, consistent with the module's existing "rather miss a group than merge unrelated
terms" docstring):

- **Remove the trailing-plural strip.** Delete the
  `if len(text) > 2 and text.endswith("s") and not text.endswith("ss")` branch entirely.
  `"Loan"` and `"Loans"` become distinct keys.
- **Remove the `.lower()` call — preserve case.** Matching becomes case-sensitive.
  `"Net Income"` and `"Net income"` no longer merge. Both inputs are already
  `kind=="definition"` defined terms, which within a single organization are
  capitalized consistently, so the recall cost is small and accepted.
- **Keep** the surrounding-whitespace trim, the quote-character strip
  (`_QUOTE_CHARS`), and the leading `"the "` strip. Update the leading-article strip to
  be case-insensitive on the article only (`the `/`The `) so case-preservation of the
  term itself is not defeated by an article.
- Update the module docstring to describe the new behavior.

The function stays pure, dependency-free, and in its current home (`definition_terms.py`)
— which is also where item #2's `normalize_org` will live on resume.

**Caller audit (build step):** confirm `canonicalize_term` is imported only by
`check/definition_checker.py`. If any other module (e.g. the contradiction path) calls
it, stop and flag — this change must not perturb the contradiction detector.

### 2. Identical/near-identical short-circuit

New pure helper in `definition_terms.py`:

```python
def definitions_equivalent(a_text: str, b_text: str) -> bool:
    """True if two definition texts are equal after normalization
    (whitespace collapse + casefold + surrounding-punctuation strip)."""
```

Normalization for *this comparison only* (distinct from `canonicalize_term`, which keys
*terms*): collapse internal whitespace to single spaces, strip leading/trailing
punctuation and quotes, `casefold()`. Equivalence is exact string equality after that.

Applied in `LLMDefinitionJudge.judge` (`check/definition_judge.py`), **before** the
provider call: if `definitions_equivalent(a.assertion_text, b.assertion_text)`, return

```python
DefinitionJudgeVerdict(
    assertion_a_id=a.assertion_id,
    assertion_b_id=b.assertion_id,
    verdict="definition_consistent",
    confidence=1.0,
    rationale="Definitions are textually identical after normalization.",
    evidence_spans=[],
)
```

with **no LLM call**. This is deterministic, kills the exact observed failure, and saves
a request. The verdict flows through the existing audit path unchanged. `FixtureDefinitionJudge`
is not modified (fixtures are explicit by design).

### 3. Judge prompt tightening (`check/prompts/definition_judge_system.txt`)

Add, in the verdict-rules region, an explicit equivalence rule as a backstop for
*near*-identical cases the short-circuit's normalization will not catch (reordered
clauses, trivial synonyms):

> If the two definitions are identical or differ only in whitespace, punctuation,
> capitalization, or cross-reference numbering, the verdict is `definition_consistent`.

Plus 1–2 schema-stable few-shot examples appended to the system prompt: one
identical-text pair → `definition_consistent`, one genuine scope-shift pair →
`definition_divergent`. Examples illustrate the verdict vocabulary already in the schema;
**no change** to `DefinitionJudgePayload`, `DefinitionVerdictLabel`, or the user prompt
template.

### 4. Labeled eval set + harness (`benchmarks/definition_eval/`)

Mirrors `benchmarks/targeted_eval/`:

- `pairs.jsonl` — ~36 hand-curated pairs, one JSON object per line:
  ```json
  {"term": "...", "def_a": "...", "def_b": "...", "label": "consistent|divergent", "category": "..."}
  ```
  Categories and rough counts:
  - `identical` (~6) — byte-identical or whitespace/case-only differences → consistent.
  - `reworded_consistent` (~8) — same scope, different wording → consistent.
  - `scope_divergent` (~8) — materially different scope → divergent.
  - `threshold_divergent` (~5) — different numeric thresholds → divergent.
  - `inclusion_exclusion_divergent` (~5) — different inclusions/exclusions → divergent.
  - `distinct_term` (~4) — *different* terms (e.g. `Loan` vs `Loans`, `Director` vs
    `Independent Director`) that the canonicalizer must **not** group; assert no pair forms.
- `harness.py` — groups inputs via `canonicalize_term`, runs `LLMDefinitionJudge` over
  formed pairs, reports accuracy + precision/recall **by category** and prints each miss
  with its texts. `__init__.py` for package import. **`live`-marked** when invoked as a
  test (hits the configured provider); intended to be run manually with an API key, not
  in CI.

The author (Claude) drafts the 36 pairs; the operator reviews/corrects labels before
they are treated as ground truth.

## Data flow (unchanged except where noted)

```
definition pass:
  iter definitions
  group by canonicalize_term(term)          # now case-sensitive, no plural-strip
  enumerate pairs within groups
  for each pair -> LLMDefinitionJudge.judge:
      if definitions_equivalent(a, b): return definition_consistent  # NEW, no LLM
      else: provider.request_payload(system, user)  # system prompt now tightened
  -> findings (divergent only counted/reported)
```

## Testing (hermetic, CI-safe — default mark)

- `canonicalize_term`: `"Loan" != "Loans"`; `"Net Income" != "Net income"`; quote strip;
  whitespace trim; `"the Board"`/`"The Board"` → `"Board"`; idempotence
  (`canonicalize_term(canonicalize_term(x)) == canonicalize_term(x)`).
- `definitions_equivalent`: identical / whitespace-only / case-only / surrounding-punct-only
  → True; genuine wording difference → False.
- `LLMDefinitionJudge` short-circuit: inject a stub `DefinitionJudgeProvider` whose
  `request_payload` raises; assert identical texts return `definition_consistent` and the
  stub is never called; assert non-identical texts DO call the provider.
- `definition_checker` grouping: a corpus with `"Loan"` and `"Loans"` definitions forms
  **no** pair; two same-term divergent definitions still form one pair.
- Regression sweep: audit existing definition tests (e.g. fixtures under
  `tests/fixtures/definition_e2e`) for any reliance on plural-strip/lowercase merging;
  update expectations.
- Hermetic slice of the eval set: the `identical` and `distinct_term` rows are asserted
  deterministically (short-circuit + canonicalizer resolve them without an LLM).

## Measurement / acceptance

- **Primary (gate):** accuracy on `pairs.jsonl` by category, before vs after. Acceptance:
  `identical` and `reworded_consistent` precision improves (these stop being called
  `divergent`) with **no regression** on `scope_divergent` / `threshold_divergent` /
  `inclusion_exclusion_divergent` recall. Run via `harness.py` with the operator's key.
- **Secondary (context, not a gate):** re-run the bylaws corpus; report the
  divergent-rate move. Read only alongside the labeled-set numbers, so a rate drop cannot
  masquerade as progress while hiding lost recall.

## Risks / known limitations

- **Case-sensitivity false negatives.** A legitimately-same term capitalized differently
  across documents (`"Net Income"` vs `"Net income"`) will no longer group. Accepted
  (false-negative-preferring per module philosophy); recorded as a known limitation. If a
  real corpus shows this hurting, a case-insensitive fallback *within an already-grouped*
  set could be added later.
- **Short-circuit normalization is narrow by design.** It catches whitespace/case/punct
  equivalence only; semantically-equivalent-but-reworded definitions fall through to the
  LLM (handled by the tightened prompt, measured by the eval set).
- **Eval-set label quality** is the ground-truth ceiling — hence the operator review gate
  on the drafted pairs.

## Roadmap bookkeeping

On completion, move item #1 from "Eval findings & next levers" to the Completed section
of `futureplans.md`, noting alias-aware grouping remains deferred. No ADR required (no new
provider surface or architectural seam — this is a precision fix to existing components).
After #1 lands, resume the parked item #2 spec against the improved canonicalizer.
