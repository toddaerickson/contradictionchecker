# Definition-inconsistency detector — design

**Date:** 2026-05-15
**Status:** Approved by user, pending implementation plan
**Tracks:** [`futureplans.md`](../../../futureplans.md) item #20 (consistency-detector family)
**Relates to:** [ADR-0008](../../decisions/0008-persona-aware-analysis.md) (persona-aware analysis: each detector gets its own ADR)

## Summary

Build the second member of the consistency-detector family: a detector that
flags **divergent definitions of the same term across the corpus**. Reuses
CrossCheck's ingest, embedding, judge, and audit infrastructure; introduces
two migrations (`assertions` gains `kind` / `term` / `definition_text`;
`findings` gains `detector_type`) and one new detection stage.

The contradiction detector remains the default; the definition detector runs
alongside it under the existing `check` pipeline.

## Motivation

`futureplans.md` item #20 motivates an expanded detector family because the
contradiction judge structurally cannot find issues that aren't pair-shaped
contradictions. A borrower's counsel, an underwriting reviewer, or an HR
professional reading a policy set cares about:

- Defined terms used inconsistently across documents (this build).
- Obligations promised in one document and unaddressed in another (the gap
  detector — future).
- Definition ↔ usage drift within the corpus (future, parked here as
  "flavor B").

This build covers the first item only.

## Scope

### In scope (flavor A only)

A single new detector that emits findings when **two definition assertions
referring to the same term diverge in meaning**.

Example finding:

> **Term:** "Material Adverse Effect"
> **Doc A (term-sheet.pdf):** "means a material adverse effect on the
> Borrower's business, financial condition, or operations."
> **Doc B (credit-agreement.pdf):** "means an effect that materially
> impairs the Borrower's ability to perform its obligations under this
> Agreement."
> **Verdict:** `definition_divergent` — Doc A scopes MAE broadly to
> business condition; Doc B narrows it to performance ability.

### Out of scope

- **Flavor B — usage-vs-definition drift.** Parked as future work; revisit
  once flavor A has shipped and produced real findings on user corpora.
- **Flavor C — undefined-term-used-as-if-defined.** Moved to the planned
  gap detector; it is structurally a "presence vs absence" check, not a
  pairwise definition comparison.
- **Persona-aware ranking of definition findings.** Per ADR-0008, persona
  is a view layer. It can map onto this detector later by toggling
  `detector_type ∈ {…}` per persona; no detector-side work required.

## Decisions confirmed during brainstorm

1. **Narrow build** — reuse `findings` with a `detector_type` discriminator
   column; do not introduce a shared `issues` supertype. Defer that
   refactor to when gap/ambiguity detectors arrive and force it.
2. **Flavor A only** — definition ↔ definition. Flavor B and C deferred as
   above.
3. **LLM-based extraction**, folded into the existing atomic-fact
   extractor. The motivating corpora (underwriting memos, HR policies,
   organic policy memos) define terms in free-form prose ("By 'stabilized'
   we mean 90% occupancy held for three consecutive months"; "An eligible
   employee is someone who has completed 90 days"). Syntactic patterns
   would miss 30–60% of those. LLM extraction handles them naturally and
   adds only output tokens — no new API calls — because the extractor is
   already running on every chunk.

## Data model

### Schema changes

Two migrations:

```sql
-- 0007_assertion_kind.sql
    ALTER TABLE assertions ADD COLUMN kind TEXT NOT NULL DEFAULT 'claim';
    -- Values: 'claim' (current behavior), 'definition' (new).
    -- 'definition' rows additionally populate the term/definition_text
    -- columns introduced below.
    ALTER TABLE assertions ADD COLUMN term TEXT;
    ALTER TABLE assertions ADD COLUMN definition_text TEXT;
    CREATE INDEX idx_assertions_kind ON assertions(kind);
    CREATE INDEX idx_assertions_term ON assertions(term) WHERE kind='definition';

-- 0008_finding_detector_type.sql
    ALTER TABLE findings ADD COLUMN detector_type TEXT NOT NULL DEFAULT 'contradiction';
    -- Values: 'contradiction' (current), 'definition_inconsistency' (new).
    CREATE INDEX idx_findings_detector ON findings(detector_type);
```

Both columns default to current behavior so existing rows are valid.
Migration loader picks up new files by filename per the project rule.

### Dataclass changes (`extract/schema.py`)

`Assertion` gains three optional fields:

```python
@dataclass(frozen=True, slots=True)
class Assertion:
    ...existing fields...
    kind: str = "claim"            # "claim" | "definition"
    term: str | None = None        # populated only when kind == "definition"
    definition_text: str | None = None  # populated only when kind == "definition"
```

`assertion_id` continues to be `hash_id(doc_id, assertion_text)` —
unchanged. For definitions, `assertion_text` is the full sentence/clause
containing the definition; `term` and `definition_text` are extracted
sub-fields.

### Finding shape

Reuses the existing `findings` row. For a `detector_type =
'definition_inconsistency'` row:

- `assertion_a_id`, `assertion_b_id` → the two definition assertions.
- `judge_verdict ∈ {'definition_consistent', 'definition_divergent',
  'uncertain'}`.
- `gate_score`, `nli_*` → may be null. NLI is contradiction-tuned and
  unhelpful for definition comparison; the gate stage shortcuts straight
  to the judge for definition pairs sharing a `term`.
- `judge_rationale`, `evidence_spans_json`, `judge_confidence` → as
  today.

## Extraction strategy

### One extractor, two output kinds

Extend the existing `extract/atomic_facts.py` prompt to emit both atomic
facts and definitions in one structured response. Approximate prompt
shape:

```text
Extract two kinds of items from the passage below:

1. atomic_facts: standalone, decontextualised claims (existing behavior).
2. definitions: statements where the author defines a term, whether
   formally ("X means …", "X shall mean …") or informally ("by X we
   mean …", "an eligible Y is …", "for purposes of this policy, X
   refers to …", or an "implicit" definition that describes what X is
   in narrative form).

Return JSON:
{
  "atomic_facts": [string, ...],
  "definitions":  [{"term": string, "definition_text": string,
                    "containing_sentence": string}, ...]
}
```

The extractor returns both lists per chunk; the pipeline persists atomic
facts as `kind='claim'` assertions (unchanged) and definitions as
`kind='definition'` assertions with `term` / `definition_text` populated.

### Quality fallback

If A/B testing on a representative corpus (loan package + HR handbook +
underwriting memo) shows the combined prompt degrades atomic-fact
extraction by more than ~5% (precision or recall vs the current prompt),
split into a second LLM call per chunk. Same storage shape; only the
extractor module changes. This decision is deferred to the implementation
plan after running the A/B.

### Fixture extractor

`FixtureExtractor` in `tests/conftest.py` gains a `definitions_by_text`
map alongside its existing `facts_by_text` map, so hermetic tests can
inject definitions deterministically. No external dependency.

## Detection pipeline

A new `check/definition_checker.py` module:

```python
class DefinitionChecker(Protocol):
    def find_inconsistencies(
        self, definitions: Sequence[Assertion]
    ) -> Iterable[DefinitionFinding]: ...
```

Stages (modeled on the contradiction pipeline, but simpler):

1. **Group by term.** Canonicalise terms: case-fold, strip leading "the
   ", drop possessive `'s`, drop surrounding quotes. Group definition
   assertions by the canonical term.
2. **Skip singletons.** If a term has only one definition assertion in
   the corpus, no comparison is possible — skip. (A future "undefined
   term referenced as defined" check belongs to the gap detector.)
3. **Pair within group.** For each term with ≥ 2 definitions, enumerate
   all unordered pairs.
4. **Judge each pair.** Reuse the existing `JudgeProvider` Protocol with
   a new prompt (`check/prompts/definition_judge_system.txt` and
   `definition_judge_user.txt`) that asks: "Do these two definitions of
   the same term express the same meaning, or do they materially
   diverge?" Output schema mirrors `JudgePayload` with verdict values
   `definition_consistent` / `definition_divergent` / `uncertain`.
5. **Persist** findings with `detector_type='definition_inconsistency'`
   into the existing `findings` table.

### Why the NLI gate is bypassed for this stage

The DeBERTa NLI gate is fine-tuned on contradiction detection between
declarative claims. Two definitions of "Material Adverse Effect" don't
"contradict" in the entailment sense — they describe different scopes of
the same concept. NLI returns noise on this input distribution. The
"gate" for the definition stage is the term-grouping step above:
definitions only get judged if they share a canonical term. This is
cheaper than the NLI gate and more precise for this question.

### Pipeline integration

`pipeline.check()` runs definition detection **after** the contradiction
pipeline, in the same `run_id`. Both stages contribute findings to the
same run. CLI flag `--no-definitions` lets a user opt out (mirrors
`--deep` from v0.2). Default: enabled.

## Reporting / surfacing

### Markdown report (`audit/report.py`)

Add a new top-level section after "Contradictions": `## Definition
inconsistencies`. Each finding renders as:

```markdown
### "<canonical-term>"

- **<doc A title>** (<doc A path>): <definition_text from assertion A>
- **<doc B title>** (<doc B path>): <definition_text from assertion B>

**Verdict:** definition_divergent — <judge_rationale>
```

Grouped by term, sorted by document-pair frequency descending.

### Web UI (`web/`)

A new `cc__definitions.html` partial renders the same finding shape as
the Contradictions tab. Add a top-level nav entry "Definitions" alongside
"Contradictions" / "Documents" / "Assertions" / "Stats" / "Ingest".

Counter additions on the Stats tab: `n_definition_findings`,
`n_definition_pairs_judged`. Backed by `detector_type`-filtered queries
against `findings`.

### What `uncertain` means here

Same semantics as the contradiction detector: stored in the audit DB,
**hidden** from reports and the web UI. Threshold tuning may surface them
during eval but they don't reach the user.

## Migration plan

Order:

1. Add migrations 0007 + 0008 (schema only; no behavior change).
2. Update `Assertion` dataclass with new fields (defaults preserve
   existing call sites).
3. Update `AssertionStore` insert / fetch to round-trip `kind`, `term`,
   `definition_text`.
4. Update the atomic-fact extractor prompt + response schema.
5. Add `FixtureExtractor` definition support and round-trip tests.
6. Add `DefinitionChecker` + `FixtureDefinitionChecker` + prompts.
7. Wire into `pipeline.check()` behind the new flag.
8. Add report + web rendering.
9. Update CLI `--no-definitions` flag.
10. Write ADR-0009 capturing the final decisions (per item #20: each new
    detector gets its own ADR).

Existing audit DBs survive untouched: migrations default the new columns
to current-behavior values; old findings render as contradictions; old
assertions render as claims.

## Testing strategy

Hermetic tests (default mark):

- `test_definition_extraction_round_trip`: `FixtureExtractor` emits a
  definition; pipeline persists it with `kind='definition'`, `term`
  populated, `assertion_id` stable across runs.
- `test_term_canonicalization`: "the Borrower", "Borrower",
  "Borrowers", "\"Borrower\"" all collapse to the same canonical
  `term` for grouping.
- `test_definition_singleton_skipped`: a term with one definition
  produces zero pairs and zero findings.
- `test_definition_pair_judged`: two diverging definitions of the same
  term, fed through `FixtureDefinitionChecker`, produce one finding with
  `detector_type='definition_inconsistency'`.
- `test_existing_contradiction_pipeline_unaffected`: a corpus with no
  definitions produces the same findings count as before this change.
- `test_migrations_idempotent`: re-running 0007/0008 on an upgraded DB
  is a no-op.

`slow`-marked tests:

- `test_atomic_fact_prompt_does_not_regress`: A/B the combined prompt
  vs current prompt on the fixture corpus; assert atomic-fact
  precision/recall doesn't drop more than 5%. Gates the "should we
  split into two LLM calls?" decision.

`live`-marked tests:

- `test_definition_detection_on_loan_fixture`: real Anthropic call
  against a hand-curated 2-doc fixture with one known MAE divergence;
  asserts the divergence is found with verdict `definition_divergent`.

## Decisions deferred to the implementation plan

1. **Combined extractor prompt vs two-pass.** Decide after the A/B test.
2. **Exact judge prompt wording.** Iterate during implementation; final
   wording captured in ADR-0009.
3. **Same-document definition pairs.** Probably in scope (a single doc
   can define a term twice and diverge). Confirm with one fixture during
   implementation; if false-positive prone, restrict to cross-doc pairs.
4. **Persona filter wiring.** Out of scope here; tracked separately under
   ADR-0008.
5. **CLI naming for the opt-out flag.** `--no-definitions` is a
   placeholder; the plan can pick between that and `--skip-definitions`
   / `--detectors contradiction` style.

## Future extensions (explicitly deferred)

- **Flavor B — definition ↔ usage drift.** Adds a new finding shape:
  `(definition_assertion, usage_assertion, verdict)`. Requires a
  "usage extraction" pass — finding every occurrence of a defined term
  in a context that is not itself the definition. Larger build.
- **Gap detector.** Will require the "issue supertype" refactor flagged
  in the narrow-vs-bridging-vs-forward-looking choice. When that
  refactor lands, this detector's findings migrate naturally.
- **Persona-aware filters.** Per ADR-0008, surfacing the same findings
  with different rankings per persona.

## Acceptance

This detector ships when:

- Migrations 0007 + 0008 applied and idempotent.
- Atomic-fact extractor emits both `claim` and `definition` items
  without > 5% regression on existing claim extraction.
- `DefinitionChecker` produces correct findings on the live fixture
  loan-doc pair.
- Markdown report and web UI render definition findings in their own
  section.
- All hermetic + `e2e_fixture` tests pass; CI gate green.
- ADR-0009 written and merged.
