# Targeted-eval set

120 hand-crafted pairs probing three specific failure modes of the contradiction judge that the triple-expert panel (2026-05-15) flagged as load-bearing for real corporate corpora. Used to measure the **delta** between the current pipeline and any prompt/architecture changes.

## Why this exists

CONTRADOC is balanced and synthetic in the wrong way: it doesn't isolate modal logic, defeasible override, or temporal scope. The semantic-analyst expert put expected real-corpus precision at 0.2%–1% (policy text) to ~15% (tight loan packages) once base rates are folded in — the 89% benchmark figure does not survive. This set is the cheapest path to a real signal on the three relations the panel agreed matter most.

The pairs are designed so the **bucket totals matter**, not individual labels. Use them as paired before/after measurements when changing prompts, extractor behaviour, or judge providers.

## Bucket taxonomy

| Bucket | n | What it probes |
|---|---:|---|
| `rule_exception` | 40 | Defeasible override — general rule + specific carve-out. All labelled `not_contradiction`/`override`. The linguist panel identified this as the dominant FP engine on policy/loan text. |
| `modal_divergence` | 40 | Modal-logic handling — `shall`/`may`/`must`/`will`/`shall not`/`may not` permutations. 25 `not_contradiction` (compatible permissions and obligations), 15 `contradiction` (incompatible obligations / prohibition-of-required-action). |
| `temporal_scope` | 40 | Temporal-scope mismatch — same predicate at different effective dates, fiscal years, or phases. All labelled `not_contradiction`/`scope_mismatch`. The judge currently has no document-date metadata, which is the structural cause of this failure mode. |

Total: 120 pairs. 105 `not_contradiction` + 15 `contradiction`. The skew is intentional — the panel's hypothesis is FP rate on non-contradictions, not recall of true contradictions.

## Schema

```json
{
  "pair_id": "rule_exc_001",
  "bucket": "rule_exception",
  "assertion_a": "...",
  "assertion_b": "...",
  "ground_truth": "not_contradiction",
  "relation": "override",
  "notes": "..."
}
```

- `ground_truth` ∈ `{contradiction, not_contradiction, uncertain}` — matches the current three-value judge enum so a direct comparison works without translation.
- `relation` ∈ `{contradiction, contrariety, override, entailment, scope_mismatch, independent}` — finer-grained label per the logician expert's seven-relation taxonomy. Used for per-relation slicing in the harness; not consumed by the judge.
- `notes` documents *why* a pair is in the set so a future reviewer can decide whether to keep, edit, or drop it.

## Running the harness

```sh
# Real Anthropic judge (consumes API credits)
uv run python -m benchmarks.targeted_eval.harness \
    --output metrics_before.json \
    --markdown summary_before.md

# After applying a prompt or extractor change, re-run
uv run python -m benchmarks.targeted_eval.harness \
    --output metrics_after.json \
    --markdown summary_after.md

# Then diff the two markdown files in your PR
```

Skip the NLI gate by design (see harness docstring). Most pairs in this set have no surface contradiction that DeBERTa would flag, so gating would silently drop them and hide the failure mode under test.

For programmatic testing (no API calls), construct a `FixtureJudge` keyed on the pair texts and pass it to `run_targeted_eval`. See `tests/test_targeted_eval.py` for the pattern.

## Output

The harness emits two artefacts:

- `metrics.json` — structured `TargetedResult` with `n_pairs`, `overall_accuracy`, per-bucket `BucketMetrics`, and the flat `predictions` list. Diff-friendly across runs.
- `summary.md` (optional) — human-readable per-bucket table. Drop straight into a PR description.

## Extending

To add pairs, append to `pairs.jsonl`. Keep `pair_id` prefixes consistent with the bucket (`rule_exc_NNN`, `modal_NNN`, `temporal_NNN`). The harness validates every row's `bucket`, `ground_truth`, and `relation` against the literals in `harness.py` — typos fail loudly at load time rather than producing silently empty buckets.

When extending, **add pairs from your own corpora** if possible. Synthetic pairs are valuable for isolating a phenomenon, but they don't capture the lexical patterns your judge actually sees in production. The reviewer-workflow UI plus `consistency-check eval` is the complementary signal: pairs the system has actually flagged, labelled by analysts who saw them in context.

## Stop-rule (semantic-analyst expert, adopted 2026-05-15)

Kill the pairwise contradiction detector entirely if, on this set:

- precision ≤ 15% on any bucket after one round of prompt fixes, **AND**
- the Claude-Projects baseline (see PR3) lands within 10 points of this system on precision with ≥ 80% of the seeded-contradiction recall.

At that point: keep the definition-inconsistency detector, build the prompt-cached whole-corpus detector, and redirect new pairwise work to cross-reference and obligation/date extractors.
