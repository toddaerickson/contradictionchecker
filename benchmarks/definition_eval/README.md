# Definition-detector evaluation

Measures the default definition-inconsistency detector's precision / recall / F1
against a labelled set of term→definition pairs. Three steps:

## 1. Mine candidate pairs from a real corpus

```sh
uv run python -m benchmarks.definition_eval.mine_pairs \
    --db data/store/assertions.db --corpus atkins \
    --out benchmarks/definition_eval/candidates_atkins.jsonl
```

Surfaces same-term definition pairs (cross-document first), each `label: ""`.

## 2. Label them (human verdict)

```sh
uv run python -m benchmarks.definition_eval.label \
    --in benchmarks/definition_eval/candidates_atkins.jsonl \
         benchmarks/definition_eval/candidates_fcs.jsonl \
    --out benchmarks/definition_eval/pairs_labeled.jsonl
```

Opens a local page (`http://127.0.0.1:8011`) showing one pair at a time with a
word-diff highlight. Keys: **c** = consistent, **d** = divergent, **s** = skip,
**← / →** = move. Autosaves; resumable; relabel anytime. `divergent` = the two
definitions genuinely conflict for the same term (the positive class).

## 3. Score the detector

```sh
uv run python -m benchmarks.definition_eval.harness \
    --pairs benchmarks/definition_eval/pairs_labeled.jsonl \
    --metrics benchmarks/definition_eval/baseline.json
```

Prints per-category accuracy and overall P/R/F1 (divergent = positive), and the
list of misses. Needs a provider key in `config.yml` (it runs the real judge).

## Data hygiene

`pairs.jsonl` is a small **synthetic** sanity set (committed). Mined candidates
and human labels are derived from **private corpora** (e.g. the Atkins
agreement), so `candidates_*.jsonl`, `pairs_labeled.jsonl`, and `review.jsonl`
are git-ignored — publish the resulting **numbers** (`baseline.json` metrics),
not the underlying document text.
