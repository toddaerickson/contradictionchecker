# Benchmarks

## CONTRADOC

`benchmarks/contradoc_harness.py` runs the Stage A NLI + Stage B LLM judge against a normalised CONTRADOC dataset and reports precision / recall / F1 against gold contradiction labels.

The atomic-fact extraction stage is **bypassed** for this benchmark — CONTRADOC labels are per-document-pair, not per-claim. The harness treats each document text as a single assertion-equivalent input to the judge.

### Dataset format

The original CONTRADOC dataset is **not redistributed** with this repository. Convert it to a normalised JSONL with one object per line:

```json
{"pair_id": "...", "document_a": "...", "document_b": "...", "label": "contradiction"}
{"pair_id": "...", "document_a": "...", "document_b": "...", "label": "not_contradiction"}
```

Place the file anywhere accessible and pass the path via `--input`.

### Running

```sh
ANTHROPIC_API_KEY=...
uv run python -m benchmarks.contradoc_harness \
    --input path/to/contradoc.jsonl \
    --output benchmark/metrics.json \
    --sample 50 \
    --nli-threshold 0.5 \
    --judge-provider anthropic \
    --judge-model claude-sonnet-4-6
```

OpenAI is also supported:

```sh
OPENAI_API_KEY=...
uv run python -m benchmarks.contradoc_harness \
    --input path/to/contradoc.jsonl \
    --output benchmark/metrics-openai.json \
    --sample 50 \
    --judge-provider openai \
    --judge-model gpt-4o-2024-08-06
```

### Metrics output

The JSON written to `--output` contains the confusion matrix, precision/recall/F1, the NLI threshold used, and per-pair predictions (gold label, predicted label, judge verdict, judge confidence, NLI p_contradiction). This is enough to redo the scoring with a different threshold post-hoc without re-running inference.

### Pass-bar

The initial bar is **precision ≥ 0.7 on a 50-sample slice**. The hybrid NLI + LLM design was justified largely on the LegalWiz precision claim (~89%); if our implementation falls below 0.7 on a representative sample, the threshold from Step 10 needs re-tuning or the prompt from Step 11 needs iteration.

### What this benchmark does not test

- The atomic-fact extractor (Step 7) is bypassed because CONTRADOC pairs are document-level.
- The candidate-pair gate (Step 9) is bypassed because every pair to be scored is enumerated in the input file.
- The audit logger and report generator (Steps 12, 13) are bypassed; metrics are written directly to JSON.

To exercise the full pipeline end-to-end use `tests/test_e2e.py` instead.
