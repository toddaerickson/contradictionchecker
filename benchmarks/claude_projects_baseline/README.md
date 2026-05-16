# Claude-Projects baseline

The "lazy baseline" the triple-expert panel (2026-05-15) ruled the contradictionchecker pipeline must outperform to earn its complexity. Operator does steps 1–3 by hand in Claude.ai; the harness in this directory owns the ingestion, dedup, labelling-CSV scaffolding, and comparison metrics.

## When to run this

After Phase 0 PR1 (`consistency-check eval`) and PR2 (targeted-eval) ship, before any Phase 1 prompt/extractor changes. The three signals (reviewer-verdict mining, targeted-eval, this baseline) are the precondition for deciding whether to keep, fix, or kill the pairwise contradiction core. See the stop-rule at the end of this file.

## Protocol

### 1. Create the Project (one-time per corpus)

In Claude.ai (https://claude.ai/projects):

1. New Project. Name it after the corpus (e.g. `loan_alpha_baseline`).
2. Upload all corpus files to Project Knowledge. The current size limit is roughly 1M tokens; corpora larger than that exceed this baseline's capacity and the pipeline wins by default.
3. Paste the contents of `prompts/system.txt` into the Project's **Custom Instructions** field.

### 2. Run the baseline three times

Three identical chats inside the same Project, each with a fresh chat session. **Do not vary the prompt between runs** — variance is the point.

- For each run, open a new chat in the Project.
- Paste the contents of `prompts/user.txt` as the first message.
- Wait for the JSON response.
- Save the response into a file at `benchmarks/claude_projects_baseline/runs/<corpus>_run_<n>.json` shaped like this:

```json
{
  "run_id": "loan_alpha_run_1",
  "model": "claude-opus-4-7",
  "corpus_name": "loan_alpha",
  "timestamp": "2026-05-15T14:30:00Z",
  "findings": [ /* paste the JSON array Claude returned verbatim */ ],
  "input_tokens": 142000,
  "output_tokens": 3200,
  "cost_usd": 2.18
}
```

Token counts and cost are optional but enable the cost-per-real-finding metric. Claude.ai's web UI doesn't expose per-chat token counts directly; check your billing page after the runs and divide by 3 for a rough per-run estimate, or skip these fields entirely. The harness reports `cost_per_real_finding = None` when costs are absent rather than failing.

If Claude wraps the JSON in a `​```json … ​```` ` code fence even though the prompt asks it not to, the loader tolerates it — paste verbatim.

### 3. Dedupe + emit the labelling CSV

```sh
uv run python -m benchmarks.claude_projects_baseline.harness dedupe \
    benchmarks/claude_projects_baseline/runs/loan_alpha_run_1.json \
    benchmarks/claude_projects_baseline/runs/loan_alpha_run_2.json \
    benchmarks/claude_projects_baseline/runs/loan_alpha_run_3.json \
    --out benchmarks/claude_projects_baseline/labels/loan_alpha_to_label.csv
```

This produces a CSV with one row per *unique* finding across the three runs (deduped by content-addressed `finding_id`). Columns: `finding_id`, `doc_a`, `span_a`, `doc_b`, `span_b`, `type`, `confidence`, `rationale`, `label`, `notes`.

### 4. Hand-label

Open the CSV in your spreadsheet of choice. Fill in the `label` column for each row:

| label | meaning |
|---|---|
| `real` | A genuine contradiction a reviewer would act on |
| `false_positive` | Not a real contradiction (rule/exception, scope mismatch, paraphrase, etc.) |
| `dismissed` | Skip — not eval-relevant (e.g. flagged a definition, not a contradiction) |

Skip rows you're unsure about — leave the `label` cell blank. The harness ignores blank-label rows for precision.

Notes column is for your own notes; the harness round-trips it but does not use it.

### 5. Compute metrics

```sh
uv run python -m benchmarks.claude_projects_baseline.harness compare \
    benchmarks/claude_projects_baseline/runs/loan_alpha_run_1.json \
    benchmarks/claude_projects_baseline/runs/loan_alpha_run_2.json \
    benchmarks/claude_projects_baseline/runs/loan_alpha_run_3.json \
    --labels benchmarks/claude_projects_baseline/labels/loan_alpha_to_label.csv \
    --json-out benchmarks/claude_projects_baseline/metrics/loan_alpha.json \
    --markdown-out benchmarks/claude_projects_baseline/metrics/loan_alpha.md
```

(Explicit file arguments avoid shell-glob surprises on systems where unmatched globs fail. The harness creates any missing parent directories under `--json-out` / `--markdown-out` on its own.) The markdown summary is the artefact you paste into a comparison PR alongside the pipeline's `consistency-check eval` output and the targeted-eval harness output.

## What the three numbers mean

| Number | Source | Read as |
|---|---|---|
| **precision_intersection** | findings present in *all three* runs, labelled `real` / `(real + false_positive)` | The lazy baseline's *floor* precision. Anything the baseline saw three times. |
| **precision_union** | every unique finding across all three runs | The lazy baseline's *ceiling* precision. Includes ideas Claude only had once. |
| **agreement_ratio** | intersection / union | How stable the baseline is. Low ratio = high run-to-run variance, which means you need more runs for stable comparison. |
| **cost_per_real_finding** | sum of `cost_usd` / count of `real` labels | The dollar baseline you measure the pipeline against. |

## Seeded-contradiction recall (optional but recommended)

The semantic-analyst expert's 2-hour labelling protocol includes 30 hand-injected contradictions you produce by editing two real docs in your corpus (flip a number, swap a deadline, change `shall` to `may`). Drop those edited docs into the Project before running the baseline. After labelling, count how many of the 30 seeds appear in the union — that's your recall. The harness doesn't automate this match (too easy to overfit); a 30-row spreadsheet you fill in by hand is the right tool.

## Stop-rule (adopted 2026-05-15)

Kill the pairwise contradiction core entirely if, on at least one real corpus:

- `precision_intersection` of this baseline lands within **10 percentage points** of the pipeline's precision (from `consistency-check eval` and targeted-eval), **AND**
- this baseline catches **≥ 80%** of the seeded contradictions that the pipeline catches.

At that point: keep the definition-inconsistency detector, build the prompt-cached `WholeCorpusJudge` provider (the programmatic version of this baseline, with audit trail + idempotency), and redirect new pairwise work to cross-reference / obligation-date detectors.

## What this baseline does *not* test

- **Corpora over ~1M tokens** — they don't fit. Pipeline wins by default on those.
- **Recurring scans** — Claude.ai chats have no idempotency / audit-trail. Pipeline wins by default.
- **Cross-document contradiction recall** — Claude in one chat sees the corpus as a whole; this is its strongest mode and where it tends to *outperform* the pipeline, not the other way around. Read this as the baseline's natural advantage, not a flaw.
