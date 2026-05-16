# Phase 0 UI plan — zero-CLI eval workflow for analysts

**Goal.** A non-programmer analyst can run all three Phase 0 eval signals and read the results without ever opening a terminal. Today the underlying logic ships (PRs #51, #52, #53) but only as CLI commands. This plan brings them into the existing FastAPI + HTMX web UI so an analyst sees a single browser tab from "drop the corpus in" to "here is the pipeline's real-corpus precision vs. the Claude-Projects baseline."

**Target user.** Solo analyst, novice with Python, comfortable with web forms, file uploads, and clicking buttons. Reads the operator profile in `/home/terickson/.claude/CLAUDE.md`. Should never need to think about:

- virtualenvs / `uv sync` / package installs (handled by Phase 3 packaging, item #14 in `futureplans.md`)
- which subcommand to type
- JSONL or YAML editing
- where output files land

## What works today (no change required)

| Task | Where in UI |
|---|---|
| Upload a corpus | **Ingest** tab — drag-and-drop, file input, `POST /uploads` |
| Run the contradiction scan | Big "Check now" button on Contradictions tab, `POST /runs` (BackgroundTasks) |
| Watch the scan progress | **Stats** tab self-polls every few seconds, partials at `cc__stats_live.html` |
| Review findings (C / F / D) | Inline buttons on every finding row; keyboard shortcuts; persistent undo toast — shipped in PR #50 |
| Drill into a finding | Diff dialog (`/findings/.../diff`, `/multi_party_findings/.../diff`) |
| Skim documents / statements / definitions | Tabs already exist |

The analyst already has the *ingest → check → review* loop in a browser. What they don't have is **eval**: the answer to "did this run produce good findings?"

## What is CLI-only or external today (the gap)

| Signal | Where it lives now | What an analyst has to do today |
|---|---|---|
| Per-detector precision + calibration (PR #51) | `consistency-check eval` | Open a terminal, source env, type a command, read printed table |
| Targeted-eval on 120 hand-crafted pairs (PR #52) | `python -m benchmarks.targeted_eval.harness` | Same; plus needs API key in env, picks model name, manages output file paths |
| Claude-Projects baseline (PR #53) | manual claude.ai chats + CLI `dedupe` + spreadsheet labelling + CLI `compare` | Multi-tool workflow: web (claude.ai) → editor (write JSON files) → terminal (dedupe) → spreadsheet (label) → terminal (compare) |

## Plan

Three blocks (A, B, C). Each block is independently shippable and follows the project's existing Block G pattern (one PR per block, schema-stable where possible, HTMX partials for live updates).

A → B → C is the strict order because **Block A surfaces signal the user already produces** (just by clicking C/F/D), **Block B adds the diagnostic the panel said is the second-cheapest signal**, and **Block C is the largest because it replaces a multi-tool workflow with a single browser surface**.

---

### Block A — "Eval" tab: per-detector precision + calibration

**One PR. Smallest of the three. Renders data that already exists.**

**Adds a new tab** between Stats and Ingest:

```
Contradictions | Definitions | Documents | Statements | Stats | Eval | Ingest
```

**Routes** (new):

- `GET /tabs/eval` — full eval-tab HTML (HTMX-friendly partial when `HX-Request` header is set).
- `GET /tabs/eval/precision` — partial that re-renders just the precision table (so live polling is cheap once findings get reviewed).
- `GET /tabs/eval/calibration?detector=contradiction` — partial for the calibration table; detector selector swaps it.

**Implementation:** thin wrapper around the existing pure functions in `consistency_checker/audit/eval.py`. No new SQL. No new schema. The route opens the store, calls `iter_eval_rows`, `compute_detector_precision`, `compute_calibration`, hands the dataclasses to Jinja.

**New templates**:

- `cc_eval.html` — tab shell + the two tables inline
- `cc__eval_precision.html` — partial for the precision table (matches the existing `cc__stats_*.html` partials pattern)
- `cc__eval_calibration.html` — partial for the calibration table

**Sample-size guard.** Per-detector precision needs ~30 reviewed findings to be meaningful. If `n_reviewed < 30`, the table shows a banner: *"Needs at least 30 reviewed findings for stable precision. Currently: N. Keep reviewing on the Contradictions tab."* Direct link back.

**Empty-state.** Zero reviewed findings: render a three-step "How this works" panel (Run a check → Review findings → Come back here) with deep links. Same pattern as the existing Ingest empty state (futureplans item U4).

**Effort:** one session. ~150 lines of route code, ~3 small templates, ~100 lines of tests against the existing `iter_eval_rows` fixtures.

**Net complexity:** +3 templates, +1 module touched (`web/app.py`). No new dependencies. The CLI command (`consistency-check eval`) stays and is unchanged — the web UI is an *alternative* surface, not a replacement, so the existing test_cli tests continue to pass.

---

### Block B — Targeted-eval: launch + result viewer

**One PR. Medium effort. Mostly new orchestration.**

**Adds, inside the Eval tab from Block A**, a second section: *"Targeted-eval (120 hand-crafted pairs)"*.

**The novice flow:**

1. Click **"Run targeted-eval"**.
2. A confirmation dialog appears: *"This sends 120 pairs to <judge_provider>:<judge_model>. Estimated cost: $1–$8 depending on model. Continue?"* — uses the existing `<dialog class="cc-dialog">` pattern. Mirrors futureplans U6.
3. Click Continue. Tab switches to a live-progress view (polled like the existing Stats tab live view).
4. When done, the per-bucket table renders inline. Prior runs are listed below in a small history so the analyst can see before-vs-after deltas.

**Routes** (new):

- `POST /eval/targeted/runs` — kicks off the harness in a `BackgroundTask`, returns a run id. Reuses the existing `BackgroundTasks` pattern from `POST /runs`.
- `GET /eval/targeted/runs/{run_id}` — live partial: progress counter + "X of 120 pairs scored."
- `GET /eval/targeted/runs/{run_id}/result` — final partial: per-bucket markdown table rendered as HTML.
- `GET /eval/targeted/runs` — history list (most recent first).

**Storage.** The harness already writes `metrics.json` + `summary.md` to disk. The web layer adopts the existing `audit/naming.py` convention: `data_dir/eval/targeted_run_<ts>_<run_id_short>.json`. New `iter_targeted_runs(data_dir)` helper in a small new module `consistency_checker/audit/targeted_runs.py` lists them.

**Background-task semantics.** The harness blocks for 3–10 minutes depending on model. The pattern is identical to `_run_check_in_background` in `web/app.py`: register a pending run, fire `BackgroundTasks.add_task`, update on completion. Failures go to a `cc__targeted_failed.html` partial with the error message (mirrors futureplans U2/U5).

**API key UX.** First-time launch checks for `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` in the process environment. If missing, render an inline panel: *"Add your Anthropic API key to enable this. We don't store it — paste here to set it for this server session."* Stores in process memory only, never on disk. Optional second checkbox: *"Read from ./.env on next server start instead"* (writes the line for them). The novice never types `export`.

**Model selector.** Dropdown (Opus / Sonnet / Haiku) with cost estimates next to each. Defaults to whatever `config.yml`'s `judge_model` is.

**New templates:**

- `cc__targeted_launcher.html` — the section embedded inside the Eval tab
- `cc__targeted_dialog.html` — confirmation dialog body
- `cc__targeted_live.html` — live progress partial (self-polling)
- `cc__targeted_final.html` — per-bucket result table
- `cc__targeted_failed.html` — failure partial
- `cc__targeted_history.html` — prior runs list

**Effort:** two sessions. Bulk of the work is the background-task plumbing + the API-key UX. Result-rendering reuses `format_markdown_summary` from PR #52's harness.

**Net complexity:** +6 templates, +1 small module (`audit/targeted_runs.py`), +1 module touched (`web/app.py`). No new dependencies. The CLI `python -m benchmarks.targeted_eval.harness` stays unchanged.

---

### Block C — Claude-Projects baseline: in-browser workflow

**Two PRs — Block C1 (ingestion + dedup + comparison) and Block C2 (in-browser labelling page).** Largest of the three; the only block that replaces an external multi-tool workflow.

The semantic-analyst's protocol today: web (claude.ai) → editor → terminal → spreadsheet → terminal. This block collapses everything past the first step into the existing web UI.

**Adds a third section to the Eval tab**: *"Claude-Projects baseline"*.

#### Block C1 — Ingest + dedup + comparison

**The novice flow:**

1. Click **"New baseline"**, supply corpus name.
2. Three file-drop boxes appear: *Run 1*, *Run 2*, *Run 3*. Drop the JSON output from each claude.ai chat into the matching box. Loader tolerates the existing pasted-JSON-array and `​```json` fence cases (PR #53 already handles).
3. Click **"Dedupe and label"**. Server-side dedup runs; analyst lands on the labelling page (Block C2).
4. After labelling, click **"Compute comparison"**. Renders the per-bucket table + cost-per-real-finding + agreement ratio inline, same shape as `format_comparison_markdown`.

**Storage.** Runs land in `data_dir/baseline/<corpus>/run_<N>.json`. Labels land in `data_dir/baseline/<corpus>/labels.csv` (round-trippable with the existing CSV writer/reader). Metrics in `data_dir/baseline/<corpus>/metrics.json` + `metrics.md`.

**Routes** (new):

- `POST /eval/baseline/uploads` — multi-file upload, validates each via `load_baseline_run`, writes to disk.
- `POST /eval/baseline/{corpus}/dedupe` — calls `write_findings_for_labelling`, redirects to the labelling page.
- `POST /eval/baseline/{corpus}/compare` — calls `compute_comparison` + `format_comparison_markdown`, renders inline.
- `GET /tabs/eval/baseline` — section list of in-flight + complete baselines.

**New templates:**

- `cc__baseline_section.html` — the section embedded inside the Eval tab
- `cc__baseline_uploader.html` — three-file drop UI
- `cc__baseline_comparison.html` — final result partial

**Effort:** one session. The pure logic (`compute_comparison`, etc.) is done.

#### Block C2 — In-browser labelling page

**Replaces the CSV-plus-spreadsheet step.** Brings labelling into a dedicated tab with one finding per screen and big buttons.

**The novice flow:**

1. Lands here after Block C1's dedupe step.
2. One finding visible at a time: `doc_a` excerpt on the left, `doc_b` excerpt on the right, the LLM rationale below.
3. Three large buttons: **Real** / **False positive** / **Dismiss / not sure** (uses the same C / F / D verdict pattern as the existing reviewer workflow). Keyboard shortcuts the same.
4. Skip / back navigation. Notes field for each.
5. Progress bar: "Labelled 12 of 47."
6. Done button (visible only once all rows are labelled or explicitly skipped) returns to the baseline section.

**Routes** (new):

- `GET /eval/baseline/{corpus}/label` — labelling page; renders the first un-labelled finding by default
- `GET /eval/baseline/{corpus}/label/{finding_id}` — single-finding view
- `POST /eval/baseline/{corpus}/label/{finding_id}` — set label, redirect to next un-labelled

**Storage.** Backs the existing `labels.csv` from C1; on each label-set the CSV is rewritten via `load_labels` → mutate → write. Same round-tripping shape.

**New templates:**

- `cc__baseline_label.html` — the per-finding labelling view
- `cc__baseline_progress.html` — progress-bar partial

**Effort:** two sessions. Labelling UI design is the bulk of the work; the data layer is already there.

**Net complexity (C1 + C2):** +5 templates, +1 module touched (`web/app.py`). No new dependencies. CLI subcommands (`dedupe`, `compare`) stay unchanged.

---

## Sequencing

Strict A → B → C order. Each block is one PR (C is two PRs).

```
PR #N+1  Block A  Eval tab + precision + calibration                        (1 session)
PR #N+2  Block B  Targeted-eval launcher + live + history + key UX          (2 sessions)
PR #N+3  Block C1 Baseline uploader + dedupe + comparison                   (1 session)
PR #N+4  Block C2 In-browser labelling page                                 (2 sessions)
```

Total: **~6 sessions of work** across **4 PRs**, all squash-merged onto `main` per the project git philosophy.

## Stop criteria per block

Each block has its own go/no-go check. The block ships only if its check passes; otherwise back out and revisit.

| Block | Stop criterion |
|---|---|
| A | Renders correct precision + calibration tables for a known fixture (the seeded reviewer-verdicts test from `tests/test_audit_eval.py`). Empty-state and `n < 30` banner display correctly. |
| B | Can launch a targeted-eval run, get live progress, see the final per-bucket table, see prior runs in history. API key flow works without exporting in shell. Failed runs render the failure partial. |
| C1 | Upload 3 sample run JSONs (test fixture), dedupe writes the labelling CSV, comparison renders. The fixture CSV produces the same metrics as the CLI `compare` command on the same inputs. |
| C2 | Label all findings via clicks, progress bar advances, CSV round-trips with `load_labels`. After-labelling comparison matches CLI behaviour. |

## What this plan does *not* cover

- **Phase 1 prompt + extractor repair** — that's downstream of running Phase 0 on real corpora. UI changes there happen only if the audit/findings schema changes.
- **Packaging / pipx / installer** — futureplans item #14, separate effort. This plan assumes the analyst already has a running `consistency-check serve` somewhere they can navigate to.
- **Persona-aware analysis** — futureplans item #19, separate effort, gated on ADR-0008.
- **Whole-corpus prompt-cached judge** — the Phase 2 pivot. Sequenced after the three Phase 0 signals have produced a corpus-grounded decision.

## Net-complexity audit

Per the project's `CLAUDE.md` "no net complexity" guardrail: every PR that adds a file must justify it.

| Files added | Justification |
|---|---|
| `cc_eval.html` + 4 partials (Block A) | New tab, no existing template to extend. Partials follow the established `cc__*.html` pattern. |
| `audit/targeted_runs.py` (Block B) | Targeted-eval runs are a *different shape* than `pipeline_runs` (no Stage A NLI, no `findings` table writes — runs entirely in `benchmarks/`). Stuffing them into `audit/logger.py` would conflate two concerns. |
| Per-block templates | Same `cc__*.html` partial-per-state pattern as Block G. |

No new Python dependencies anywhere in the four PRs. No new database migrations except optionally one: a `targeted_eval_runs` table to replace the disk-file approach in Block B if the analyst is expected to run dozens of times. Defer that decision until Block B has shipped and we see how many runs accumulate.

## Open questions for operator approval

1. **Block B model selector** — show all three (Opus / Sonnet / Haiku) or default-only with a config-edit hint? I'd recommend showing all three with cost estimates inline.
2. **Block C1 corpus_name** — derive from the first uploaded run's JSON `corpus_name` field, or ask the analyst at upload time? Asking is one more step but prevents the analyst from accidentally overwriting an existing baseline.
3. **API key persistence** — process-memory only by default (loses on server restart) vs offer to write to `.env` on first launch? Writing requires the analyst to trust the server with key persistence; process-memory is the safer default.

Once approved, work starts at Block A immediately.
