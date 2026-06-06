# Definition-detector evaluation — design

**Status:** approved (brainstorm 2026-06-02), pending implementation plan.
**Goal:** Produce a credible, publishable accuracy number for the **default** detector (definition-inconsistency), to replace the current state where the only published precision/recall figures are for the *deprioritized* pairwise/CONTRADOC path. This is the standing commercial-credibility blocker.

## Problem

The definition-inconsistency detector is the product default (ADR-0015 made pairwise opt-in). It groups assertions by canonical term and asks the LLM judge whether two same-term definitions diverge. Today there is **no credible accuracy number** for it:

- `benchmarks/definition_eval/` exists but its `pairs.jsonl` is **17 hand-crafted pairs** (10 consistent / 7 divergent) and its own docstring calls it "a regression guard, not the primary precision gate." `harness.py` reports only accuracy, not P/R/F1.
- `audit/eval.py` + the `eval` CLI mine `reviewer_verdicts` into per-detector **precision** (`confirmed / (confirmed + false_positive)`) on real corpora — but give **no recall** (they only label what the detector surfaced).
- The `CONTRADOC` harness (`benchmarks/contradoc_harness.py`) reports P/R/F1, but for the **pairwise** detector only.

## Decision (hybrid)

Report **two complementary numbers**, each honest about what it measures:

### Number A — Judge discrimination P / R / F1 on a real-document gold pair set
Extend the existing `benchmarks/definition_eval/harness.py` (it already feeds `(term, def_a, def_b, label)` straight to the definition judge, bypassing grouping). Grow `pairs.jsonl` from 17 to a real, curated set mined from documents.

- **Measures:** given a same-term definition pair, how accurately the judge calls `divergent` vs `consistent`.
- **Scope caveat (documented, not overclaimed):** this is **judge-level** recall on *surfaced* pairs — it does **NOT** measure end-to-end corpus recall (divergences the canonical-term grouping never surfaces). True end-to-end recall requires exhaustive per-document ground truth (every true divergence found by human reading) and is an explicit follow-up, out of scope here.

### Number B — Production precision on a live corpus
Run the full detector on one already-ingested corpus; the maintainer marks each surfaced finding `confirmed` / `false_positive` in the web UI; `consistency-check eval` reports precision.

- **Measures:** of the divergences the detector actually surfaces in production conditions, how many are real.
- Reuses existing infra (`eval` subcommand + `audit/eval.py`) — no new code, just the run + the review.

## Gold set construction (Number A)

- **Source for v1:** the already-ingested corpora — Atkins bylaws + amendments, the FCS UCR call report, the nonprofit-bylaws set. (Public-filings / EDGAR expansion is a noted follow-up, not v1.)
- **Mining:** a script enumerates candidate `(term, def_a, def_b)` pairs from real assertions via the detector's canonical-term grouping, and deliberately includes base-vs-amendment pairs (likely real divergences) and same-section repeats (likely real consistents).
- **Curation (the maintainer's task):** the script emits a ~**100-candidate** review file (JSONL/table); the maintainer labels each `consistent` / `divergent` and discards junk. Estimated ~1 hour. The curated file becomes the new `pairs.jsonl`.
- **Category tags:** each pair tagged (`identical`, `paraphrase`, `scope`, `numeric/threshold`, `temporal`, `cross-org-noise`) so the harness can report per-category accuracy and expose *where* the judge fails.
- **Balance:** skew candidate selection to ensure enough divergent cases for a meaningful recall denominator (a corpus naturally yields mostly consistents).

## Harness changes (Number A)

- Extend `benchmarks/definition_eval/harness.py`: add **precision / recall / F1**, a **confusion matrix**, and a **per-category breakdown** (today it prints only accuracy + per-pair lines). Treat `divergent` as the positive class.
- Output a committed `metrics.json` plus a human-readable run note under `benchmarks/definition_eval/runs/` (mirroring the existing `benchmarks/targeted_eval/runs/` baseline convention).
- Keep it a `live`-style manual run (it calls the real judge) — not part of the hermetic CI gate.

## Publication

- New **Definition detector** subsection in `docs/benchmarks.md`: methodology, the gold-set provenance, the run command, and the explicit judge-level-vs-end-to-end-recall caveat.
- A headline line in the README "What runs by default" section reporting Number A (P/R/F1) and Number B (production precision on corpus X), replacing the current absence of any definition-detector figure.

## Components / data flow

```
ingested corpora (SQLite assertions)
   │  mine_pairs.py (new; uses canonical-term grouping)
   ▼
candidate review file (~100 pairs, JSONL)
   │  maintainer curates labels (consistent | divergent | discard)
   ▼
benchmarks/definition_eval/pairs.jsonl  (curated gold set)
   │  harness.py run (real judge)  ──►  metrics.json + runs/<date>.md   [Number A: P/R/F1]
   
one ingested corpus
   │  consistency-check check  →  findings
   │  maintainer marks verdicts in web UI
   ▼
consistency-check eval  →  per-detector precision               [Number B: production precision]
   
docs/benchmarks.md + README  ◄── both numbers + methodology + caveats
```

## Success criteria

- A curated `pairs.jsonl` of ~100 real, labeled definition pairs (replacing the 17-pair regression stub), with category tags.
- `harness.py` emits P/R/F1 + confusion matrix + per-category accuracy and writes a committed `metrics.json` + run note.
- A production-precision number from `eval` on one real corpus.
- `docs/benchmarks.md` + README updated with both numbers and an honest methodology/caveat section.
- Gate stays green (the new harness is a manual/live run, not added to hermetic CI).

## Out of scope (explicit follow-ups)

- **End-to-end corpus recall** (exhaustive per-document ground truth) — the expensive labeling effort; deferred.
- **Public-filings / EDGAR gold-set expansion** — improves external validity and redistributability; deferred.
- Any change to the detector itself — this is measurement only, not tuning.

## Risks

- **Author-equals-judge circularity** — mitigated by mining definitions from *real documents* (not LLM-invented) and maintainer-curated labels.
- **Cross-org noise** — comparing different organizations' bylaws makes every shared term "diverge" by construction (per the 2026-05-21 eval); category tags isolate this so it does not distort the headline number.
- **Small divergent denominator** — if mining yields too few divergent pairs, recall is noisy; mitigated by deliberately sourcing base-vs-amendment pairs and, if needed, widening the source corpora.
