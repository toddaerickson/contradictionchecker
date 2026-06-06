# Definition-detector eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce two publishable accuracy numbers for the default definition-inconsistency detector — judge P/R/F1 on a real-document-mined gold pair set (Number A) and production precision on a live corpus (Number B) — and publish them with an honest methodology.

**Architecture:** Two code units (a candidate-mining script and a metrics extension to the existing definition-eval harness) plus three maintainer-gated data steps (curate labels, run, UI-review). The mining script and harness are pure-function-cored for hermetic testing; the real judge is only invoked in manual runs, never in CI.

**Tech Stack:** Python 3.11, `consistency_checker` (AssertionStore, DefinitionChecker, FixtureDefinitionJudge), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-06-02-definition-detector-eval-design.md`

---

## Execution note — human-in-the-loop

Tasks 1, 2, 6 are pure code (subagent-executable, TDD). **Tasks 3 and 5 are maintainer steps** (curating labels, reviewing findings in the UI) and **Task 4 depends on Task 3's output**. A subagent run should complete Tasks 1–2, then **STOP and hand back** with the Task 3 runbook for the maintainer; resume Tasks 4 and 6 after the maintainer produces `pairs.jsonl` (Task 3) and the production-precision number (Task 5).

## File structure

- Modify `benchmarks/definition_eval/harness.py` — add P/R/F1 + confusion + metrics.json (Task 1).
- Create `benchmarks/definition_eval/mine_pairs.py` — candidate-pair miner (Task 2).
- Create `tests/benchmarks/test_definition_eval.py` — hermetic tests for both (Tasks 1, 2).
- Create (by maintainer, Task 3) `benchmarks/definition_eval/pairs.jsonl` — curated gold set (replaces the 17-row stub).
- Create (Task 4 run) `benchmarks/definition_eval/metrics.json` + `benchmarks/definition_eval/runs/<date>.md`.
- Modify `docs/benchmarks.md` + `README.md` (Task 6).

---

## Task 1: Add P/R/F1 + confusion matrix to the harness

**Files:**
- Modify: `benchmarks/definition_eval/harness.py`
- Test: `tests/benchmarks/test_definition_eval.py`

The harness `run()` already returns a `predictions` list where each item has `label` (`"consistent"`/`"divergent"`) and `predicted` (same domain). `divergent` is the positive class. Compute metrics as a **pure function over that list** so it is testable without a judge.

- [ ] **Step 1: Write the failing test for `_metrics`**

Create `tests/benchmarks/test_definition_eval.py`:
```python
from benchmarks.definition_eval.harness import _metrics


def _pred(label, predicted):
    return {"label": label, "predicted": predicted}


def test_metrics_perfect():
    preds = [_pred("divergent", "divergent"), _pred("consistent", "consistent")]
    m = _metrics(preds)
    assert m["confusion"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 1}
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_metrics_mixed():
    # 2 true divergent: 1 caught (tp), 1 missed (fn). 1 consistent flagged divergent (fp).
    preds = [
        _pred("divergent", "divergent"),
        _pred("divergent", "consistent"),
        _pred("consistent", "divergent"),
        _pred("consistent", "consistent"),
    ]
    m = _metrics(preds)
    assert m["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5


def test_metrics_no_positive_predictions_is_none():
    preds = [_pred("consistent", "consistent")]
    m = _metrics(preds)
    assert m["precision"] is None  # no tp+fp
    assert m["recall"] is None     # no tp+fn
    assert m["f1"] is None
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/benchmarks/test_definition_eval.py -v`
Expected: FAIL — `ImportError: cannot import name '_metrics'`.

- [ ] **Step 3: Implement `_metrics` in `harness.py`**

Add after `_predicted_label` (around line 43):
```python
def _metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Precision/recall/F1 with ``divergent`` as the positive class.

    Pure over the predictions list so it is unit-testable without a judge.
    Returns ``None`` for a metric whose denominator is zero (distinguishes
    "undefined" from "scored 0").
    """
    tp = fp = fn = tn = 0
    for p in predictions:
        gold_pos = p["label"] == "divergent"
        pred_pos = p["predicted"] == "divergent"
        if gold_pos and pred_pos:
            tp += 1
        elif not gold_pos and pred_pos:
            fp += 1
        elif gold_pos and not pred_pos:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall) > 0
        else None
    )
    return {
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n": len(predictions),
    }
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `uv run pytest tests/benchmarks/test_definition_eval.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Wire metrics into `run()` and `main()`**

In `run()`, change the return to include metrics (keep `summary` + `predictions`):
```python
    return {"metrics": _metrics(predictions), "summary": summary, "predictions": predictions}
```
In `main()`, replace the `--baseline` block tail with a metrics print + `--metrics` output. After the per-category print loop, add:
```python
    m = result["metrics"]
    cm = m["confusion"]

    def _pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x * 100:.1f}%"

    print(
        f"\nOverall (divergent = positive): n={m['n']}  "
        f"P={_pct(m['precision'])}  R={_pct(m['recall'])}  F1={_pct(m['f1'])}"
    )
    print(f"  confusion: tp={cm['tp']} fp={cm['fp']} fn={cm['fn']} tn={cm['tn']}")
```
Rename the `--baseline` arg to `--metrics` (help: "Write metrics + per-category summary JSON here.") and write the full `result` (not just summary):
```python
    if args.metrics:
        args.metrics.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote metrics to {args.metrics}")
```

- [ ] **Step 6: Run the gate + commit**

```bash
uv run pytest -m "not slow and not live" -q
uv run ruff check . && uv run ruff format --check . && uv run mypy consistency_checker
git add benchmarks/definition_eval/harness.py tests/benchmarks/test_definition_eval.py
git commit -m "feat(eval): definition harness reports P/R/F1 + confusion matrix"
```

---

## Task 2: Candidate-pair mining script

**Files:**
- Create: `benchmarks/definition_eval/mine_pairs.py`
- Test: `tests/benchmarks/test_definition_eval.py` (append)

Mine same-term definition pairs from a corpus into a review file the maintainer labels. Core pairing logic is a pure function over a list of `Assertion`s (hermetic test); a thin wrapper reads the store.

- [ ] **Step 1: Write the failing test**

Append to `tests/benchmarks/test_definition_eval.py`:
```python
from consistency_checker.extract.schema import Assertion
from benchmarks.definition_eval.mine_pairs import build_candidates


def _defn(doc, term, text):
    return Assertion.build(doc, f'"{term}" means {text}.', kind="definition", term=term, definition_text=text)


def test_build_candidates_pairs_same_canonical_term():
    defs = [
        _defn("d1", "Board", "the board of directors"),
        _defn("d2", "board", "the board of directors"),       # identical text, canonical-equal term
        _defn("d3", "Board", "the supervisory board only"),   # divergent candidate
    ]
    cands = build_candidates(defs, max_pairs=100)
    # 3 unordered pairs over 3 same-term defs
    assert len(cands) == 3
    cats = {c["category"] for c in cands}
    assert "identical" in cats and "review" in cats
    # every candidate carries the labeling contract
    for c in cands:
        assert c["label"] == ""
        assert c["term"] and c["def_a"] and c["def_b"]
        assert c["doc_a"] and c["doc_b"]


def test_build_candidates_skips_singletons_and_caps():
    defs = [_defn("d1", "Quorum", "a majority"), _defn("d2", "Notice", "written notice")]
    assert build_candidates(defs, max_pairs=100) == []  # no term has >= 2 defs
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/benchmarks/test_definition_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: benchmarks.definition_eval.mine_pairs`.

- [ ] **Step 3: Implement `mine_pairs.py`**

Create `benchmarks/definition_eval/mine_pairs.py`:
```python
"""Mine candidate same-term definition pairs from an ingested corpus into a
review file for human labeling. The output feeds benchmarks/definition_eval/
after the maintainer fills the empty ``label`` field (consistent | divergent)
and deletes junk rows.

Run:
    uv run python -m benchmarks.definition_eval.mine_pairs \
        --db data/store/assertions.db --corpus <name-or-id> \
        --out benchmarks/definition_eval/candidates.jsonl --max 100
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from consistency_checker.check.definition_terms import canonicalize_term
from consistency_checker.extract.schema import Assertion
from consistency_checker.index.assertion_store import AssertionStore


def build_candidates(definitions: list[Assertion], max_pairs: int) -> list[dict[str, Any]]:
    """Group definitions by canonical term and enumerate unordered candidate pairs.

    ``label`` is left empty for the maintainer to fill. ``category`` is a heuristic
    seed (``identical`` when the two definition texts match exactly, else ``review``).
    Non-identical pairs are surfaced first (more label-informative) before capping.
    """
    groups: dict[str, list[Assertion]] = {}
    for a in definitions:
        if a.kind != "definition" or a.term is None or a.definition_text is None:
            continue
        canon = canonicalize_term(a.term)
        if not canon:
            continue
        groups.setdefault(canon, []).append(a)

    candidates: list[dict[str, Any]] = []
    seq = 0
    for canon, defs in sorted(groups.items()):
        if len(defs) < 2:
            continue
        for a, b in itertools.combinations(defs, 2):
            ta = (a.definition_text or "").strip()
            tb = (b.definition_text or "").strip()
            if not ta or not tb:
                continue
            seq += 1
            candidates.append(
                {
                    "pair_id": f"{canon[:24]}_{seq:04d}",
                    "category": "identical" if ta == tb else "review",
                    "term": a.term,
                    "def_a": ta,
                    "def_b": tb,
                    "doc_a": a.doc_id,
                    "doc_b": b.doc_id,
                    "label": "",  # maintainer: "consistent" | "divergent" (or delete the row)
                }
            )
    candidates.sort(key=lambda c: 0 if c["category"] == "review" else 1)
    return candidates[:max_pairs]


def mine(db_path: Path, corpus_id: str | None, max_pairs: int) -> list[dict[str, Any]]:
    store = AssertionStore(db_path)
    store.migrate()
    try:
        definitions = [a for a, _org in store.iter_definitions(corpus_id=corpus_id)]
    finally:
        store.close()
    return build_candidates(definitions, max_pairs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--corpus", type=str, default=None, help="corpus_id (or omit for all).")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/definition_eval/candidates.jsonl"))
    ap.add_argument("--max", type=int, default=100)
    args = ap.parse_args()
    cands = mine(args.db, args.corpus, args.max)
    with args.out.open("w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(c) + "\n")
    n_review = sum(1 for c in cands if c["category"] == "review")
    print(f"Wrote {len(cands)} candidate pairs ({n_review} 'review', rest 'identical') to {args.out}")
    print("Next: open the file, set each row's \"label\" to consistent|divergent, delete junk rows,")
    print("then save the curated subset as benchmarks/definition_eval/pairs.jsonl")


if __name__ == "__main__":
    main()
```

Note: `--corpus` takes the **corpus_id** (run `consistency-check corpus list --db <path>` to map name → id). If the store needs the corpus name, pass the id shown there.

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run pytest tests/benchmarks/test_definition_eval.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Gate + commit**

```bash
uv run pytest -m "not slow and not live" -q
uv run ruff check . && uv run ruff format --check . && uv run mypy consistency_checker
git add benchmarks/definition_eval/mine_pairs.py tests/benchmarks/test_definition_eval.py
git commit -m "feat(eval): mine candidate same-term definition pairs for labeling"
```

**After Task 2: subagent STOPS and hands the Task 3 runbook to the maintainer.**

---

## Task 3 (MAINTAINER): Build the curated gold set

Not subagent-automatable — requires the real ingested DB and human labels.

- [ ] **Step 1: List corpora to get ids**

```bash
uv run consistency-check corpus list --db data/store/assertions.db
```

- [ ] **Step 2: Mine candidates from the ingested corpora**

For each corpus with definitions (Atkins bylaws+amendments, FCS call report, nonprofit bylaws), run:
```bash
uv run python -m benchmarks.definition_eval.mine_pairs \
    --db data/store/assertions.db --corpus <corpus_id> \
    --out benchmarks/definition_eval/candidates_<name>.jsonl --max 120
```

- [ ] **Step 2a: Watch the divergent balance.** Open the `review`-category rows. If there are very few genuinely-divergent pairs, prioritize **base-vs-amendment** corpora (a definition the amendment changed is a real divergence) and widen `--max` or add another corpus, so the gold set has enough positives for a non-noisy recall denominator (aim ≥ ~25 divergent).

- [ ] **Step 3: Label.** In each file, set every row's `"label"` to `"consistent"` or `"divergent"` (delete extraction-junk rows). Concatenate the curated rows into `benchmarks/definition_eval/pairs.jsonl` (replacing the 17-row stub). Refine `category` where useful (`identical`, `paraphrase`, `scope`, `numeric`, `temporal`, `cross-org`).

- [ ] **Step 4: Commit the gold set**

```bash
git add benchmarks/definition_eval/pairs.jsonl
git commit -m "data(eval): curated definition gold set mined from ingested corpora"
```

---

## Task 4: Run the harness on the gold set; record Number A

**Files:** Create `benchmarks/definition_eval/metrics.json`, `benchmarks/definition_eval/runs/<YYYY-MM-DD>.md`. Depends on Task 3.

- [ ] **Step 1: Run (real judge — needs `MOONSHOT_API_KEY` in `.env`, from repo root)**

```bash
uv run python -m benchmarks.definition_eval.harness \
    --pairs benchmarks/definition_eval/pairs.jsonl \
    --config config.yml \
    --metrics benchmarks/definition_eval/metrics.json
```
Capture the printed `Overall (divergent = positive): n=… P=… R=… F1=…` line.

- [ ] **Step 2: Write the run note**

Create `benchmarks/definition_eval/runs/<YYYY-MM-DD>.md` with: the command, the judge provider/model from `config.yml`, the overall P/R/F1 + confusion, the per-category table, and a one-line note that this is judge-level discrimination on surfaced pairs (not end-to-end recall).

- [ ] **Step 3: Commit**

```bash
git add benchmarks/definition_eval/metrics.json benchmarks/definition_eval/runs/
git commit -m "data(eval): definition gold-set run — Number A (judge P/R/F1)"
```

---

## Task 5 (MAINTAINER): Number B — production precision

Not subagent-automatable — requires a live check run + human verdicts.

- [ ] **Step 1: Run the default check on one corpus** (pick the one Task 3 showed has the richest definition findings):
```bash
uv run consistency-check check --corpus <corpus_id>
```
- [ ] **Step 2: Review in the UI.** `uv run consistency-check serve --open` → open the corpus → mark each definition finding `confirmed` / `false_positive` (`dismissed` = not eval-relevant).
- [ ] **Step 3: Read the precision:**
```bash
uv run consistency-check eval --db data/store/assertions.db
```
Record the `definition_inconsistency` precision row (and `n_reviewed`).

---

## Task 6: Publish both numbers

**Files:** Modify `docs/benchmarks.md`, `README.md`. Depends on Tasks 4 + 5 (real numbers).

- [ ] **Step 1: Add a "Definition detector" section to `docs/benchmarks.md`** with: how the gold set was built (mined from ingested corpora + maintainer-curated), the run command, the overall + per-category table from `metrics.json`, the Number B production-precision figure + corpus, and the explicit caveat: *Number A is judge-level recall on surfaced pairs, not end-to-end corpus recall; end-to-end recall and a public-filings gold set are tracked follow-ups.*

- [ ] **Step 2: Add a headline line to the README "What runs by default" section**, e.g.: "On a <N>-pair gold set mined from real bylaws/filings, the definition judge scores P=<x>% / R=<y>% / F1=<z>% (judge-level, divergent = positive); on a live <corpus> run its production precision was <p>% over <n> reviewed findings. See `docs/benchmarks.md`." Use the real numbers from `metrics.json` + Task 5.

- [ ] **Step 3: Gate + commit**

```bash
uv run pytest -m "not slow and not live" -q && uv run ruff check .
git add docs/benchmarks.md README.md
git commit -m "docs: publish definition-detector precision/recall (gold set + production precision)"
```

---

## Verification (end)

```bash
uv run pytest -m "not slow and not live" -q
uv run ruff check . && uv run ruff format --check . && uv run mypy consistency_checker
uv build
```
All green. `metrics.json` exists with non-null precision/recall; `docs/benchmarks.md` + README cite both numbers with the recall caveat.

## Out of scope (follow-ups)

- End-to-end corpus recall (exhaustive per-document ground truth).
- Public-filings / EDGAR gold-set expansion.
- Any detector tuning (measurement only).
