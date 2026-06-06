"""Definition-eval harness: scores the definition checker (short-circuit + judge)
against a labeled pair set, by category.

Regression guard, not the primary precision gate. Run manually with a provider
key configured (reads config.yml like the CLI does):

    uv run python -m benchmarks.definition_eval.harness --metrics benchmarks/definition_eval/baseline.json

Dataset format (JSONL, one object per line):
    {"pair_id": str, "category": str, "term": str,
     "def_a": str, "def_b": str, "label": "consistent" | "divergent"}
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from consistency_checker.config import Config
from consistency_checker.extract.schema import Assertion
from consistency_checker.pipeline import make_definition_checker

HERE = Path(__file__).resolve().parent
DEFAULT_PAIRS = HERE / "pairs.jsonl"


def _assertion(doc: str, term: str, definition_text: str) -> Assertion:
    return Assertion.build(
        doc,
        f'"{term}" means {definition_text}.',
        kind="definition",
        term=term,
        definition_text=definition_text,
    )


def _predicted_label(verdict: str) -> str:
    # divergent is the positive class; everything else counts as "consistent"
    return "divergent" if verdict == "definition_divergent" else "consistent"


def _metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Precision/recall/F1 with ``divergent`` as the positive class.

    Pure over the predictions list so it is unit-testable without a judge.
    Returns ``None`` for a metric whose denominator is zero.
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


def run(pairs_path: Path, config_path: Path | None) -> dict[str, Any]:
    config = Config.from_yaml(config_path) if config_path and config_path.exists() else Config()
    checker = make_definition_checker(config)
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    predictions = []
    for line in pairs_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        a = _assertion("doc_a", row["term"], row["def_a"])
        b = _assertion("doc_b", row["term"], row["def_b"])
        findings = list(checker.find_inconsistencies([a, b]))
        verdict = findings[0].verdict.verdict if findings else "uncertain"
        predicted = _predicted_label(verdict)
        correct = predicted == row["label"]
        by_cat[row["category"]]["total"] += 1
        by_cat[row["category"]]["correct"] += int(correct)
        predictions.append(
            {
                "pair_id": row["pair_id"],
                "category": row["category"],
                "label": row["label"],
                "verdict": verdict,
                "predicted": predicted,
                "correct": correct,
            }
        )
    summary = {
        cat: {"accuracy": c["correct"] / c["total"], **c} for cat, c in sorted(by_cat.items())
    }
    return {"metrics": _metrics(predictions), "summary": summary, "predictions": predictions}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--config", type=Path, default=Path("config.yml"))
    ap.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Write metrics + per-category summary JSON here.",
    )
    args = ap.parse_args()
    result = run(args.pairs, args.config)
    for cat, s in result["summary"].items():
        print(f"{cat:<32} {s['correct']:>3}/{s['total']:<3}  acc={s['accuracy']:.2f}")

    m = result["metrics"]
    cm = m["confusion"]

    def _pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x * 100:.1f}%"

    print(
        f"\nOverall (divergent = positive): n={m['n']}  "
        f"P={_pct(m['precision'])}  R={_pct(m['recall'])}  F1={_pct(m['f1'])}"
    )
    print(f"  confusion: tp={cm['tp']} fp={cm['fp']} fn={cm['fn']} tn={cm['tn']}")
    misses = [p for p in result["predictions"] if not p["correct"]]
    if misses:
        print("\nMisses:")
        for m in misses:
            print(f"  [{m['category']}] {m['pair_id']}: label={m['label']} verdict={m['verdict']}")
    if args.metrics:
        args.metrics.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote metrics to {args.metrics}")


if __name__ == "__main__":
    main()
