"""CONTRADOC benchmark harness.

Drives Stage A (NLI) + Stage B (LLM judge) against a normalised CONTRADOC
dataset and reports precision / recall / F1 against gold labels. The pipeline's
atomic-fact extraction stage is bypassed here because CONTRADOC's labels are
per-document-pair, not per-claim — the harness treats each document text as a
single assertion-equivalent input.

Dataset format (JSONL, one object per line):

    {"pair_id": "...", "document_a": "...", "document_b": "...",
     "label": "contradiction" | "not_contradiction"}

The CONTRADOC dataset itself is not redistributed; convert the original source
into this normalised form and pass the path via ``--input``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from consistency_checker.check.llm_judge import Judge, LLMJudge
from consistency_checker.check.nli_checker import (
    NliChecker,
    TransformerNliChecker,
    passes_threshold,
    score_bidirectional,
)
from consistency_checker.check.providers.anthropic import AnthropicProvider
from consistency_checker.check.providers.openai import OpenAIProvider
from consistency_checker.extract.schema import Assertion
from consistency_checker.logging_setup import get_logger

_log = get_logger(__name__)

GoldLabel = Literal["contradiction", "not_contradiction"]


@dataclass(frozen=True, slots=True)
class ContradocEntry:
    pair_id: str
    document_a: str
    document_b: str
    label: GoldLabel


@dataclass(frozen=True, slots=True)
class BenchmarkPrediction:
    pair_id: str
    gold: GoldLabel
    predicted: GoldLabel
    judge_verdict: str
    judge_confidence: float
    nli_p_contradiction: float


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    n_pairs: int
    n_true_positive: int
    n_false_positive: int
    n_true_negative: int
    n_false_negative: int
    precision: float
    recall: float
    f1: float
    nli_threshold: float
    predictions: list[BenchmarkPrediction]


def load_contradoc(path: Path | str, *, sample: int | None = None) -> list[ContradocEntry]:
    """Load CONTRADOC entries from a JSONL file. ``sample`` limits the row count."""
    rows: list[ContradocEntry] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row["label"]
            if label not in {"contradiction", "not_contradiction"}:
                raise ValueError(f"Unexpected label {label!r} in {path}")
            rows.append(
                ContradocEntry(
                    pair_id=str(row["pair_id"]),
                    document_a=str(row["document_a"]),
                    document_b=str(row["document_b"]),
                    label=label,
                )
            )
            if sample is not None and len(rows) >= sample:
                break
    return rows


def _document_assertion(pair_id: str, side: str, text: str) -> Assertion:
    """Adapt a CONTRADOC document text into an Assertion the judge can consume."""
    return Assertion.build(doc_id=f"{pair_id}_{side}", assertion_text=text)


def _confusion(predictions: list[BenchmarkPrediction]) -> tuple[int, int, int, int]:
    tp = sum(1 for p in predictions if p.predicted == "contradiction" and p.gold == "contradiction")
    fp = sum(
        1 for p in predictions if p.predicted == "contradiction" and p.gold == "not_contradiction"
    )
    tn = sum(
        1
        for p in predictions
        if p.predicted == "not_contradiction" and p.gold == "not_contradiction"
    )
    fn = sum(
        1 for p in predictions if p.predicted == "not_contradiction" and p.gold == "contradiction"
    )
    return tp, fp, tn, fn


def _f1_components(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def run_benchmark(
    entries: list[ContradocEntry],
    *,
    nli_checker: NliChecker,
    judge: Judge,
    nli_threshold: float = 0.5,
) -> BenchmarkResult:
    """Score every CONTRADOC pair through Stage A + Stage B."""
    predictions: list[BenchmarkPrediction] = []
    for entry in entries:
        nli = score_bidirectional(nli_checker, entry.document_a, entry.document_b)
        if not passes_threshold(nli, nli_threshold):
            # Stage A didn't flag this pair — predict not_contradiction.
            predictions.append(
                BenchmarkPrediction(
                    pair_id=entry.pair_id,
                    gold=entry.label,
                    predicted="not_contradiction",
                    judge_verdict="not_invoked",
                    judge_confidence=0.0,
                    nli_p_contradiction=nli.p_contradiction,
                )
            )
            continue

        a = _document_assertion(entry.pair_id, "a", entry.document_a)
        b = _document_assertion(entry.pair_id, "b", entry.document_b)
        verdict = judge.judge(a, b)
        # Stage B verdicts other than "contradiction" (i.e. uncertain or
        # not_contradiction) all collapse to a negative prediction.
        predicted: GoldLabel = (
            "contradiction" if verdict.verdict == "contradiction" else "not_contradiction"
        )
        predictions.append(
            BenchmarkPrediction(
                pair_id=entry.pair_id,
                gold=entry.label,
                predicted=predicted,
                judge_verdict=verdict.verdict,
                judge_confidence=verdict.confidence,
                nli_p_contradiction=nli.p_contradiction,
            )
        )

    tp, fp, tn, fn = _confusion(predictions)
    precision, recall, f1 = _f1_components(tp, fp, fn)
    return BenchmarkResult(
        n_pairs=len(predictions),
        n_true_positive=tp,
        n_false_positive=fp,
        n_true_negative=tn,
        n_false_negative=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        nli_threshold=nli_threshold,
        predictions=predictions,
    )


def write_metrics_json(result: BenchmarkResult, path: Path | str) -> None:
    payload: dict[str, object] = asdict(result)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _make_nli(model_name: str | None) -> NliChecker:
    if not model_name:
        return TransformerNliChecker()
    return TransformerNliChecker(model_name=model_name)


def _make_judge(provider: str, model: str | None) -> Judge:
    if provider == "anthropic":
        kwargs: dict[str, str] = {}
        if model:
            kwargs["model"] = model
        return LLMJudge(AnthropicProvider(**kwargs))
    if provider == "openai":
        kwargs = {}
        if model:
            kwargs["model"] = model
        return LLMJudge(OpenAIProvider(**kwargs))
    raise ValueError(f"Unsupported judge provider: {provider!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="contradoc_harness", description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to CONTRADOC JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write metrics JSON.")
    parser.add_argument("--sample", type=int, default=None, help="Limit number of pairs scored.")
    parser.add_argument("--nli-threshold", type=float, default=0.5)
    parser.add_argument(
        "--judge-provider",
        choices=["anthropic", "openai"],
        default="anthropic",
    )
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--nli-model", default=None)
    args = parser.parse_args(argv)

    entries = load_contradoc(args.input, sample=args.sample)
    _log.info("Loaded %d CONTRADOC pairs from %s", len(entries), args.input)

    nli = _make_nli(args.nli_model)
    judge = _make_judge(args.judge_provider, args.judge_model)

    result = run_benchmark(entries, nli_checker=nli, judge=judge, nli_threshold=args.nli_threshold)
    write_metrics_json(result, args.output)
    print(
        f"pairs={result.n_pairs} "
        f"TP={result.n_true_positive} FP={result.n_false_positive} "
        f"TN={result.n_true_negative} FN={result.n_false_negative} "
        f"precision={result.precision:.3f} recall={result.recall:.3f} f1={result.f1:.3f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
