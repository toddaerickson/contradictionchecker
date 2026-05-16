"""Targeted-eval harness: scores the judge directly against a labeled pair set.

Unlike :mod:`benchmarks.contradoc_harness`, this harness **skips the NLI gate**.
The pairs in ``pairs.jsonl`` are designed to isolate one linguistic phenomenon
at a time (defeasible override, modal divergence, temporal-scope mismatch),
and most of them have no surface-level contradiction that an NLI checker
would flag — gating would silently drop them from the test, hiding the very
failure mode we're trying to measure.

Dataset format (JSONL, one object per line):

    {"pair_id": "rule_exc_001",
     "bucket": "rule_exception" | "modal_divergence" | "temporal_scope",
     "assertion_a": "...",
     "assertion_b": "...",
     "ground_truth": "contradiction" | "not_contradiction" | "uncertain",
     "relation": "contradiction" | "contrariety" | "override" |
                 "entailment" | "scope_mismatch" | "independent",
     "notes": "..."}

``ground_truth`` is what the judge *should* return given the current
three-value enum. ``relation`` is the finer-grained logical relation (per
the logician's seven-relation taxonomy) for downstream slicing.

Outputs:
- Per-bucket accuracy, FP rate, FN rate, abstain rate.
- A flat ``predictions`` list so a diff between two runs is straightforward.
- Optional markdown summary for paste-into-PR before/after deltas.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from consistency_checker.check.llm_judge import Judge, LLMJudge
from consistency_checker.check.providers.anthropic import AnthropicProvider
from consistency_checker.check.providers.openai import OpenAIProvider
from consistency_checker.extract.schema import Assertion
from consistency_checker.logging_setup import get_logger

_log = get_logger(__name__)

GroundTruth = Literal["contradiction", "not_contradiction", "uncertain"]
JudgePredicted = Literal["contradiction", "not_contradiction", "uncertain"]
Bucket = Literal["rule_exception", "modal_divergence", "temporal_scope"]
Relation = Literal[
    "contradiction",
    "contrariety",
    "override",
    "entailment",
    "scope_mismatch",
    "independent",
]

_VALID_GROUND_TRUTH: frozenset[str] = frozenset({"contradiction", "not_contradiction", "uncertain"})
_VALID_BUCKETS: frozenset[str] = frozenset({"rule_exception", "modal_divergence", "temporal_scope"})
_VALID_RELATIONS: frozenset[str] = frozenset(
    {
        "contradiction",
        "contrariety",
        "override",
        "entailment",
        "scope_mismatch",
        "independent",
    }
)


@dataclass(frozen=True, slots=True)
class TargetedPair:
    """One labeled pair in the targeted eval set."""

    pair_id: str
    bucket: Bucket
    assertion_a: str
    assertion_b: str
    ground_truth: GroundTruth
    relation: Relation
    notes: str = ""


@dataclass(frozen=True, slots=True)
class TargetedPrediction:
    """One judge prediction for a labeled pair."""

    pair_id: str
    bucket: Bucket
    ground_truth: GroundTruth
    relation: Relation
    predicted: JudgePredicted
    judge_confidence: float
    judge_rationale: str


@dataclass(frozen=True, slots=True)
class BucketMetrics:
    """Per-bucket metrics for one slice of the eval set."""

    bucket: str
    n_pairs: int
    n_correct: int
    n_false_positive: int
    n_false_negative: int
    n_abstain: int
    accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    abstain_rate: float


@dataclass(frozen=True, slots=True)
class TargetedResult:
    """Aggregate result of one targeted-eval run."""

    n_pairs: int
    overall_accuracy: float
    overall_false_positive_rate: float
    overall_false_negative_rate: float
    overall_abstain_rate: float
    per_bucket: list[BucketMetrics] = field(default_factory=list)
    predictions: list[TargetedPrediction] = field(default_factory=list)


_REQUIRED_FIELDS: tuple[str, ...] = (
    "pair_id",
    "bucket",
    "assertion_a",
    "assertion_b",
    "ground_truth",
    "relation",
)


def load_pairs(path: Path | str) -> list[TargetedPair]:
    """Load and validate the targeted-eval JSONL.

    Raises ``ValueError`` with file:line context on malformed JSON, missing
    required fields, invalid enum values, or duplicate ``pair_id`` across rows.
    """
    rows: list[TargetedPair] = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("//"):
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            for field_name in _REQUIRED_FIELDS:
                if field_name not in row:
                    raise ValueError(f"{path}:{lineno}: missing required field {field_name!r}")
            bucket = row["bucket"]
            ground_truth = row["ground_truth"]
            relation = row["relation"]
            if bucket not in _VALID_BUCKETS:
                raise ValueError(f"{path}:{lineno}: invalid bucket {bucket!r}")
            if ground_truth not in _VALID_GROUND_TRUTH:
                raise ValueError(f"{path}:{lineno}: invalid ground_truth {ground_truth!r}")
            if relation not in _VALID_RELATIONS:
                raise ValueError(f"{path}:{lineno}: invalid relation {relation!r}")
            pair_id = str(row["pair_id"])
            if pair_id in seen_ids:
                raise ValueError(f"{path}:{lineno}: duplicate pair_id {pair_id!r}")
            seen_ids.add(pair_id)
            rows.append(
                TargetedPair(
                    pair_id=pair_id,
                    bucket=bucket,
                    assertion_a=str(row["assertion_a"]),
                    assertion_b=str(row["assertion_b"]),
                    ground_truth=ground_truth,
                    relation=relation,
                    notes=str(row.get("notes", "")),
                )
            )
    return rows


def _to_assertions(pair: TargetedPair) -> tuple[Assertion, Assertion]:
    """Adapt the eval pair into two ``Assertion`` objects the judge can consume.

    Uses the pair_id as a doc-id prefix so the resulting assertion ids are
    unique across pairs and the judge can't accidentally reuse cached state.
    """
    return (
        Assertion.build(doc_id=f"{pair.pair_id}_a", assertion_text=pair.assertion_a),
        Assertion.build(doc_id=f"{pair.pair_id}_b", assertion_text=pair.assertion_b),
    )


def _bucket_metrics(bucket: str, predictions: Sequence[TargetedPrediction]) -> BucketMetrics:
    n = len(predictions)
    correct = sum(1 for p in predictions if p.predicted == p.ground_truth)
    fp = sum(
        1
        for p in predictions
        if p.predicted == "contradiction" and p.ground_truth == "not_contradiction"
    )
    fn = sum(
        1
        for p in predictions
        if p.predicted == "not_contradiction" and p.ground_truth == "contradiction"
    )
    abstain = sum(1 for p in predictions if p.predicted == "uncertain")
    n_not = sum(1 for p in predictions if p.ground_truth == "not_contradiction")
    n_contra = sum(1 for p in predictions if p.ground_truth == "contradiction")
    accuracy = (correct / n) if n else 0.0
    fp_rate = (fp / n_not) if n_not else 0.0
    fn_rate = (fn / n_contra) if n_contra else 0.0
    abstain_rate = (abstain / n) if n else 0.0
    return BucketMetrics(
        bucket=bucket,
        n_pairs=n,
        n_correct=correct,
        n_false_positive=fp,
        n_false_negative=fn,
        n_abstain=abstain,
        accuracy=accuracy,
        false_positive_rate=fp_rate,
        false_negative_rate=fn_rate,
        abstain_rate=abstain_rate,
    )


def run_targeted_eval(
    pairs: Iterable[TargetedPair],
    *,
    judge: Judge,
) -> TargetedResult:
    """Score every pair through the judge and compute per-bucket metrics."""
    predictions: list[TargetedPrediction] = []
    for pair in pairs:
        a, b = _to_assertions(pair)
        verdict = judge.judge(a, b)
        raw = verdict.verdict
        if raw not in _VALID_GROUND_TRUTH:
            # Defensive: the judge may surface a label outside the three-value
            # enum (e.g. numeric_short_circuit) under future detector branches.
            # Collapse to the closest meaning rather than crashing the run.
            predicted: JudgePredicted = (
                "contradiction" if raw == "numeric_short_circuit" else "uncertain"
            )
        else:
            predicted = raw  # type: ignore[assignment]
        predictions.append(
            TargetedPrediction(
                pair_id=pair.pair_id,
                bucket=pair.bucket,
                ground_truth=pair.ground_truth,
                relation=pair.relation,
                predicted=predicted,
                judge_confidence=verdict.confidence,
                judge_rationale=verdict.rationale,
            )
        )

    by_bucket: dict[str, list[TargetedPrediction]] = {}
    for p in predictions:
        by_bucket.setdefault(p.bucket, []).append(p)
    per_bucket = [_bucket_metrics(b, preds) for b, preds in sorted(by_bucket.items())]
    overall = _bucket_metrics("__overall__", predictions)
    return TargetedResult(
        n_pairs=len(predictions),
        overall_accuracy=overall.accuracy,
        overall_false_positive_rate=overall.false_positive_rate,
        overall_false_negative_rate=overall.false_negative_rate,
        overall_abstain_rate=overall.abstain_rate,
        per_bucket=per_bucket,
        predictions=predictions,
    )


def format_markdown_summary(result: TargetedResult) -> str:
    """Render a markdown summary suitable for pasting into a PR description."""
    header = (
        "# Targeted-eval result\n"
        "\n"
        f"- Pairs scored: **{result.n_pairs}**\n"
        f"- Overall accuracy: **{result.overall_accuracy * 100:.1f}%**\n"
        f"- Overall false-positive rate: **{result.overall_false_positive_rate * 100:.1f}%**\n"
        f"- Overall false-negative rate: **{result.overall_false_negative_rate * 100:.1f}%**\n"
        f"- Abstain rate: **{result.overall_abstain_rate * 100:.1f}%**\n"
        "\n## Per-bucket\n"
        "\n"
        "| Bucket | n | accuracy | FP rate | FN rate | abstain |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    rows = [
        f"| {b.bucket} | {b.n_pairs} | {b.accuracy * 100:.1f}% | "
        f"{b.false_positive_rate * 100:.1f}% | {b.false_negative_rate * 100:.1f}% | "
        f"{b.abstain_rate * 100:.1f}% |"
        for b in result.per_bucket
    ]
    return header + "\n".join(rows) + "\n"


def write_metrics_json(result: TargetedResult, path: Path | str) -> None:
    payload = asdict(result)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
    parser = argparse.ArgumentParser(prog="targeted_eval_harness", description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "pairs.jsonl",
        help="Path to targeted-eval JSONL (defaults to bundled pairs.jsonl).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Where to write metrics JSON.")
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Optional markdown-summary path (e.g. for paste into a PR).",
    )
    parser.add_argument("--judge-provider", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--judge-model", default=None)
    args = parser.parse_args(argv)

    pairs = load_pairs(args.input)
    _log.info("Loaded %d targeted pairs from %s", len(pairs), args.input)
    judge = _make_judge(args.judge_provider, args.judge_model)
    result = run_targeted_eval(pairs, judge=judge)
    write_metrics_json(result, args.output)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(format_markdown_summary(result), encoding="utf-8")
    print(format_markdown_summary(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
