"""Ingest manual Claude-Projects baseline runs and compare against this codebase.

The semantic-analyst expert (triple-expert panel, 2026-05-15) specified
the "lazy baseline" against which the contradictionchecker pipeline must
justify its complexity:

1. Create a Claude.ai Project per corpus; upload all docs.
2. Use the fixed system + user prompts in ``prompts/``.
3. Run the same prompt 3 times (LLMs are stochastic).
4. Paste each run's JSON output into a ``runs/`` directory.
5. Compute the **union** of findings (recall measurement) and the
   **intersection** (precision-conservative measurement) across the 3 runs.
6. Hand-label the union; compute precision, recall against seeded
   contradictions you injected into the corpus, and cost per real
   contradiction found.

This module owns steps 4-6. The operator owns steps 1-3 (manual,
Claude.ai web UI) and the hand-labelling.

Schema for each baseline-run JSON file (one per run):

    {
      "run_id": "alpha_run_1",
      "model": "claude-opus-4-7",
      "corpus_name": "loan_package_alpha",
      "timestamp": "2026-05-15T14:30:00Z",
      "findings": [
        {
          "doc_a": "credit_agreement.txt",
          "span_a": "...",
          "doc_b": "term_sheet.txt",
          "span_b": "...",
          "type": "contradiction",
          "confidence": 0.85,
          "rationale": "..."
        }
      ],
      "input_tokens": 142000,
      "output_tokens": 3200,
      "cost_usd": 2.18
    }

Labels file format (CSV) - produced by ``write_findings_for_labelling``,
edited by the operator, consumed by ``load_labels``:

    finding_id,label,notes
    a1b2c3d4...,real,
    e5f6g7h8...,false_positive,override pair
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Literal

LabelValue = Literal["real", "false_positive", "dismissed"]
_VALID_LABELS: frozenset[str] = frozenset({"real", "false_positive", "dismissed"})


@dataclass(frozen=True, slots=True)
class BaselineFinding:
    """One finding returned by a single Claude-Projects baseline run."""

    finding_id: str
    doc_a: str
    doc_b: str
    span_a: str
    span_b: str
    finding_type: str
    confidence: float | None
    rationale: str


@dataclass(frozen=True, slots=True)
class BaselineRun:
    """One Claude.ai Project chat-output capture."""

    run_id: str
    model: str
    corpus_name: str
    findings: list[BaselineFinding]
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    timestamp: str | None = None


@dataclass(frozen=True, slots=True)
class FindingLabel:
    finding_id: str
    label: LabelValue
    notes: str = ""


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    """Aggregate comparison across 1..N baseline runs for one corpus."""

    corpus_name: str
    n_runs: int
    n_findings_total: int
    n_findings_union: int
    n_findings_intersection: int
    agreement_ratio: float
    labels_summary: dict[str, int]
    precision_intersection: float | None
    precision_union: float | None
    total_cost_usd: float
    cost_per_real_finding: float | None
    per_run_finding_counts: list[int] = field(default_factory=list)


# --- normalisation + dedup --------------------------------------------------


def _normalise_span(text: str) -> str:
    """Collapse whitespace + lowercase + truncate for content-keying.

    Span text drifts between runs (Claude paraphrases verbatim quotes
    occasionally); normalising lets re-runs of the same finding dedupe.
    The 120-char prefix is empirical: long enough to distinguish real
    findings, short enough to absorb minor wording variation.
    """
    return " ".join(text.lower().strip().split())[:120]


def _normalise_doc(name: str) -> str:
    return name.strip().lower()


def compute_finding_id(doc_a: str, span_a: str, doc_b: str, span_b: str) -> str:
    """Content-addressable id for a finding (order-independent on the doc-span pair).

    Sorts the ``(doc, span)`` pairs together so each span stays bound to its
    document. The naive approach -- sorting docs and spans independently --
    collides ``(A, "x", B, "y")`` with ``(A, "y", B, "x")``, which are
    semantically distinct findings on the same doc pair. The 16-char prefix
    matches the project's hash_id convention.
    """
    pair_a = (_normalise_doc(doc_a), _normalise_span(span_a))
    pair_b = (_normalise_doc(doc_b), _normalise_span(span_b))
    lo, hi = sorted([pair_a, pair_b])
    payload = "|".join([lo[0], lo[1], hi[0], hi[1]])
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


# --- load -------------------------------------------------------------------


def _build_finding(row: Mapping[str, object]) -> BaselineFinding:
    doc_a = str(row["doc_a"])
    doc_b = str(row["doc_b"])
    span_a = str(row["span_a"])
    span_b = str(row["span_b"])
    confidence_raw = row.get("confidence")
    confidence = float(confidence_raw) if confidence_raw is not None else None
    return BaselineFinding(
        finding_id=compute_finding_id(doc_a, span_a, doc_b, span_b),
        doc_a=doc_a,
        doc_b=doc_b,
        span_a=span_a,
        span_b=span_b,
        finding_type=str(row.get("type", "contradiction")),
        confidence=confidence,
        rationale=str(row.get("rationale", "")),
    )


def _strip_code_fences(text: str) -> str:
    """Tolerate Claude wrapping JSON in ```json ... ``` even when asked not to.

    Strips exactly the opening triple-backtick (and any immediately-following
    language tag) plus the closing triple-backtick. Does not eat additional
    backticks that may appear inside the content.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:]  # exactly the three opening backticks
        # drop optional language tag on the opening fence
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    return stripped


def load_baseline_run(path: Path | str) -> BaselineRun:
    """Load a single baseline-run JSON file.

    The ``findings`` field may itself be a JSON array (typical when
    Claude returns the raw array and the operator wrapped it) or a
    string containing JSON (typical when the operator pasted the raw
    chat output as a string field). Both are tolerated.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected an object at top level, got {type(payload).__name__}")
    findings_raw = payload.get("findings", [])
    if isinstance(findings_raw, str):
        try:
            findings_raw = json.loads(_strip_code_fences(findings_raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: 'findings' string is not valid JSON: {exc}") from exc
    if not isinstance(findings_raw, list):
        raise ValueError(f"{path}: 'findings' must be a list")
    findings = [_build_finding(row) for row in findings_raw]
    return BaselineRun(
        run_id=str(payload.get("run_id", Path(path).stem)),
        model=str(payload.get("model", "unknown")),
        corpus_name=str(payload.get("corpus_name", "unknown")),
        findings=findings,
        input_tokens=_optional_int(payload.get("input_tokens")),
        output_tokens=_optional_int(payload.get("output_tokens")),
        cost_usd=_optional_float(payload.get("cost_usd")),
        timestamp=_optional_str(payload.get("timestamp")),
    )


def load_baseline_runs(paths: Iterable[Path | str]) -> list[BaselineRun]:
    return [load_baseline_run(p) for p in paths]


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None  # type: ignore[arg-type]


def _optional_float(value: object) -> float | None:
    return float(value) if value is not None else None  # type: ignore[arg-type]


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


# --- union / intersection --------------------------------------------------


def union_findings(runs: Sequence[BaselineRun]) -> dict[str, BaselineFinding]:
    """All findings across all runs, deduped by content-address ``finding_id``.

    When the same id appears in multiple runs, the first occurrence wins —
    later runs may paraphrase the rationale but the underlying finding is
    the same content-keyed pair.
    """
    out: dict[str, BaselineFinding] = {}
    for run in runs:
        for f in run.findings:
            out.setdefault(f.finding_id, f)
    return out


def intersection_findings(runs: Sequence[BaselineRun]) -> dict[str, BaselineFinding]:
    """Findings present in *every* run, by ``finding_id``.

    With zero runs this is empty; with one run it equals the union.
    """
    if not runs:
        return {}
    common = {f.finding_id for f in runs[0].findings}
    for run in runs[1:]:
        common &= {f.finding_id for f in run.findings}
    # Resolve ids back to the first-occurrence BaselineFinding object.
    by_id = union_findings(runs)
    return {fid: by_id[fid] for fid in common}


# --- labels ----------------------------------------------------------------


def write_findings_for_labelling(findings: Mapping[str, BaselineFinding], path: Path | str) -> None:
    """Emit a CSV the operator hand-labels in a spreadsheet.

    Columns: finding_id, doc_a, span_a, doc_b, span_b, type, confidence,
    rationale, label, notes. ``label`` and ``notes`` are blank; the
    operator fills them in (label ∈ {real, false_positive, dismissed}).
    """
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "finding_id",
                "doc_a",
                "span_a",
                "doc_b",
                "span_b",
                "type",
                "confidence",
                "rationale",
                "label",
                "notes",
            ]
        )
        for f in findings.values():
            writer.writerow(
                [
                    f.finding_id,
                    f.doc_a,
                    f.span_a,
                    f.doc_b,
                    f.span_b,
                    f.finding_type,
                    "" if f.confidence is None else f"{f.confidence:.3f}",
                    f.rationale,
                    "",
                    "",
                ]
            )


def load_labels(path: Path | str) -> dict[str, FindingLabel]:
    """Load the operator-edited CSV. Rows with empty 'label' are skipped."""
    out: dict[str, FindingLabel] = {}
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for lineno, row in enumerate(reader, start=2):  # header is line 1
            label_raw = (row.get("label") or "").strip()
            if not label_raw:
                continue
            if label_raw not in _VALID_LABELS:
                raise ValueError(
                    f"{path}:{lineno}: invalid label {label_raw!r}; "
                    f"expected one of {sorted(_VALID_LABELS)}"
                )
            finding_id = (row.get("finding_id") or "").strip()
            if not finding_id:
                raise ValueError(f"{path}:{lineno}: missing finding_id")
            out[finding_id] = FindingLabel(
                finding_id=finding_id,
                label=label_raw,  # type: ignore[arg-type]
                notes=(row.get("notes") or "").strip(),
            )
    return out


# --- comparison metrics -----------------------------------------------------


def _precision(labels: Mapping[str, FindingLabel], ids: Iterable[str]) -> float | None:
    confirmed = 0
    false_positive = 0
    for fid in ids:
        label = labels.get(fid)
        if label is None or label.label == "dismissed":
            continue
        if label.label == "real":
            confirmed += 1
        elif label.label == "false_positive":
            false_positive += 1
    denom = confirmed + false_positive
    return (confirmed / denom) if denom else None


def compute_comparison(
    runs: Sequence[BaselineRun],
    labels: Mapping[str, FindingLabel] | None = None,
) -> BaselineComparison:
    """Aggregate metrics across the supplied runs.

    ``labels`` is optional; without it the precision and cost-per-real
    fields are ``None`` but the union/intersection/agreement metrics are
    still computed.
    """
    if not runs:
        raise ValueError("compute_comparison requires at least one run")
    union = union_findings(runs)
    inter = intersection_findings(runs)
    n_total = sum(len(r.findings) for r in runs)
    n_union = len(union)
    n_inter = len(inter)
    agreement = (n_inter / n_union) if n_union else 0.0

    has_labels = labels is not None
    if labels is None:
        labels = {}
    labels_summary: dict[str, int] = {"real": 0, "false_positive": 0, "dismissed": 0}
    for fid in union:
        lbl = labels.get(fid)
        if lbl is not None:
            labels_summary[lbl.label] += 1

    # Use the original 'labels is None' sentinel rather than truthiness so that
    # an explicit empty-dict argument distinguishes from 'no labels supplied'.
    precision_inter = _precision(labels, inter.keys()) if has_labels else None
    precision_union = _precision(labels, union.keys()) if has_labels else None

    total_cost = sum(r.cost_usd for r in runs if r.cost_usd is not None)
    n_real = labels_summary["real"]
    cost_per_real = (total_cost / n_real) if (n_real and total_cost) else None

    return BaselineComparison(
        corpus_name=runs[0].corpus_name,
        n_runs=len(runs),
        n_findings_total=n_total,
        n_findings_union=n_union,
        n_findings_intersection=n_inter,
        agreement_ratio=agreement,
        labels_summary=labels_summary,
        precision_intersection=precision_inter,
        precision_union=precision_union,
        total_cost_usd=total_cost,
        cost_per_real_finding=cost_per_real,
        per_run_finding_counts=[len(r.findings) for r in runs],
    )


def format_comparison_markdown(comparison: BaselineComparison) -> str:
    """Markdown summary suitable for pasting into a comparison PR."""
    lines = [
        f"# Claude-Projects baseline: {comparison.corpus_name}",
        "",
        f"- Runs: **{comparison.n_runs}** "
        f"(finding counts per run: {comparison.per_run_finding_counts})",
        f"- Findings total (with duplicates across runs): **{comparison.n_findings_total}**",
        f"- Findings union (recall set): **{comparison.n_findings_union}**",
        f"- Findings intersection (precision-conservative set): "
        f"**{comparison.n_findings_intersection}**",
        f"- Cross-run agreement ratio "
        f"(intersection / union): **{comparison.agreement_ratio * 100:.1f}%**",
        "",
    ]
    if comparison.labels_summary["real"] or comparison.labels_summary["false_positive"]:
        prec_inter = (
            "n/a"
            if comparison.precision_intersection is None
            else f"{comparison.precision_intersection * 100:.1f}%"
        )
        prec_union = (
            "n/a"
            if comparison.precision_union is None
            else f"{comparison.precision_union * 100:.1f}%"
        )
        lines += [
            "## Labels",
            "",
            f"- Real: **{comparison.labels_summary['real']}**",
            f"- False positive: **{comparison.labels_summary['false_positive']}**",
            f"- Dismissed: **{comparison.labels_summary['dismissed']}**",
            f"- Precision on intersection: **{prec_inter}**",
            f"- Precision on union: **{prec_union}**",
            "",
        ]
    if comparison.total_cost_usd:
        cost_per_real = (
            "n/a"
            if comparison.cost_per_real_finding is None
            else f"${comparison.cost_per_real_finding:.2f}"
        )
        lines += [
            "## Cost",
            "",
            f"- Total cost across runs: **${comparison.total_cost_usd:.2f}**",
            f"- Cost per real finding: **{cost_per_real}**",
            "",
        ]
    lines.append("## Stop-rule (semantic-analyst, adopted 2026-05-15)")
    lines.append("")
    lines.append(
        "Kill the pairwise contradiction core if precision_intersection <= 15% AND "
        "this baseline lands within 10 points of the pipeline on precision with "
        ">=80% of the seeded-contradiction recall."
    )
    return "\n".join(lines) + "\n"


def write_metrics_json(comparison: BaselineComparison, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(comparison), indent=2) + "\n", encoding="utf-8")


# --- CLI -------------------------------------------------------------------


def _cmd_dedupe(args: argparse.Namespace) -> int:
    runs = load_baseline_runs(args.runs)
    union = union_findings(runs)
    write_findings_for_labelling(union, args.out)
    print(f"Wrote {len(union)} deduped findings (union across {len(runs)} runs) to {args.out}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    runs = load_baseline_runs(args.runs)
    labels = load_labels(args.labels) if args.labels else None
    comparison = compute_comparison(runs, labels=labels)
    if args.json_out:
        write_metrics_json(comparison, args.json_out)
    md = format_comparison_markdown(comparison)
    if args.markdown_out:
        Path(args.markdown_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_out).write_text(md, encoding="utf-8")
    print(md)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="claude_projects_baseline",
        description="Ingest manual Claude-Projects baseline runs and compute metrics.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dedupe = sub.add_parser(
        "dedupe",
        help=("Load N baseline runs, dedupe to the union, write a labelling CSV."),
    )
    p_dedupe.add_argument("runs", nargs="+", type=Path, help="Paths to baseline-run JSON files.")
    p_dedupe.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    p_dedupe.set_defaults(func=_cmd_dedupe)

    p_compare = sub.add_parser(
        "compare",
        help="Compute union/intersection/agreement + optional precision from labels.",
    )
    p_compare.add_argument("runs", nargs="+", type=Path, help="Paths to baseline-run JSON files.")
    p_compare.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional CSV of operator-set labels (from `dedupe`).",
    )
    p_compare.add_argument(
        "--json-out", type=Path, default=None, help="Optional JSON metrics output path."
    )
    p_compare.add_argument(
        "--markdown-out", type=Path, default=None, help="Optional markdown summary output path."
    )
    p_compare.set_defaults(func=_cmd_compare)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
