"""Tests for the Claude-Projects baseline ingestion harness."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmarks.claude_projects_baseline.harness import (
    BaselineFinding,
    BaselineRun,
    FindingLabel,
    _strip_code_fences,
    compute_comparison,
    compute_finding_id,
    format_comparison_markdown,
    intersection_findings,
    load_baseline_run,
    load_baseline_runs,
    load_labels,
    main,
    union_findings,
    write_findings_for_labelling,
    write_metrics_json,
)

# --- compute_finding_id ---------------------------------------------------


def test_finding_id_order_independent_on_doc_pair() -> None:
    a = compute_finding_id("doc_a.txt", "X is 5%", "doc_b.txt", "X is 8%")
    b = compute_finding_id("doc_b.txt", "X is 8%", "doc_a.txt", "X is 5%")
    assert a == b


def test_finding_id_distinct_for_different_spans() -> None:
    a = compute_finding_id("doc_a.txt", "X is 5%", "doc_b.txt", "X is 8%")
    b = compute_finding_id("doc_a.txt", "X is 5%", "doc_b.txt", "Y is 8%")
    assert a != b


def test_finding_id_tolerates_whitespace_and_case() -> None:
    """Span paraphrases that only differ in whitespace/case must dedupe."""
    a = compute_finding_id("doc_a.txt", "X is 5%", "doc_b.txt", "X is 8%")
    b = compute_finding_id("DOC_A.TXT", "   x is 5%   ", "doc_b.txt", "x  is  8%")
    assert a == b


def test_finding_id_truncates_long_spans_safely() -> None:
    """Span text past the 120-char prefix doesn't change the id (dedups paraphrase suffixes)."""
    shared_prefix = "x" * 130  # > 120-char truncation boundary
    a = compute_finding_id("d1", shared_prefix + "suffix-A", "d2", "Y is below")
    b = compute_finding_id("d1", shared_prefix + "suffix-B-different", "d2", "Y is below")
    assert a == b


def test_finding_id_distinct_when_first_120_chars_differ() -> None:
    """Spans that differ within the 120-char prefix must not collide."""
    a = compute_finding_id("d1", "X is exactly 5%", "d2", "Y is below")
    b = compute_finding_id("d1", "X is exactly 8%", "d2", "Y is below")
    assert a != b


def test_finding_id_keeps_span_bound_to_doc() -> None:
    """Two findings on the same doc pair with swapped span roles are distinct.

    Naive sort-docs-and-spans-independently would collide these. Sorting
    (doc, span) pairs together keeps each span bound to its document.
    """
    a = compute_finding_id("docA", "first span", "docB", "second span")
    b = compute_finding_id("docA", "second span", "docB", "first span")
    assert a != b


# --- load_baseline_run ----------------------------------------------------


def _write_run(
    tmp_path: Path,
    *,
    name: str = "run.json",
    run_id: str = "r1",
    corpus: str = "alpha",
    findings: list[dict] | str | None = None,
    cost: float | None = 1.5,
    input_tokens: int | None = 100_000,
) -> Path:
    p = tmp_path / name
    payload: dict[str, object] = {
        "run_id": run_id,
        "model": "claude-opus-4-7",
        "corpus_name": corpus,
        "findings": [] if findings is None else findings,
    }
    if cost is not None:
        payload["cost_usd"] = cost
    if input_tokens is not None:
        payload["input_tokens"] = input_tokens
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_load_baseline_run_parses_findings(tmp_path: Path) -> None:
    p = _write_run(
        tmp_path,
        findings=[
            {
                "doc_a": "credit.txt",
                "span_a": "Interest 5%",
                "doc_b": "term_sheet.txt",
                "span_b": "Interest 8%",
                "type": "contradiction",
                "confidence": 0.9,
                "rationale": "Numeric mismatch",
            }
        ],
    )
    run = load_baseline_run(p)
    assert run.run_id == "r1"
    assert run.corpus_name == "alpha"
    assert len(run.findings) == 1
    f = run.findings[0]
    assert f.doc_a == "credit.txt"
    assert f.confidence == pytest.approx(0.9)
    assert f.finding_type == "contradiction"
    assert run.cost_usd == pytest.approx(1.5)
    assert run.input_tokens == 100_000


def test_load_baseline_run_tolerates_findings_as_string(tmp_path: Path) -> None:
    """Operators sometimes paste the raw Claude output as a string field."""
    raw = json.dumps(
        [
            {
                "doc_a": "a",
                "span_a": "x",
                "doc_b": "b",
                "span_b": "y",
                "type": "contradiction",
                "confidence": 0.7,
                "rationale": "r",
            }
        ]
    )
    p = _write_run(tmp_path, findings=raw)
    run = load_baseline_run(p)
    assert len(run.findings) == 1


def test_load_baseline_run_strips_code_fences(tmp_path: Path) -> None:
    """Claude sometimes wraps JSON in ```json fences despite the prompt asking not to."""
    fenced = (
        "```json\n"
        + json.dumps(
            [
                {
                    "doc_a": "a",
                    "span_a": "x",
                    "doc_b": "b",
                    "span_b": "y",
                    "type": "contradiction",
                    "confidence": 0.5,
                    "rationale": "r",
                }
            ]
        )
        + "\n```"
    )
    p = _write_run(tmp_path, findings=fenced)
    run = load_baseline_run(p)
    assert len(run.findings) == 1


def test_strip_code_fences_handles_plain_text() -> None:
    """No-op on already-clean JSON strings."""
    text = '[{"a": 1}]'
    assert _strip_code_fences(text) == text


def test_strip_code_fences_handles_fence_without_language_tag() -> None:
    """Bare ``` fence (no 'json' tag) still strips cleanly."""
    fenced = "```\n[1, 2, 3]\n```"
    assert _strip_code_fences(fenced) == "[1, 2, 3]"


def test_strip_code_fences_strips_only_three_opening_backticks() -> None:
    """Extra backticks past the opening three remain in the content (not eaten by lstrip)."""
    # Pathological case: content begins with backticks itself.
    fenced = "```\n`leading backtick stays`\n```"
    out = _strip_code_fences(fenced)
    assert out == "`leading backtick stays`"


def test_load_baseline_run_rejects_top_level_array(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected an object at top level"):
        load_baseline_run(p)


def test_load_baseline_run_rejects_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_baseline_run(p)


def test_load_baseline_run_rejects_string_findings_not_json(tmp_path: Path) -> None:
    p = _write_run(tmp_path, findings="totally not json")
    with pytest.raises(ValueError, match="'findings' string is not valid JSON"):
        load_baseline_run(p)


def test_load_baseline_run_rejects_findings_not_a_list(tmp_path: Path) -> None:
    p = _write_run(tmp_path, findings=None)  # findings defaults to []; manually break it
    payload = json.loads(p.read_text(encoding="utf-8"))
    payload["findings"] = {"not": "a list"}
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="'findings' must be a list"):
        load_baseline_run(p)


def test_load_baseline_runs_loads_multiple(tmp_path: Path) -> None:
    p1 = _write_run(tmp_path, name="a.json")
    p2 = _write_run(tmp_path, name="b.json")
    runs = load_baseline_runs([p1, p2])
    assert len(runs) == 2


def test_load_baseline_run_handles_missing_optional_fields(tmp_path: Path) -> None:
    """Optional cost / token fields should be None when absent from the JSON."""
    p = _write_run(tmp_path, cost=None, input_tokens=None)
    run = load_baseline_run(p)
    assert run.cost_usd is None
    assert run.input_tokens is None
    assert run.output_tokens is None


# --- union + intersection -------------------------------------------------


def _finding(
    doc_a: str, span_a: str, doc_b: str, span_b: str, *, conf: float = 0.9
) -> BaselineFinding:
    return BaselineFinding(
        finding_id=compute_finding_id(doc_a, span_a, doc_b, span_b),
        doc_a=doc_a,
        doc_b=doc_b,
        span_a=span_a,
        span_b=span_b,
        finding_type="contradiction",
        confidence=conf,
        rationale="r",
    )


def _run(run_id: str, findings: list[BaselineFinding], *, cost: float | None = None) -> BaselineRun:
    return BaselineRun(
        run_id=run_id,
        model="m",
        corpus_name="alpha",
        findings=findings,
        cost_usd=cost,
    )


def test_union_dedupes_across_runs() -> None:
    f1 = _finding("a.txt", "X is 5%", "b.txt", "X is 8%")
    f2 = _finding("a.txt", "Y is on", "b.txt", "Y is off")
    runs = [_run("r1", [f1, f2]), _run("r2", [f1])]
    union = union_findings(runs)
    assert set(union.keys()) == {f1.finding_id, f2.finding_id}


def test_intersection_returns_findings_in_all_runs() -> None:
    f1 = _finding("a", "X 5%", "b", "X 8%")
    f2 = _finding("a", "Y on", "b", "Y off")
    f3 = _finding("a", "Z lo", "b", "Z hi")
    runs = [_run("r1", [f1, f2]), _run("r2", [f1, f3]), _run("r3", [f1, f2, f3])]
    inter = intersection_findings(runs)
    assert set(inter.keys()) == {f1.finding_id}


def test_intersection_empty_input_returns_empty() -> None:
    assert intersection_findings([]) == {}


def test_union_empty_input_returns_empty() -> None:
    assert union_findings([]) == {}


def test_intersection_single_run_equals_union() -> None:
    f1 = _finding("a", "X 5%", "b", "X 8%")
    runs = [_run("r1", [f1])]
    assert intersection_findings(runs).keys() == union_findings(runs).keys()


# --- labels round-trip ----------------------------------------------------


def test_write_and_load_labels_round_trip(tmp_path: Path) -> None:
    f1 = _finding("a", "X 5%", "b", "X 8%")
    f2 = _finding("a", "Y on", "b", "Y off")
    union = {f1.finding_id: f1, f2.finding_id: f2}
    csv_path = tmp_path / "labels.csv"
    write_findings_for_labelling(union, csv_path)
    # Operator edits: fill in two labels, leave one blank to test skip-blank.
    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows[0]["label"] = "real"
    rows[0]["notes"] = "confirmed by underwriter"
    rows[1]["label"] = ""  # blank → skipped on load
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    labels = load_labels(csv_path)
    assert len(labels) == 1
    lbl = next(iter(labels.values()))
    assert lbl.label == "real"
    assert lbl.notes == "confirmed by underwriter"


def test_load_labels_rejects_invalid_label(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        "finding_id,doc_a,span_a,doc_b,span_b,type,confidence,rationale,label,notes\n"
        "abc,a,x,b,y,contradiction,0.9,r,verified_real,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid label 'verified_real'"):
        load_labels(csv_path)


def test_load_labels_rejects_missing_finding_id(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        "finding_id,doc_a,span_a,doc_b,span_b,type,confidence,rationale,label,notes\n"
        ",a,x,b,y,contradiction,0.9,r,real,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing finding_id"):
        load_labels(csv_path)


# --- compute_comparison ---------------------------------------------------


def test_compute_comparison_without_labels_still_works() -> None:
    f1 = _finding("a", "X 5%", "b", "X 8%")
    f2 = _finding("a", "Y on", "b", "Y off")
    runs = [_run("r1", [f1, f2], cost=2.0), _run("r2", [f1], cost=1.5)]
    cmp = compute_comparison(runs)
    assert cmp.n_runs == 2
    assert cmp.n_findings_total == 3
    assert cmp.n_findings_union == 2
    assert cmp.n_findings_intersection == 1
    assert cmp.agreement_ratio == 0.5
    assert cmp.total_cost_usd == pytest.approx(3.5)
    assert cmp.precision_intersection is None
    assert cmp.precision_union is None
    assert cmp.cost_per_real_finding is None
    assert cmp.per_run_finding_counts == [2, 1]


def test_compute_comparison_with_labels_computes_precision() -> None:
    f1 = _finding("a", "X 5%", "b", "X 8%")  # in intersection, labelled real
    f2 = _finding("a", "Y on", "b", "Y off")  # in intersection, labelled FP
    f3 = _finding("a", "Z lo", "b", "Z hi")  # in union only (run1), labelled real
    runs = [_run("r1", [f1, f2, f3], cost=2.0), _run("r2", [f1, f2], cost=2.0)]
    labels = {
        f1.finding_id: FindingLabel(f1.finding_id, "real"),
        f2.finding_id: FindingLabel(f2.finding_id, "false_positive"),
        f3.finding_id: FindingLabel(f3.finding_id, "real"),
    }
    cmp = compute_comparison(runs, labels=labels)
    assert cmp.precision_intersection == pytest.approx(0.5)  # 1 real / (1 real + 1 FP)
    assert cmp.precision_union == pytest.approx(2 / 3)  # 2 real / (2 real + 1 FP)
    assert cmp.labels_summary == {"real": 2, "false_positive": 1, "dismissed": 0}
    assert cmp.cost_per_real_finding == pytest.approx(4.0 / 2)


def test_compute_comparison_dismissed_excluded_from_precision_denominator() -> None:
    f1 = _finding("a", "X 5%", "b", "X 8%")
    runs = [_run("r1", [f1], cost=1.0)]
    labels = {f1.finding_id: FindingLabel(f1.finding_id, "dismissed")}
    cmp = compute_comparison(runs, labels=labels)
    assert cmp.precision_intersection is None
    assert cmp.precision_union is None


def test_compute_comparison_stale_labels_csv_ignored_silently() -> None:
    """Labels for finding_ids not in any run must not inflate label counts or precision."""
    f1 = _finding("a", "X 5%", "b", "X 8%")
    runs = [_run("r1", [f1], cost=1.0)]
    # Operator's labels CSV is stale: includes a finding id from a prior run
    # that no longer appears in the current run set.
    stale_id = "deadbeefdeadbeef"
    labels = {
        f1.finding_id: FindingLabel(f1.finding_id, "real"),
        stale_id: FindingLabel(stale_id, "real"),
    }
    cmp = compute_comparison(runs, labels=labels)
    assert cmp.labels_summary["real"] == 1  # only f1, not the stale row
    assert cmp.precision_intersection == 1.0


def test_compute_comparison_explicit_empty_labels_dict_distinguishes_from_none() -> None:
    """Passing labels={} (empty dict) means 'no real labels found' rather than 'never asked'.

    Both currently return precision=None for empty/no-matching labels, but the
    has_labels=True path also runs the labels_summary loop -- worth covering.
    """
    f1 = _finding("a", "X 5%", "b", "X 8%")
    runs = [_run("r1", [f1], cost=1.0)]
    cmp_explicit = compute_comparison(runs, labels={})
    cmp_none = compute_comparison(runs, labels=None)
    assert cmp_explicit.precision_intersection is None
    assert cmp_none.precision_intersection is None


def test_compute_comparison_raises_on_no_runs() -> None:
    with pytest.raises(ValueError, match="at least one run"):
        compute_comparison([])


# --- formatters + round-trip files ----------------------------------------


def test_format_comparison_markdown_includes_corpus_and_counts() -> None:
    f1 = _finding("a", "X 5%", "b", "X 8%")
    runs = [_run("r1", [f1], cost=1.0)]
    cmp = compute_comparison(runs)
    md = format_comparison_markdown(cmp)
    assert "alpha" in md
    assert "Runs: **1**" in md
    assert "Findings union" in md
    assert "Stop-rule" in md


def test_format_comparison_markdown_with_labels_renders_precision() -> None:
    f1 = _finding("a", "X 5%", "b", "X 8%")
    runs = [_run("r1", [f1], cost=1.0)]
    labels = {f1.finding_id: FindingLabel(f1.finding_id, "real")}
    cmp = compute_comparison(runs, labels=labels)
    md = format_comparison_markdown(cmp)
    assert "Precision on intersection: **100.0%**" in md
    assert "Cost per real finding" in md


def test_write_metrics_json_round_trip(tmp_path: Path) -> None:
    f1 = _finding("a", "X 5%", "b", "X 8%")
    runs = [_run("r1", [f1], cost=1.0)]
    cmp = compute_comparison(runs)
    out = tmp_path / "metrics.json"
    write_metrics_json(cmp, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["corpus_name"] == "alpha"
    assert payload["n_runs"] == 1


# --- CLI -----------------------------------------------------------------


def test_main_dedupe_writes_labelling_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p1 = _write_run(
        tmp_path,
        name="r1.json",
        findings=[
            {
                "doc_a": "a",
                "span_a": "X is 5%",
                "doc_b": "b",
                "span_b": "X is 8%",
                "type": "contradiction",
                "confidence": 0.9,
                "rationale": "r",
            }
        ],
    )
    p2 = _write_run(
        tmp_path,
        name="r2.json",
        findings=[
            {
                "doc_a": "a",
                "span_a": "X is 5%",
                "doc_b": "b",
                "span_b": "X is 8%",
                "type": "contradiction",
                "confidence": 0.7,
                "rationale": "same",
            }
        ],
    )
    out = tmp_path / "labels.csv"
    rc = main(["dedupe", str(p1), str(p2), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    with open(out, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # Same finding across both runs -> 1 deduped row.
    assert len(rows) == 1
    captured = capsys.readouterr()
    assert "1 deduped findings" in captured.out


def test_main_compare_writes_metrics_and_markdown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p1 = _write_run(
        tmp_path,
        name="r1.json",
        findings=[
            {
                "doc_a": "a",
                "span_a": "X is 5%",
                "doc_b": "b",
                "span_b": "X is 8%",
                "type": "contradiction",
                "confidence": 0.9,
                "rationale": "r",
            }
        ],
    )
    json_out = tmp_path / "metrics.json"
    md_out = tmp_path / "metrics.md"
    rc = main(
        [
            "compare",
            str(p1),
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(md_out),
        ]
    )
    assert rc == 0
    assert json_out.exists()
    assert md_out.exists()
    captured = capsys.readouterr()
    assert "Findings union" in captured.out
