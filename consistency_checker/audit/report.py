"""Markdown report generation for confirmed contradictions.

Reads from the audit ``findings`` table plus the canonical ``documents`` and
``assertions`` tables. Output is deterministic (no timestamps in the body)
so reports diff cleanly across runs and round-trip through golden-file tests.

Findings are grouped by ``(doc_a, doc_b)`` pair, ordered by max confidence
within group, then by descending confidence within each group. Only
``contradiction``-verdict findings are emitted; ``uncertain`` and
``not_contradiction`` are filtered out (they live in the audit DB for replay
and tuning).
"""

from __future__ import annotations

from collections import defaultdict

from consistency_checker.audit.logger import AuditLogger, Finding, MultiPartyFinding
from consistency_checker.extract.schema import Assertion, Document
from consistency_checker.index.assertion_store import AssertionStore

VERDICT_LABELS: dict[str, str] = {
    "confirmed": "Real issue",
    "false_positive": "Not an issue",
    "dismissed": "Dismissed",
}


def _format_score(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def _quote_spans(spans: list[str]) -> str:
    if not spans:
        return ""
    return ", ".join(f"`{s}`" for s in spans)


def render_report(
    store: AssertionStore,
    audit_logger: AuditLogger,
    *,
    run_id: str,
    min_confidence: float = 0.0,
) -> str:
    """Render a deterministic markdown report for the given run.

    Args:
        store: canonical SQLite store, used to resolve assertion & document text.
        audit_logger: audit logger bound to the same store; supplies findings.
        run_id: only findings from this run are emitted.
        min_confidence: lower bound on judge confidence (default 0.0 → include all).
    """
    run = audit_logger.get_run(run_id)
    # Both LLM contradictions and deterministic numeric short-circuits land in
    # the report. iter_findings(verdict=...) can only filter on one label at a
    # time, so we make two passes and merge — cheaper than fetching all rows
    # and filtering in Python at scale.
    contradictions = [
        f
        for f in (
            *audit_logger.iter_findings(run_id=run_id, verdict="contradiction"),
            *audit_logger.iter_findings(run_id=run_id, verdict="numeric_short_circuit"),
        )
        if (f.judge_confidence or 0.0) >= min_confidence
    ]

    contradiction_keys = [
        (":".join(sorted([f.assertion_a_id, f.assertion_b_id])), "contradiction")
        for f in contradictions
    ]
    contradiction_verdicts = audit_logger.get_reviewer_verdicts_bulk(contradiction_keys)
    contradictions = [
        f
        for f in contradictions
        if (
            contradiction_verdicts.get(
                (":".join(sorted([f.assertion_a_id, f.assertion_b_id])), "contradiction")
            )
            is None
            or contradiction_verdicts[
                (":".join(sorted([f.assertion_a_id, f.assertion_b_id])), "contradiction")
            ].verdict
            != "false_positive"
        )
    ]

    lines: list[str] = []
    lines.append("# Consistency check report")
    lines.append("")
    if run is not None:
        lines.append(f"- Run: `{run_id}`")
        lines.append(f"- Assertions scanned: {run.n_assertions}")
        lines.append(f"- Candidate pairs (gated): {run.n_pairs_gated}")
        lines.append(f"- Pairs judged: {run.n_pairs_judged}")
        lines.append(f"- Contradictions found: {len(contradictions)}")
    else:
        lines.append(f"- Run: `{run_id}` (no metadata)")
        lines.append(f"- Contradictions found: {len(contradictions)}")
    lines.append("")

    if not contradictions:
        lines.append("_No contradictions met the reporting threshold._")
        lines.append("")
        _append_multi_party_section(
            lines, audit_logger, store, run_id=run_id, min_confidence=min_confidence
        )
        _append_definition_section(
            lines, audit_logger, store, run_id=run_id, min_confidence=min_confidence
        )
        return "\n".join(lines) + "\n"

    # --- summary table ------------------------------------------------------
    lines.append("## Summary")
    lines.append("")
    lines.append("| Confidence | NLI(p_contradiction) | Doc A | Doc B | Rationale |")
    lines.append("| --- | --- | --- | --- | --- |")

    # Resolve everything once.
    assertions: dict[str, Assertion] = {}
    documents: dict[str, Document] = {}
    grouped: dict[tuple[str, str], list[Finding]] = defaultdict(list)

    for finding in contradictions:
        for aid in (finding.assertion_a_id, finding.assertion_b_id):
            if aid not in assertions:
                got = store.get_assertion(aid)
                if got is not None:
                    assertions[aid] = got
                    if got.doc_id not in documents:
                        doc = store.get_document(got.doc_id)
                        if doc is not None:
                            documents[got.doc_id] = doc

        a = assertions.get(finding.assertion_a_id)
        b = assertions.get(finding.assertion_b_id)
        if a is None or b is None:
            continue
        doc_pair = tuple(sorted([a.doc_id, b.doc_id]))
        # mypy: tuple(sorted(...)) is tuple[str, ...]; we know it's 2 elements
        grouped[(doc_pair[0], doc_pair[1])].append(finding)

    # Summary rows, sorted by confidence desc.
    summary_sorted = sorted(
        contradictions,
        key=lambda f: -(f.judge_confidence or 0.0),
    )
    for finding in summary_sorted:
        a = assertions.get(finding.assertion_a_id)
        b = assertions.get(finding.assertion_b_id)
        if a is None or b is None:
            continue
        doc_a = documents.get(a.doc_id)
        doc_b = documents.get(b.doc_id)
        doc_a_label = doc_a.title or doc_a.source_path if doc_a else a.doc_id
        doc_b_label = doc_b.title or doc_b.source_path if doc_b else b.doc_id
        rationale_short = (finding.judge_rationale or "").splitlines()[0][:120]
        lines.append(
            "| "
            f"{_format_score(finding.judge_confidence)} | "
            f"{_format_score(finding.nli_p_contradiction)} | "
            f"{doc_a_label} | "
            f"{doc_b_label} | "
            f"{rationale_short} |"
        )
    lines.append("")

    # --- per-pair details ---------------------------------------------------
    lines.append("## Findings")
    lines.append("")
    # Stable group order: by (doc_a, doc_b) ids.
    for doc_a_id, doc_b_id in sorted(grouped.keys()):
        items = sorted(
            grouped[(doc_a_id, doc_b_id)],
            key=lambda f: -(f.judge_confidence or 0.0),
        )
        doc_a = documents.get(doc_a_id)
        doc_b = documents.get(doc_b_id)
        a_label = doc_a.title or doc_a.source_path if doc_a else doc_a_id
        b_label = doc_b.title or doc_b.source_path if doc_b else doc_b_id

        lines.append(f"### {a_label} ⇄ {b_label}")
        lines.append("")
        for finding in items:
            a = assertions.get(finding.assertion_a_id)
            b = assertions.get(finding.assertion_b_id)
            if a is None or b is None:
                continue
            lines.append(f"#### Finding `{finding.finding_id}`")
            lines.append("")
            lines.append(f"- **Confidence:** {_format_score(finding.judge_confidence)}")
            lines.append(
                f"- **NLI p(contradiction):** {_format_score(finding.nli_p_contradiction)}"
            )
            lines.append(f"- **Gate score:** {_format_score(finding.gate_score)}")
            spans = _quote_spans(finding.evidence_spans)
            if spans:
                lines.append(f"- **Evidence spans:** {spans}")
            lines.append("")
            lines.append(f"> **A** ({a_label}): {a.assertion_text}")
            lines.append("")
            lines.append(f"> **B** ({b_label}): {b.assertion_text}")
            lines.append("")
            pk = ":".join(sorted([finding.assertion_a_id, finding.assertion_b_id]))
            rv = contradiction_verdicts.get((pk, "contradiction"))
            reviewer_label = VERDICT_LABELS[rv.verdict] if rv else "Pending review"
            lines.append(f"**Reviewer:** {reviewer_label}")
            lines.append("")
            if finding.judge_rationale:
                lines.append(f"**Rationale.** {finding.judge_rationale}")
                lines.append("")

    _append_multi_party_section(
        lines,
        audit_logger,
        store,
        run_id=run_id,
        min_confidence=min_confidence,
    )

    _append_definition_section(
        lines,
        audit_logger,
        store,
        run_id=run_id,
        min_confidence=min_confidence,
    )

    return "\n".join(lines) + "\n"


def _append_multi_party_section(
    lines: list[str],
    audit_logger: AuditLogger,
    store: AssertionStore,
    *,
    run_id: str,
    min_confidence: float,
) -> None:
    """Render the optional multi-document conditional contradictions section.

    The section is emitted only when the run produced multi-party findings, so
    a normal (pair-only) run's report stays byte-stable against existing
    golden files.
    """
    multi: list[MultiPartyFinding] = [
        f
        for f in audit_logger.iter_multi_party_findings(
            run_id=run_id, verdict="multi_party_contradiction"
        )
        if (f.judge_confidence or 0.0) >= min_confidence
    ]

    multi_keys = [(":".join(sorted(f.assertion_ids)), "multi_party") for f in multi]
    multi_verdicts = audit_logger.get_reviewer_verdicts_bulk(multi_keys)
    multi = [
        f
        for f in multi
        if (
            multi_verdicts.get((":".join(sorted(f.assertion_ids)), "multi_party")) is None
            or multi_verdicts[(":".join(sorted(f.assertion_ids)), "multi_party")].verdict
            != "false_positive"
        )
    ]

    if not multi:
        return

    lines.append("## Multi-document conditional contradictions")
    lines.append("")
    multi_sorted = sorted(multi, key=lambda f: (-(f.judge_confidence or 0.0), f.finding_id))
    for finding in multi_sorted:
        lines.append(f"### Finding `{finding.finding_id}`")
        lines.append("")
        lines.append(f"- **Confidence:** {_format_score(finding.judge_confidence)}")
        if finding.triangle_edge_scores:
            min_edge = min(score for _, _, score in finding.triangle_edge_scores)
            lines.append(f"- **Min edge score:** {_format_score(min_edge)}")
        spans = _quote_spans(finding.evidence_spans)
        if spans:
            lines.append(f"- **Evidence spans:** {spans}")
        lines.append("")
        for label, aid in zip(("A", "B", "C"), finding.assertion_ids, strict=False):
            assertion = store.get_assertion(aid)
            if assertion is None:
                continue
            doc = store.get_document(assertion.doc_id)
            doc_label = doc.title or doc.source_path if doc else assertion.doc_id
            lines.append(f"> **{label}** ({doc_label}): {assertion.assertion_text}")
            lines.append("")
        mp_pk = ":".join(sorted(finding.assertion_ids))
        mp_rv = multi_verdicts.get((mp_pk, "multi_party"))
        mp_reviewer_label = VERDICT_LABELS[mp_rv.verdict] if mp_rv else "Pending review"
        lines.append(f"**Reviewer:** {mp_reviewer_label}")
        lines.append("")
        if finding.judge_rationale:
            lines.append(f"**Rationale.** {finding.judge_rationale}")
            lines.append("")


def _append_definition_section(
    lines: list[str],
    audit_logger: AuditLogger,
    store: AssertionStore,
    *,
    run_id: str,
    min_confidence: float,
) -> None:
    """Render the optional definition-inconsistencies section.

    Emitted only when the run produced at least one `definition_divergent`
    finding so prior-shape runs stay byte-stable.
    """
    findings = [
        f
        for f in audit_logger.iter_findings(
            run_id=run_id,
            verdict="definition_divergent",
            detector_type="definition_inconsistency",
        )
        if (f.judge_confidence or 0.0) >= min_confidence
    ]

    def_keys = [
        (":".join(sorted([f.assertion_a_id, f.assertion_b_id])), "definition_inconsistency")
        for f in findings
    ]
    def_verdicts = audit_logger.get_reviewer_verdicts_bulk(def_keys)
    findings = [
        f
        for f in findings
        if (
            def_verdicts.get(
                (":".join(sorted([f.assertion_a_id, f.assertion_b_id])), "definition_inconsistency")
            )
            is None
            or def_verdicts[
                (":".join(sorted([f.assertion_a_id, f.assertion_b_id])), "definition_inconsistency")
            ].verdict
            != "false_positive"
        )
    ]

    if not findings:
        return

    # Resolve assertions + canonical terms (term lives on the assertion row).
    assertions: dict[str, Assertion] = {}
    documents: dict[str, Document] = {}
    for finding in findings:
        for aid in (finding.assertion_a_id, finding.assertion_b_id):
            if aid in assertions:
                continue
            got = store.get_assertion(aid)
            if got is None:
                continue
            assertions[aid] = got
            if got.doc_id not in documents:
                doc = store.get_document(got.doc_id)
                if doc is not None:
                    documents[got.doc_id] = doc

    # Group by the term on assertion A.
    grouped: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        a = assertions.get(finding.assertion_a_id)
        if a is None or a.term is None:
            continue
        grouped[a.term].append(finding)

    lines.append("## Definition inconsistencies")
    lines.append("")
    for term in sorted(grouped.keys(), key=str.lower):
        items = sorted(grouped[term], key=lambda f: -(f.judge_confidence or 0.0))
        lines.append(f'### "{term}"')
        lines.append("")
        for finding in items:
            a = assertions.get(finding.assertion_a_id)
            b = assertions.get(finding.assertion_b_id)
            if a is None or b is None:
                continue
            doc_a = documents.get(a.doc_id)
            doc_b = documents.get(b.doc_id)
            a_label = doc_a.title or doc_a.source_path if doc_a else a.doc_id
            b_label = doc_b.title or doc_b.source_path if doc_b else b.doc_id
            lines.append(f"- **{a_label}**: {a.assertion_text}")
            lines.append(f"- **{b_label}**: {b.assertion_text}")
            lines.append("")
            def_pk = ":".join(sorted([finding.assertion_a_id, finding.assertion_b_id]))
            def_rv = def_verdicts.get((def_pk, "definition_inconsistency"))
            def_reviewer_label = VERDICT_LABELS[def_rv.verdict] if def_rv else "Pending review"
            lines.append(f"**Reviewer:** {def_reviewer_label}")
            lines.append("")
            lines.append(f"**Verdict:** {finding.judge_verdict} — {finding.judge_rationale or ''}")
            lines.append("")
