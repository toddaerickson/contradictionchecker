"""Mine ``reviewer_verdicts`` into per-detector precision and calibration data.

The reviewer-workflow UI (item #9 / Phase A) writes a content-keyed verdict
each time an analyst marks a finding ``confirmed`` / ``false_positive`` /
``dismissed``. That table doubles as a free, in-flight eval signal — no
labeled benchmark required. This module joins reviewer verdicts to the
findings that produced them and computes:

- **per-detector precision** — ``confirmed / (confirmed + false_positive)``
  (``dismissed`` is excluded from the denominator: it means "not eval-relevant",
  not "the system was wrong").
- **calibration** — confidence-decile bins for a chosen detector, so the
  reviewer can tell whether ``judge_confidence`` actually tracks correctness.

The pair_key formula here MUST match
:func:`consistency_checker.audit.reviewer.build_pair_key`; the SQL CASE WHEN
expression is the inline equivalent for two-id pair findings, and multi-party
findings reconstruct the key in Python from ``assertion_ids_json``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.audit.reviewer import (
    DetectorType,
    ReviewerVerdictLabel,
    build_pair_key,
)

if TYPE_CHECKING:
    from consistency_checker.index.assertion_store import AssertionStore


def _parse_ts(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


#: Default confidence bin edges for the calibration plot. Chosen to spread
#: review work across realistic judge-confidence values rather than at uniform
#: deciles (most judge confidences cluster in [0.7, 1.0]).
DEFAULT_CALIBRATION_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 0.5),
    (0.5, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.0001),
)


@dataclass(frozen=True, slots=True)
class EvalRow:
    """One reviewed finding, joined to its judge output."""

    detector_type: str
    judge_verdict: str | None
    judge_confidence: float | None
    reviewer_verdict: ReviewerVerdictLabel
    set_at: datetime | None


@dataclass(frozen=True, slots=True)
class DetectorPrecision:
    """Precision summary for one detector_type."""

    detector_type: str
    n_confirmed: int
    n_false_positive: int
    n_dismissed: int
    n_reviewed: int
    precision: float | None


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """One bin in a calibration plot for a single detector."""

    confidence_low: float
    confidence_high: float
    n_reviewed: int
    n_confirmed: int
    n_false_positive: int
    n_dismissed: int
    precision: float | None


def iter_pair_eval_rows(store: AssertionStore) -> Iterator[EvalRow]:
    """Yield reviewed pair-shaped findings (contradiction + definition_inconsistency)."""
    sql = """
        SELECT COALESCE(f.detector_type, 'contradiction') AS detector_type,
               f.judge_verdict,
               f.judge_confidence,
               rv.verdict AS reviewer_verdict,
               rv.set_at AS set_at
          FROM findings f
          JOIN reviewer_verdicts rv
            ON rv.pair_key = (
                CASE WHEN f.assertion_a_id < f.assertion_b_id
                     THEN f.assertion_a_id || ':' || f.assertion_b_id
                     ELSE f.assertion_b_id || ':' || f.assertion_a_id
                END)
           AND rv.detector_type = COALESCE(f.detector_type, 'contradiction')
         ORDER BY rv.set_at, f.finding_id
    """
    for row in store._conn.execute(sql):
        yield EvalRow(
            detector_type=row["detector_type"],
            judge_verdict=row["judge_verdict"],
            judge_confidence=row["judge_confidence"],
            reviewer_verdict=row["reviewer_verdict"],
            set_at=_parse_ts(row["set_at"]),
        )


def iter_multi_party_eval_rows(store: AssertionStore) -> Iterator[EvalRow]:
    """Yield reviewed multi-party (triangle) findings.

    The pair_key for triangles is the N-ary sorted-and-joined assertion ids;
    we reconstruct it in Python rather than trying to JSON-unpack in SQLite.
    """
    logger = AuditLogger(store)
    sql = """
        SELECT judge_verdict, judge_confidence, assertion_ids_json
          FROM multi_party_findings
         ORDER BY created_at, finding_id
    """
    findings = list(store._conn.execute(sql))
    if not findings:
        return
    keys: list[tuple[str, DetectorType]] = []
    bookkeeping: list[tuple[str, str | None, float | None]] = []
    for row in findings:
        ids = json.loads(row["assertion_ids_json"])
        pair_key = build_pair_key(*ids)
        keys.append((pair_key, "multi_party"))
        bookkeeping.append((pair_key, row["judge_verdict"], row["judge_confidence"]))
    verdicts = logger.get_reviewer_verdicts_bulk(keys)
    for pair_key, judge_verdict, judge_confidence in bookkeeping:
        rv = verdicts.get((pair_key, "multi_party"))
        if rv is None:
            continue
        yield EvalRow(
            detector_type="multi_party",
            judge_verdict=judge_verdict,
            judge_confidence=judge_confidence,
            reviewer_verdict=rv.verdict,
            set_at=rv.set_at,
        )


def iter_eval_rows(store: AssertionStore) -> Iterator[EvalRow]:
    """Yield all reviewed findings across every detector_type."""
    yield from iter_pair_eval_rows(store)
    yield from iter_multi_party_eval_rows(store)


def compute_detector_precision(rows: Sequence[EvalRow]) -> list[DetectorPrecision]:
    """Aggregate eval rows into per-detector precision summaries.

    Precision is ``confirmed / (confirmed + false_positive)`` with ``dismissed``
    excluded from the denominator (dismissed means "not eval-relevant", not
    "the system was wrong"). When the denominator is zero, ``precision`` is
    ``None`` rather than zero — the distinction matters when only dismissed
    verdicts exist.
    """
    by_detector: dict[str, dict[str, int]] = {}
    for row in rows:
        counts = by_detector.setdefault(
            row.detector_type, {"confirmed": 0, "false_positive": 0, "dismissed": 0}
        )
        counts[row.reviewer_verdict] += 1
    out: list[DetectorPrecision] = []
    for detector, counts in sorted(by_detector.items()):
        confirmed = counts["confirmed"]
        false_positive = counts["false_positive"]
        dismissed = counts["dismissed"]
        denom = confirmed + false_positive
        precision = (confirmed / denom) if denom > 0 else None
        out.append(
            DetectorPrecision(
                detector_type=detector,
                n_confirmed=confirmed,
                n_false_positive=false_positive,
                n_dismissed=dismissed,
                n_reviewed=confirmed + false_positive + dismissed,
                precision=precision,
            )
        )
    return out


def compute_calibration(
    rows: Sequence[EvalRow],
    *,
    detector_type: str,
    bins: Sequence[tuple[float, float]] = DEFAULT_CALIBRATION_BINS,
) -> list[CalibrationBin]:
    """Bucket reviewed rows for one detector by ``judge_confidence``.

    Rows with a null ``judge_confidence`` (e.g. ``numeric_short_circuit`` or
    detector branches that don't surface a numeric confidence) are dropped
    silently — calibration is undefined for them.
    """
    scoped = [
        r for r in rows if r.detector_type == detector_type and r.judge_confidence is not None
    ]
    out: list[CalibrationBin] = []
    for low, high in bins:
        bucket = [
            r for r in scoped if r.judge_confidence is not None and low <= r.judge_confidence < high
        ]
        confirmed = sum(1 for r in bucket if r.reviewer_verdict == "confirmed")
        false_positive = sum(1 for r in bucket if r.reviewer_verdict == "false_positive")
        dismissed = sum(1 for r in bucket if r.reviewer_verdict == "dismissed")
        denom = confirmed + false_positive
        precision = (confirmed / denom) if denom > 0 else None
        out.append(
            CalibrationBin(
                confidence_low=low,
                confidence_high=high,
                n_reviewed=len(bucket),
                n_confirmed=confirmed,
                n_false_positive=false_positive,
                n_dismissed=dismissed,
                precision=precision,
            )
        )
    return out


def format_precision_table(precisions: Sequence[DetectorPrecision]) -> str:
    """Human-readable pretty-printed table of per-detector precision."""
    if not precisions:
        return "No reviewed findings yet — start clicking C/F/D in the web UI."
    header = (
        f"{'detector_type':<28} {'reviewed':>9} {'confirmed':>10} "
        f"{'false_pos':>10} {'dismissed':>10} {'precision':>10}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for p in precisions:
        prec = "n/a" if p.precision is None else f"{p.precision * 100:.1f}%"
        lines.append(
            f"{p.detector_type:<28} {p.n_reviewed:>9} {p.n_confirmed:>10} "
            f"{p.n_false_positive:>10} {p.n_dismissed:>10} {prec:>10}"
        )
    return "\n".join(lines)


def format_calibration_table(bins: Sequence[CalibrationBin], *, detector_type: str) -> str:
    """Human-readable pretty-printed table of confidence-decile calibration."""
    header_label = f"Calibration on '{detector_type}' (confidence vs. real precision)"
    header = (
        f"{'confidence range':<18} {'reviewed':>9} {'confirmed':>10} "
        f"{'false_pos':>10} {'precision':>10}"
    )
    sep = "-" * len(header)
    lines = [header_label, sep, header, sep]
    any_data = False
    for b in bins:
        any_data = any_data or b.n_reviewed > 0
        # Right-open intervals on display; cap the top edge visually at 1.0.
        top = min(b.confidence_high, 1.0)
        rng = f"[{b.confidence_low:.2f}, {top:.2f})"
        prec = "n/a" if b.precision is None else f"{b.precision * 100:.1f}%"
        lines.append(
            f"{rng:<18} {b.n_reviewed:>9} {b.n_confirmed:>10} {b.n_false_positive:>10} {prec:>10}"
        )
    if not any_data:
        lines.append(f"(no reviewed '{detector_type}' findings with a non-null confidence)")
    return "\n".join(lines)


def write_precision_csv(
    precisions: Sequence[DetectorPrecision], path: str | os.PathLike[str]
) -> None:
    """Write per-detector precision to CSV."""
    import csv

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "detector_type",
                "n_reviewed",
                "n_confirmed",
                "n_false_positive",
                "n_dismissed",
                "precision",
            ]
        )
        for p in precisions:
            writer.writerow(
                [
                    p.detector_type,
                    p.n_reviewed,
                    p.n_confirmed,
                    p.n_false_positive,
                    p.n_dismissed,
                    "" if p.precision is None else f"{p.precision:.6f}",
                ]
            )


def write_calibration_csv(
    bins_by_detector: Mapping[str, Sequence[CalibrationBin]],
    path: str | os.PathLike[str],
) -> None:
    """Write confidence-decile calibration to long-format CSV (one row per bin per detector)."""
    import csv

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "detector_type",
                "confidence_low",
                "confidence_high",
                "n_reviewed",
                "n_confirmed",
                "n_false_positive",
                "n_dismissed",
                "precision",
            ]
        )
        for detector, bins in sorted(bins_by_detector.items()):
            for b in bins:
                writer.writerow(
                    [
                        detector,
                        f"{b.confidence_low:.4f}",
                        f"{b.confidence_high:.4f}",
                        b.n_reviewed,
                        b.n_confirmed,
                        b.n_false_positive,
                        b.n_dismissed,
                        "" if b.precision is None else f"{b.precision:.6f}",
                    ]
                )


def eval_filename(kind: str, *, now: datetime | None = None) -> str:
    """Filename for an eval CSV — e.g. ``cc_eval_precision_2026-05-15T10-30-00.csv``.

    Mirrors :mod:`consistency_checker.audit.naming`'s ``cc_`` prefix convention.
    """
    from consistency_checker.audit.naming import _format_timestamp

    return f"cc_eval_{kind}_{_format_timestamp(now)}.csv"
