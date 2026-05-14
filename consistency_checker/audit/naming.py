"""Output naming helpers for reports and exports — ADR-0007 §"Output filenames".

Both the CLI and the v0.3 web UI write reports and exports through the same
filename convention so two runs against the same corpus never overwrite each
other and so a directory full of artefacts is browsable at a glance.

Format (all filenames are prefixed ``cc_`` to namespace them under the
``consistency-checker`` package):

- Report — ``cc_report_{started_at:%Y-%m-%dT%H-%M-%S}_{run_id_short}.md``
  where ``run_id_short`` is the first 8 chars of the audit-DB ``run_id``.
- CSV export — ``cc_assertions_{timestamp:%Y-%m-%dT%H-%M-%S}.csv``.
- JSONL export — ``cc_assertions_{timestamp:%Y-%m-%dT%H-%M-%S}.jsonl``.

Reports include the run id because two runs can complete in the same wall-clock
second; exports are corpus-wide snapshots so the timestamp alone is enough.

All helpers are pure: no IO, no globals. Callers join the result onto
``data_dir / "reports"`` (or wherever they choose) and create directories
themselves.
"""

from __future__ import annotations

from datetime import datetime

_TIMESTAMP_FMT = "%Y-%m-%dT%H-%M-%S"
_RUN_ID_SHORT_LEN = 8


def _format_timestamp(when: datetime | None) -> str:
    return (when or datetime.now()).strftime(_TIMESTAMP_FMT)


def _short_run_id(run_id: str) -> str:
    """First :data:`_RUN_ID_SHORT_LEN` chars of the run id, lowercased and stripped.

    Accepts both ``uuid4().hex`` (32 chars) and the 16-char :func:`hash_id`
    form. Raises on empty input — callers always have a real run id.
    """
    cleaned = run_id.strip().lower()
    if not cleaned:
        raise ValueError("run_id must be non-empty")
    return cleaned[:_RUN_ID_SHORT_LEN]


def report_filename(run_id: str, *, started_at: datetime | None = None) -> str:
    """Return the unique filename for a markdown report.

    Example: ``cc_report_2026-05-14T10-30-00_a1b2c3d4.md``.

    ``started_at`` is the audit run's ``started_at`` timestamp; ``None`` falls
    back to ``datetime.now()`` so callers without an audit-DB handle can still
    produce a sensible name.
    """
    return f"cc_report_{_format_timestamp(started_at)}_{_short_run_id(run_id)}.md"


def export_csv_filename(*, now: datetime | None = None) -> str:
    """Return the filename for a CSV assertions export.

    Example: ``cc_assertions_2026-05-14T10-30-00.csv``. No run id — exports are
    snapshots of the whole assertions table, not per-run artefacts.
    """
    return f"cc_assertions_{_format_timestamp(now)}.csv"


def export_jsonl_filename(*, now: datetime | None = None) -> str:
    """Return the filename for a JSONL assertions export.

    Example: ``cc_assertions_2026-05-14T10-30-00.jsonl``.
    """
    return f"cc_assertions_{_format_timestamp(now)}.jsonl"
