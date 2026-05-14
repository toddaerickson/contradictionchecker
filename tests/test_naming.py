"""Tests for output-filename helpers (ADR-0007 §Output filenames, step G0b)."""

from __future__ import annotations

import re
from datetime import datetime

import pytest

from consistency_checker.audit.naming import (
    export_csv_filename,
    export_jsonl_filename,
    report_filename,
)

_TIMESTAMP_RE = r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}"


# --- report_filename ------------------------------------------------------


def test_report_filename_uses_started_at_and_short_run_id() -> None:
    started = datetime(2026, 5, 14, 10, 30, 0)
    out = report_filename("a1b2c3d4e5f60718", started_at=started)
    assert out == "cc_report_2026-05-14T10-30-00_a1b2c3d4.md"


def test_report_filename_falls_back_to_now_when_started_at_none() -> None:
    out = report_filename("a1b2c3d4e5f60718")
    assert re.fullmatch(r"cc_report_" + _TIMESTAMP_RE + r"_a1b2c3d4\.md", out)


def test_report_filename_truncates_long_run_id() -> None:
    """uuid4().hex is 32 chars — we only keep the first 8."""
    started = datetime(2026, 1, 1, 0, 0, 0)
    long_run_id = "0123456789abcdef0123456789abcdef"
    assert (
        report_filename(long_run_id, started_at=started)
        == "cc_report_2026-01-01T00-00-00_01234567.md"
    )


def test_report_filename_lowercases_run_id() -> None:
    started = datetime(2026, 1, 1, 0, 0, 0)
    out = report_filename("ABCDEFGH1234567890", started_at=started)
    assert "abcdefgh" in out
    assert "ABCDEFGH" not in out


def test_report_filename_rejects_empty_run_id() -> None:
    with pytest.raises(ValueError, match="run_id"):
        report_filename("", started_at=datetime.now())


def test_report_filename_rejects_whitespace_only_run_id() -> None:
    with pytest.raises(ValueError, match="run_id"):
        report_filename("   ", started_at=datetime.now())


# --- export_csv_filename --------------------------------------------------


def test_export_csv_filename_uses_supplied_timestamp() -> None:
    when = datetime(2026, 5, 14, 10, 30, 0)
    assert export_csv_filename(now=when) == "cc_assertions_2026-05-14T10-30-00.csv"


def test_export_csv_filename_falls_back_to_now() -> None:
    out = export_csv_filename()
    assert re.fullmatch(r"cc_assertions_" + _TIMESTAMP_RE + r"\.csv", out)


# --- export_jsonl_filename ------------------------------------------------


def test_export_jsonl_filename_uses_supplied_timestamp() -> None:
    when = datetime(2026, 5, 14, 10, 30, 0)
    assert export_jsonl_filename(now=when) == "cc_assertions_2026-05-14T10-30-00.jsonl"


def test_export_jsonl_filename_falls_back_to_now() -> None:
    out = export_jsonl_filename()
    assert re.fullmatch(r"cc_assertions_" + _TIMESTAMP_RE + r"\.jsonl", out)


# --- determinism + uniqueness -------------------------------------------


def test_same_timestamp_and_run_id_produce_same_filename() -> None:
    """Pure functions: identical inputs → identical outputs."""
    started = datetime(2026, 5, 14, 10, 30, 0)
    assert report_filename("abc12345", started_at=started) == report_filename(
        "abc12345", started_at=started
    )


def test_distinct_run_ids_at_same_timestamp_produce_distinct_filenames() -> None:
    """Reports include run_id_short precisely because two runs can complete
    in the same wall-clock second; the run id breaks the tie."""
    started = datetime(2026, 5, 14, 10, 30, 0)
    first = report_filename("aaaaaaaa1234", started_at=started)
    second = report_filename("bbbbbbbb1234", started_at=started)
    assert first != second
