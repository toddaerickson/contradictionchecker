"""Tests for the OCR detection predicate and audit sink."""

from __future__ import annotations

import json
from pathlib import Path

from consistency_checker.corpus.ocr import OcrAudit, looks_empty, needs_ocr


# --- looks_empty: cheap text-only sub-check ---------------------------------
def test_looks_empty_true_when_text_has_few_alpha_chars() -> None:
    assert looks_empty("") is True
    assert looks_empty("Page 1\n\nPage 2") is True


def test_looks_empty_false_when_text_has_real_content() -> None:
    long_text = "The Board shall consist of no fewer than three Directors. " * 20
    assert looks_empty(long_text) is False


# --- needs_ocr: positive cases (image PDFs the heuristic must catch) --------
def test_needs_ocr_empty_text_multi_page_large_file() -> None:
    assert needs_ocr(text="", page_count=10, file_size=500_000) is True


def test_needs_ocr_short_text_multi_page_large_file() -> None:
    assert needs_ocr(text="Page 1\n\nPage 2", page_count=10, file_size=500_000) is True


# --- needs_ocr: negative cases (must NOT fire) ------------------------------
def test_needs_ocr_short_text_single_page_skipped() -> None:
    # single-page cover sheets are legitimately short
    assert needs_ocr(text="", page_count=1, file_size=500_000) is False


def test_needs_ocr_short_text_tiny_file_skipped() -> None:
    # placeholder PDFs under 100 KB don't deserve a slow re-pass
    assert needs_ocr(text="", page_count=10, file_size=10_000) is False


def test_needs_ocr_substantive_text_skipped() -> None:
    long_text = "The Board shall consist of no fewer than three Directors. " * 20
    assert needs_ocr(text=long_text, page_count=10, file_size=500_000) is False


# --- OcrAudit ---------------------------------------------------------------
def test_ocr_audit_records_escalation_and_failure(tmp_path: Path) -> None:
    audit = OcrAudit(tmp_path / "ocr_events.jsonl")
    audit.record(event="escalated", path="a.pdf", page_count=10)
    audit.record(event="ocr_failed", path="b.pdf", page_count=5)
    assert audit.counts == {"escalated": 1, "ocr_failed": 1}
    lines = (tmp_path / "ocr_events.jsonl").read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["event"] == "escalated" and rec["path"] == "a.pdf"
    assert "doc_id" not in rec


def test_ocr_audit_none_path_is_memory_only() -> None:
    audit = OcrAudit(None)
    audit.record(event="escalated", path="a.pdf", page_count=3)
    assert audit.counts == {"escalated": 1}


def test_ocr_audit_write_failure_is_swallowed(tmp_path: Path) -> None:
    # parent is a regular file → mkdir(parents=True) raises NotADirectoryError
    bad = tmp_path / "afile"
    bad.write_text("x")
    audit = OcrAudit(bad / "nested" / "events.jsonl")
    audit.record(event="escalated", path="a.pdf", page_count=3)  # no exception
    assert audit.counts == {"escalated": 1}
