"""Tests for the deterministic junk filter (pure predicates + audit sink)."""

from __future__ import annotations

import json
from pathlib import Path

from consistency_checker.corpus.junk_filter import (
    JunkAudit,
    is_junk_assertion,
    is_junk_line,
)


# --- is_junk_line: junk cases (each traces to a real observed example) ------
def test_is_junk_line_dot_leader() -> None:
    assert is_junk_line("Expenses ........ 15") == "dot_leader"
    assert is_junk_line("...............") == "dot_leader"


def test_is_junk_line_page_number() -> None:
    assert is_junk_line("15") == "page_number"
    assert is_junk_line("Page 3") == "page_number"
    assert is_junk_line("- 12 -") == "page_number"


def test_is_junk_line_mostly_non_alpha() -> None:
    assert is_junk_line("_________________") == "mostly_non_alpha"


# --- is_junk_line: clean cases that must NOT be dropped ---------------------
def test_is_junk_line_keeps_real_clause() -> None:
    assert is_junk_line("The Board shall consist of no fewer than three Directors.") is None


def test_is_junk_line_keeps_short_heading() -> None:
    assert is_junk_line("ARTICLE V") is None  # has alpha, no dots, not a bare number


def test_is_junk_line_keeps_sentence_with_ellipsis() -> None:
    # three-dot ellipsis must not trip the >=5-dot rule
    assert is_junk_line("The committee deliberated... and then voted.") is None


# --- is_junk_assertion: junk cases ------------------------------------------
def test_is_junk_assertion_cross_reference() -> None:
    assert is_junk_assertion("as defined in this Article 11") == "cross_reference"
    assert is_junk_assertion("See Section 4.2 below.") == "cross_reference"


def test_is_junk_assertion_near_empty() -> None:
    assert is_junk_assertion("1.") == "near_empty"
    assert is_junk_assertion("   ") == "near_empty"


def test_is_junk_assertion_dot_fragment() -> None:
    assert is_junk_assertion(".......... 15") == "dot_fragment"


def test_is_junk_assertion_mostly_non_alpha() -> None:
    assert is_junk_assertion("________________") == "mostly_non_alpha"


# --- is_junk_assertion: clean cases that must NOT be dropped ----------------
def test_is_junk_assertion_keeps_real_definition() -> None:
    text = "Quorum means a majority of the Directors then in office."
    assert is_junk_assertion(text) is None


def test_is_junk_assertion_keeps_substantive_clause_referencing_section() -> None:
    # starts with a pointer phrase AND names a section, but carries real substance
    text = (
        "As set forth in Section 4, the Quorum for any meeting of the Board shall be "
        "a majority of the Directors then in office, present in person or by proxy."
    )
    assert is_junk_assertion(text) is None


# --- JunkAudit --------------------------------------------------------------
def test_junk_audit_counts_and_writes(tmp_path: Path) -> None:
    audit = JunkAudit(tmp_path / "junk_drops.jsonl")
    audit.record(stage="text", reason="dot_leader", doc_id="doc1", text="x" * 500)
    audit.record(stage="assertion", reason="cross_reference", doc_id="doc1", text="see X")
    assert audit.counts == {"dot_leader": 1, "cross_reference": 1}
    lines = (tmp_path / "junk_drops.jsonl").read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["stage"] == "text" and rec["reason"] == "dot_leader" and rec["doc_id"] == "doc1"
    assert len(rec["text_snippet"]) <= 200  # snippet is truncated


def test_junk_audit_none_path_is_memory_only(tmp_path: Path) -> None:
    audit = JunkAudit(None)
    audit.record(stage="text", reason="page_number", doc_id=None, text="15")
    assert audit.counts == {"page_number": 1}  # no file, counts still tracked


def test_junk_audit_write_failure_is_swallowed(tmp_path: Path) -> None:
    # point at a path whose parent is a file, so mkdir/open fails; must not raise
    bad = tmp_path / "afile"
    bad.write_text("x")
    audit = JunkAudit(bad / "nested" / "drops.jsonl")
    audit.record(stage="text", reason="dot_leader", doc_id=None, text="...")  # no exception
    assert audit.counts == {"dot_leader": 1}


def test_is_junk_assertion_cross_reference_still_fires_on_bare_pointers() -> None:
    # regression guard: the observed junk must still be caught after tightening
    assert is_junk_assertion("as defined in this Article 11") == "cross_reference"
    assert is_junk_assertion("as defined in the bylaws") == "cross_reference"
    assert is_junk_assertion("See Section 4.2 below.") == "cross_reference"


def test_is_junk_assertion_keeps_short_real_clauses_with_pointer_phrase() -> None:
    # real governance clauses that START with a pointer phrase but carry substance
    # must NOT be dropped (they exceed the 30-alpha guard)
    keepers = [
        "As used herein, Director means a board member.",
        "As defined above, Quorum is a majority of Directors.",
        "As provided in the Bylaws, the President shall preside.",
        "As set forth in Section 3, the Treasurer keeps all funds.",
    ]
    for text in keepers:
        assert is_junk_assertion(text) is None, f"wrongly dropped: {text!r}"


def test_is_junk_assertion_near_empty_boundary_keeps_real_short_clause() -> None:
    # >=10 alphabetic chars, real content → not near_empty
    assert is_junk_assertion("Members vote.") is None
