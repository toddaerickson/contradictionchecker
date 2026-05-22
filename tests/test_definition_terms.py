"""Tests for the definition-term canonicalisation helper."""

from __future__ import annotations

import pytest

from consistency_checker.check.definition_terms import canonicalize_term, definitions_equivalent


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Borrower", "borrower"),
        ("borrower", "borrower"),
        ("the Borrower", "borrower"),
        ("The Borrower", "borrower"),
        ("Borrowers", "borrower"),
        ('"Borrower"', "borrower"),
        ("“Borrower”", "borrower"),
        ("  Borrower  ", "borrower"),
        ("Material Adverse Effect", "material adverse effect"),
        ("MAE", "mae"),
    ],
)
def test_canonicalize_term(raw: str, expected: str) -> None:
    assert canonicalize_term(raw) == expected


def test_canonicalize_empty() -> None:
    assert canonicalize_term("") == ""
    assert canonicalize_term("   ") == ""


@pytest.mark.parametrize(
    "a,b,expected",
    [
        # identical
        (
            "the board of directors of the Corporation",
            "the board of directors of the Corporation",
            True,
        ),
        # whitespace-only difference
        ("a majority   of the\tdirectors", "a majority of the directors", True),
        # case-only difference
        ("The Board of Directors", "the board of directors", True),
        # surrounding punctuation / quotes only
        ('"the board of directors."', "the board of directors", True),
        ("(the board of directors)", "the board of directors", True),
        # genuine wording difference
        ("a majority of the directors", "two-thirds of the directors", False),
        # mid-string comma that changes scope must NOT be equivalent
        ("directors, officers and employees", "directors officers and employees", False),
        # all-punctuation normalizes to empty — pinned behavior
        ("...", "", True),
        ("", "", True),
        # non-breaking space collapses like normal whitespace
        ("the\N{NO-BREAK SPACE}board", "the board", True),
    ],
)
def test_definitions_equivalent(a: str, b: str, expected: bool) -> None:
    assert definitions_equivalent(a, b) is expected


def test_definitions_equivalent_is_symmetric() -> None:
    a, b = "The Board.", "the board"
    assert definitions_equivalent(a, b) == definitions_equivalent(b, a)
