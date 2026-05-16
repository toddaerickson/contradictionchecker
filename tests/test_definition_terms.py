"""Tests for the definition-term canonicalisation helper."""

from __future__ import annotations

import pytest

from consistency_checker.check.definition_terms import canonicalize_term


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
