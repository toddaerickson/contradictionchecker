"""Tests for the definition-term canonicalisation helper."""

from __future__ import annotations

import pytest

from consistency_checker.check.definition_terms import (
    canonicalize_term,
    definitions_equivalent,
    is_definitional,
)


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


# --- is_definitional: distinguish a real definitional clause from a reference ---


def test_is_definitional_accepts_term_means_clause() -> None:
    assert is_definitional(
        "Affiliated Lender",
        '"Affiliated Lender" means, at any time, any Lender that is any Person.',
    )


def test_is_definitional_accepts_shall_mean_and_smart_quotes() -> None:
    assert is_definitional("Quorum", "“Quorum” shall mean a majority of the directors.")


def test_is_definitional_rejects_usage_reference() -> None:
    # The extractor mis-tags usages of a capitalized defined term as definitions.
    assert not is_definitional(
        "Affiliated Lender",
        "Notwithstanding anything to the contrary, any Lender may assign its "
        "rights to an Affiliated Lender subject to the following limitations.",
    )


def test_is_definitional_rejects_cross_reference() -> None:
    # "has the meaning set forth in §X" points elsewhere — not a definition.
    assert not is_definitional(
        "Permitted Amendment",
        '"Permitted Amendment" has the meaning set forth in Section 2.11.',
    )


def test_is_definitional_rejects_none_or_missing_term() -> None:
    assert not is_definitional(None, '"X" means y.')
    assert not is_definitional("X", None)
    assert not is_definitional("Quorum", "The board met on Tuesday.")
