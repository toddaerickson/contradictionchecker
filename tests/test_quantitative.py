"""Table-driven tests for the quantitative extractor (E1)."""

from __future__ import annotations

import pytest

from consistency_checker.extract.quantitative import (
    QuantitativeTuple,
    extract_quantities,
    is_sign_flip,
)

# --- Single-claim extraction ------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Revenue grew 12% in fiscal 2025.",
            QuantitativeTuple(
                metric="revenue",
                value=12.0,
                unit="percent",
                polarity="up",
                scope="fiscal 2025",
            ),
        ),
        (
            "Revenue declined 5% in fiscal 2025.",
            QuantitativeTuple(
                metric="revenue",
                value=5.0,
                unit="percent",
                polarity="down",
                scope="fiscal 2025",
            ),
        ),
        (
            "Operating margin expanded 250 basis points during 2025.",
            QuantitativeTuple(
                metric="operating margin",
                value=250.0,
                unit="basis_points",
                polarity="up",
                scope="2025",
            ),
        ),
        (
            "Headcount fell 1,200 in Q3 2025.",
            QuantitativeTuple(
                metric="headcount",
                value=1200.0,
                unit=None,
                polarity="down",
                scope="q3 2025",
            ),
        ),
        (
            "Annual recurring revenue rose to $42 million in FY2025.",
            QuantitativeTuple(
                metric="annual recurring revenue",
                value=42.0,
                unit="million_usd",
                polarity="up",
                scope="fy2025",
            ),
        ),
        (
            "EBITDA dropped 8.5% year over year.",
            QuantitativeTuple(
                metric="ebitda",
                value=8.5,
                unit="percent",
                polarity="down",
                scope=None,
            ),
        ),
        (
            "Free cash flow contracted by €11 million.",
            QuantitativeTuple(
                metric="free cash flow",
                value=11.0,
                unit="million_eur",
                polarity="down",
                scope=None,
            ),
        ),
    ],
)
def test_extracts_directional_claims(text: str, expected: QuantitativeTuple) -> None:
    [out] = extract_quantities(text)
    assert out == expected


def test_explicit_positive_sign_yields_pos_polarity() -> None:
    [out] = extract_quantities("Earnings per share moved by +0.12 dollars.")
    assert out.polarity == "pos"
    assert out.value == 0.12


def test_explicit_negative_sign_yields_neg_polarity() -> None:
    [out] = extract_quantities("Net income shifted by -5 million dollars.")
    assert out.polarity == "neg"
    assert out.unit == "million"


def test_extracts_nothing_when_no_numbers_or_units() -> None:
    assert extract_quantities("Customer feedback was generally positive.") == []


def test_extracts_nothing_for_empty_input() -> None:
    assert extract_quantities("") == []


def test_value_holds_through_thousands_separators() -> None:
    [out] = extract_quantities("Revenue grew 1,234.5 million in fiscal 2024.")
    assert out.value == 1234.5
    assert out.unit == "million"


def test_metric_strips_articles() -> None:
    [out] = extract_quantities("The headcount fell 50 in 2025.")
    assert out.metric == "headcount"


def test_no_direction_verb_emits_when_unit_present() -> None:
    """Plain 'X is Y' phrasing still produces a tuple if a unit is recognised."""
    [out] = extract_quantities("Customer count is 42 thousand in 2025.")
    assert out.value == 42.0
    assert out.unit == "thousand"
    assert out.polarity == "none"


def test_bare_number_without_unit_is_dropped() -> None:
    """No unit, no direction verb, no sign → nothing to short-circuit on."""
    assert extract_quantities("There were 5 board members.") == []


# --- is_sign_flip -----------------------------------------------------------


def test_sign_flip_on_same_metric_scope_unit() -> None:
    grew = QuantitativeTuple("revenue", 12.0, "percent", "up", "fiscal 2025")
    declined = QuantitativeTuple("revenue", 5.0, "percent", "down", "fiscal 2025")
    assert is_sign_flip(grew, declined) is True
    assert is_sign_flip(declined, grew) is True


def test_no_short_circuit_when_metric_differs() -> None:
    a = QuantitativeTuple("revenue", 12.0, "percent", "up", "fiscal 2025")
    b = QuantitativeTuple("net income", 5.0, "percent", "down", "fiscal 2025")
    assert is_sign_flip(a, b) is False


def test_no_short_circuit_when_scope_differs() -> None:
    a = QuantitativeTuple("revenue", 12.0, "percent", "up", "fiscal 2025")
    b = QuantitativeTuple("revenue", 5.0, "percent", "down", "fiscal 2024")
    assert is_sign_flip(a, b) is False


def test_no_short_circuit_when_unit_differs() -> None:
    a = QuantitativeTuple("revenue", 12.0, "percent", "up", "fiscal 2025")
    b = QuantitativeTuple("revenue", 5.0, "basis_points", "down", "fiscal 2025")
    assert is_sign_flip(a, b) is False


def test_no_short_circuit_when_polarity_is_none() -> None:
    a = QuantitativeTuple("revenue", 12.0, "percent", "up", "fiscal 2025")
    b = QuantitativeTuple("revenue", 12.0, "percent", "none", "fiscal 2025")
    assert is_sign_flip(a, b) is False


def test_sign_flip_with_explicit_pos_neg_polarities() -> None:
    pos = QuantitativeTuple("eps", 0.12, "usd", "pos", None)
    neg = QuantitativeTuple("eps", 0.05, "usd", "neg", None)
    assert is_sign_flip(pos, neg) is True


def test_same_polarity_is_not_a_flip() -> None:
    a = QuantitativeTuple("revenue", 12.0, "percent", "up", "fiscal 2025")
    b = QuantitativeTuple("revenue", 8.0, "percent", "up", "fiscal 2025")
    assert is_sign_flip(a, b) is False


# --- Integration on hand-built corpus pairs ---------------------------------


def test_canonical_revenue_flip_matches_on_real_assertion_text() -> None:
    """The Alpha-revenue pair from the v0.1 e2e corpus must short-circuit."""
    a_text = "Revenue from Alpha grew 12% year-over-year in fiscal 2025."
    b_text = "Revenue from Alpha declined 5% in fiscal 2025."
    [a] = extract_quantities(a_text)
    [b] = extract_quantities(b_text)
    assert is_sign_flip(a, b)


def test_unit_mismatch_does_not_short_circuit() -> None:
    """Percent vs. absolute-dollar movement on the same metric is ambiguous."""
    a_text = "Revenue grew 12% in fiscal 2025."
    b_text = "Revenue fell 5 million dollars in fiscal 2025."
    [a] = extract_quantities(a_text)
    [b] = extract_quantities(b_text)
    assert is_sign_flip(a, b) is False
