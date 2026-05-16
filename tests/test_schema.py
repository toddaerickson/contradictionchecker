"""Tests for the Assertion / Document dataclasses."""

from __future__ import annotations

from consistency_checker.extract.schema import Assertion


def test_assertion_defaults_to_claim_kind() -> None:
    a = Assertion.build("doc1", "The Borrower is ABC Corp.")
    assert a.kind == "claim"
    assert a.term is None
    assert a.definition_text is None


def test_assertion_can_be_a_definition() -> None:
    a = Assertion.build(
        "doc1",
        '"Borrower" means ABC Corp and its Subsidiaries.',
        kind="definition",
        term="Borrower",
        definition_text="ABC Corp and its Subsidiaries",
    )
    assert a.kind == "definition"
    assert a.term == "Borrower"
    assert a.definition_text == "ABC Corp and its Subsidiaries"
