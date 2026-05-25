"""Tests for the Assertion / Document dataclasses."""

from __future__ import annotations

from consistency_checker.extract.schema import Assertion, Document


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


def test_document_dataclass_carries_org_fields() -> None:
    doc = Document(
        doc_id="abc",
        source_path="/x.txt",
        org_label="Acme Foundation, Inc.",
        org_reason="org_found",
    )
    assert doc.org_label == "Acme Foundation, Inc."
    assert doc.org_reason == "org_found"
    default_doc = Document(doc_id="abc", source_path="/x.txt")
    assert default_doc.org_label is None
    assert default_doc.org_reason is None
