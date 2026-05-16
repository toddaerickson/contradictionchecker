"""Tests for the DefinitionChecker — grouping, pairing, judging."""

from __future__ import annotations

from consistency_checker.check.definition_checker import (
    DefinitionChecker,
)
from consistency_checker.check.definition_judge import (
    DefinitionJudgeVerdict,
    FixtureDefinitionJudge,
)
from consistency_checker.extract.schema import Assertion


def _def(doc: str, term: str, text: str) -> Assertion:
    return Assertion.build(
        doc,
        f'"{term}" means {text}.',
        kind="definition",
        term=term,
        definition_text=text,
    )


def test_singleton_term_emits_zero_pairs() -> None:
    defs = [_def("docA", "Borrower", "ABC Corp")]
    checker = DefinitionChecker(judge=FixtureDefinitionJudge({}))
    findings = list(checker.find_inconsistencies(defs))
    assert findings == []


def test_pair_within_term_group_is_judged() -> None:
    a = _def("docA", "Borrower", "ABC Corp")
    b = _def("docB", "Borrower", "ABC Corp and its Subsidiaries")
    verdict = DefinitionJudgeVerdict(
        assertion_a_id=min(a.assertion_id, b.assertion_id),
        assertion_b_id=max(a.assertion_id, b.assertion_id),
        verdict="definition_divergent",
        confidence=0.9,
        rationale="scope shift",
        evidence_spans=[],
    )
    judge = FixtureDefinitionJudge(
        {(min(a.assertion_id, b.assertion_id), max(a.assertion_id, b.assertion_id)): verdict}
    )
    checker = DefinitionChecker(judge=judge)
    findings = list(checker.find_inconsistencies([a, b]))
    assert len(findings) == 1
    assert findings[0].verdict.verdict == "definition_divergent"
    assert findings[0].pair.canonical_term == "borrower"


def test_different_canonical_terms_do_not_pair() -> None:
    a = _def("docA", "Borrower", "ABC Corp")
    b = _def("docB", "Lender", "First Bank")
    checker = DefinitionChecker(judge=FixtureDefinitionJudge({}))
    findings = list(checker.find_inconsistencies([a, b]))
    assert findings == []


def test_plurals_and_articles_group_together() -> None:
    a = _def("docA", "Borrower", "ABC Corp")
    b = _def("docB", "the Borrowers", "ABC Corp and its Subsidiaries")
    judge = FixtureDefinitionJudge({})  # uncertain fallback
    checker = DefinitionChecker(judge=judge)
    findings = list(checker.find_inconsistencies([a, b]))
    assert len(findings) == 1
    assert findings[0].verdict.verdict == "uncertain"
    assert findings[0].pair.canonical_term == "borrower"


def test_non_definition_assertions_ignored() -> None:
    a = _def("docA", "Borrower", "ABC Corp")
    b = _def("docB", "Borrower", "ABC Corp and its Subsidiaries")
    noise = Assertion.build("docC", "Revenue grew 12%.")  # kind="claim"
    checker = DefinitionChecker(judge=FixtureDefinitionJudge({}))
    findings = list(checker.find_inconsistencies([a, b, noise]))
    # uncertain because no fixture matches, but the noise assertion must not
    # have caused any extra pair.
    assert len(findings) == 1


def test_three_definitions_emit_three_pairs() -> None:
    a = _def("docA", "Borrower", "ABC Corp")
    b = _def("docB", "Borrower", "ABC Corp and Subsidiaries")
    c = _def("docC", "Borrower", "ABC Corp, its Subsidiaries, and Affiliates")
    checker = DefinitionChecker(judge=FixtureDefinitionJudge({}))
    findings = list(checker.find_inconsistencies([a, b, c]))
    assert len(findings) == 3  # combinations(3, 2) == 3
