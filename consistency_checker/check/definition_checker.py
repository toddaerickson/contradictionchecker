"""Definition-inconsistency detector.

Groups definition assertions by canonical term, enumerates unordered pairs
within each group, and asks the definition judge whether each pair is
consistent or divergent. Findings flow into the existing ``findings`` table
via the audit logger under ``detector_type='definition_inconsistency'``.

Unlike the contradiction pipeline, this stage skips the NLI gate. The DeBERTa
gate is contradiction-tuned and unhelpful when comparing two definitions of
the same concept — the term-group gate above is both cheaper and more precise
for this question.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations

from consistency_checker.check.definition_judge import (
    DefinitionJudge,
    DefinitionJudgeVerdict,
)
from consistency_checker.check.definition_terms import canonicalize_term
from consistency_checker.extract.schema import Assertion


@dataclass(frozen=True, slots=True)
class DefinitionPair:
    """A pair of definition assertions whose canonical terms match."""

    a: Assertion
    b: Assertion
    canonical_term: str


@dataclass(frozen=True, slots=True)
class DefinitionFinding:
    """One definition-pair verdict — what the checker emits, before audit."""

    pair: DefinitionPair
    verdict: DefinitionJudgeVerdict


def _group_by_canonical_term(
    definitions: Iterable[Assertion],
) -> dict[str, list[Assertion]]:
    groups: dict[str, list[Assertion]] = {}
    for d in definitions:
        if d.kind != "definition" or d.term is None:
            continue
        canonical = canonicalize_term(d.term)
        if not canonical:
            continue
        groups.setdefault(canonical, []).append(d)
    return groups


def _enumerate_pairs(
    groups: dict[str, list[Assertion]],
) -> Iterator[DefinitionPair]:
    for canonical, assertions in groups.items():
        if len(assertions) < 2:
            continue
        ordered = sorted(assertions, key=lambda a: a.assertion_id)
        for a, b in combinations(ordered, 2):
            yield DefinitionPair(a=a, b=b, canonical_term=canonical)


class DefinitionChecker:
    """Orchestrates term-grouping → pair-enumeration → judge for definitions."""

    def __init__(self, *, judge: DefinitionJudge) -> None:
        self._judge = judge

    def find_inconsistencies(
        self, definitions: Sequence[Assertion]
    ) -> Iterator[DefinitionFinding]:
        groups = _group_by_canonical_term(definitions)
        for pair in _enumerate_pairs(groups):
            verdict = self._judge.judge(pair.a, pair.b)
            yield DefinitionFinding(pair=pair, verdict=verdict)
