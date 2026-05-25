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

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from itertools import combinations

from consistency_checker.check.definition_judge import (
    DefinitionJudge,
    DefinitionJudgeVerdict,
    definition_short_circuit_verdict,
)
from consistency_checker.check.definition_terms import (
    canonicalize_term,
    definitions_equivalent,
)
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


@dataclass(frozen=True, slots=True)
class SuppressedDefinitionPair:
    """A cross-org definition pair that was skipped before reaching the judge."""

    a: Assertion
    b: Assertion
    canonical_term: str
    org_key_a: str
    org_key_b: str


@dataclass(frozen=True, slots=True)
class DefinitionCheckResult:
    findings: list[DefinitionFinding] = field(default_factory=list)
    suppressed_pairs: list[SuppressedDefinitionPair] = field(default_factory=list)

    @property
    def n_judged(self) -> int:
        return len(self.findings)

    @property
    def n_suppressed(self) -> int:
        return len(self.suppressed_pairs)


def _group_by_canonical_term(
    definitions: Sequence[tuple[Assertion, str]],
) -> dict[str, list[tuple[Assertion, str]]]:
    """Backwards-compat helper used by ``estimate_cost`` for pair-count math."""
    groups: dict[str, list[tuple[Assertion, str]]] = {}
    for entry in definitions:
        d, _org_key = entry
        if d.kind != "definition" or d.term is None:
            continue
        canonical = canonicalize_term(d.term)
        if not canonical:
            continue
        groups.setdefault(canonical, []).append(entry)
    return groups


class DefinitionChecker:
    """Orchestrates term-grouping → pair-enumeration → judge for definitions."""

    def __init__(self, *, judge: DefinitionJudge, org_scope_enabled: bool = False) -> None:
        self._judge = judge
        self._org_scope_enabled = org_scope_enabled

    def find_inconsistencies(
        self, definitions: Sequence[tuple[Assertion, str]]
    ) -> Iterator[DefinitionFinding]:
        """Streaming API: yields only judged findings. Suppressed pairs are dropped."""
        yield from self.run(definitions).findings

    def run(self, definitions: Sequence[tuple[Assertion, str]]) -> DefinitionCheckResult:
        groups = self._group(definitions)
        findings: list[DefinitionFinding] = []
        suppressed: list[SuppressedDefinitionPair] = []
        for canonical, entries in groups.items():
            ordered = sorted(entries, key=lambda e: e[0].assertion_id)
            for (a, ka), (b, kb) in combinations(ordered, 2):
                if self._org_scope_enabled and ka != "" and kb != "" and ka != kb:
                    suppressed.append(
                        SuppressedDefinitionPair(
                            a=a, b=b, canonical_term=canonical, org_key_a=ka, org_key_b=kb
                        )
                    )
                    continue
                pair = DefinitionPair(a=a, b=b, canonical_term=canonical)
                if definitions_equivalent(pair.a.assertion_text, pair.b.assertion_text):
                    verdict = definition_short_circuit_verdict(pair.a, pair.b)
                else:
                    verdict = self._judge.judge(pair.a, pair.b)
                findings.append(DefinitionFinding(pair=pair, verdict=verdict))
        return DefinitionCheckResult(findings=findings, suppressed_pairs=suppressed)

    def count_pairs(self, definitions: Sequence[tuple[Assertion, str]]) -> int:
        """Return the number of pairs ``run()`` would actually judge.

        Cross-org pairs suppressed by :attr:`_org_scope_enabled` are excluded.
        Used by ``estimate_cost`` so the preview matches the real run.
        """
        total = 0
        for entries in self._group(definitions).values():
            ordered = sorted(entries, key=lambda e: e[0].assertion_id)
            for (_a, ka), (_b, kb) in combinations(ordered, 2):
                if self._org_scope_enabled and ka != "" and kb != "" and ka != kb:
                    continue
                total += 1
        return total

    def _group(
        self, definitions: Sequence[tuple[Assertion, str]]
    ) -> dict[str, list[tuple[Assertion, str]]]:
        """Group definitions by canonical term.

        Org scoping is applied later in :meth:`run` so cross-org pairs can
        still be enumerated and recorded as suppressed (audit trail) rather
        than silently dropped at the grouping step.
        """
        out: dict[str, list[tuple[Assertion, str]]] = {}
        for a, org_key in definitions:
            if a.kind != "definition" or a.term is None:
                continue
            canonical = canonicalize_term(a.term)
            if not canonical:
                continue
            out.setdefault(canonical, []).append((a, org_key))
        return out
