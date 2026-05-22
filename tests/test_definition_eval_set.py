"""Hermetic checks on the definition eval set: schema + identical-row determinism."""

from __future__ import annotations

import json
from pathlib import Path

from consistency_checker.check.definition_checker import DefinitionChecker
from consistency_checker.check.providers.definition_base import DEFINITION_CONSISTENT_AUTO
from consistency_checker.extract.schema import Assertion

PAIRS = Path("benchmarks/definition_eval/pairs.jsonl")


def _rows() -> list[dict]:
    return [json.loads(ln) for ln in PAIRS.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_pairs_schema() -> None:
    rows = _rows()
    assert rows, "pairs.jsonl is empty"
    ids = [r["pair_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate pair_id"
    for r in rows:
        assert r["label"] in {"consistent", "divergent"}
        assert {"pair_id", "category", "term", "def_a", "def_b", "label"} <= r.keys()


class _RaisingJudge:
    def judge(self, a, b):  # type: ignore[no-untyped-def]
        raise AssertionError("identical rows must short-circuit, never reach the judge")


def test_identical_rows_short_circuit() -> None:
    checker = DefinitionChecker(judge=_RaisingJudge())
    for r in _rows():
        if r["category"] != "identical":
            continue
        a = Assertion.build(
            "doc_a",
            f'"{r["term"]}" means {r["def_a"]}.',
            kind="definition",
            term=r["term"],
            definition_text=r["def_a"],
        )
        b = Assertion.build(
            "doc_b",
            f'"{r["term"]}" means {r["def_b"]}.',
            kind="definition",
            term=r["term"],
            definition_text=r["def_b"],
        )
        findings = list(checker.find_inconsistencies([a, b]))
        assert len(findings) == 1
        assert findings[0].verdict.verdict == DEFINITION_CONSISTENT_AUTO, r["pair_id"]
