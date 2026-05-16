"""Tests for the DefinitionJudgePayload schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from consistency_checker.check.providers.definition_base import (
    DefinitionJudgePayload,
)


def test_payload_accepts_three_verdicts() -> None:
    for v in ("definition_consistent", "definition_divergent", "uncertain"):
        p = DefinitionJudgePayload(
            verdict=v,  # type: ignore[arg-type]
            confidence=0.8,
            rationale="x",
            evidence_spans=[],
        )
        assert p.verdict == v


def test_payload_rejects_unknown_verdict() -> None:
    with pytest.raises(ValidationError):
        DefinitionJudgePayload(
            verdict="contradiction",  # type: ignore[arg-type]
            confidence=0.5,
            rationale="x",
            evidence_spans=[],
        )


def test_payload_rejects_extras() -> None:
    with pytest.raises(ValidationError):
        DefinitionJudgePayload.model_validate(
            {
                "verdict": "definition_consistent",
                "confidence": 0.5,
                "rationale": "x",
                "evidence_spans": [],
                "extra_field": "nope",
            }
        )


def test_payload_requires_nonempty_rationale() -> None:
    with pytest.raises(ValidationError):
        DefinitionJudgePayload(
            verdict="uncertain",
            confidence=0.0,
            rationale="",
            evidence_spans=[],
        )


def test_payload_confidence_bounds() -> None:
    DefinitionJudgePayload(verdict="uncertain", confidence=0.0, rationale="x")
    DefinitionJudgePayload(verdict="uncertain", confidence=1.0, rationale="x")
    with pytest.raises(ValidationError):
        DefinitionJudgePayload(verdict="uncertain", confidence=1.5, rationale="x")
    with pytest.raises(ValidationError):
        DefinitionJudgePayload(verdict="uncertain", confidence=-0.1, rationale="x")
