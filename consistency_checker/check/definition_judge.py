"""Definition-inconsistency judge.

Mirrors :mod:`consistency_checker.check.llm_judge` but for the definition
detector — same retry-with-degraded-fallback pattern, different prompts,
different verdict vocabulary, different payload type.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from consistency_checker.check.providers.definition_base import (
    DefinitionJudgePayload,
    DefinitionJudgeProvider,
    DefinitionVerdictLabel,
)
from consistency_checker.extract.schema import Assertion
from consistency_checker.logging_setup import get_logger

_log = get_logger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "definition_judge_system.txt"
USER_PROMPT_PATH = PROMPTS_DIR / "definition_judge_user.txt"


def render_definition_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def render_definition_user_prompt(a: Assertion, b: Assertion) -> str:
    """Render the user prompt with both definitions and the shared term."""
    if a.term is None or b.term is None:
        raise ValueError("definition judge requires both assertions to have a `term`")
    template = USER_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template.replace("{term}", a.term)
        .replace("{doc_a_id}", a.doc_id)
        .replace("{doc_b_id}", b.doc_id)
        .replace("{assertion_a_text}", a.assertion_text)
        .replace("{assertion_b_text}", b.assertion_text)
    )


@dataclass(frozen=True, slots=True)
class DefinitionJudgeVerdict:
    """Verdict for one definition pair plus its provenance."""

    assertion_a_id: str
    assertion_b_id: str
    verdict: DefinitionVerdictLabel
    confidence: float
    rationale: str
    evidence_spans: list[str] = field(default_factory=list)

    @classmethod
    def from_payload(
        cls, a: Assertion, b: Assertion, payload: DefinitionJudgePayload
    ) -> DefinitionJudgeVerdict:
        return cls(
            assertion_a_id=a.assertion_id,
            assertion_b_id=b.assertion_id,
            verdict=payload.verdict,
            confidence=payload.confidence,
            rationale=payload.rationale,
            evidence_spans=list(payload.evidence_spans),
        )


def definition_uncertain_fallback(
    a: Assertion, b: Assertion, reason: str
) -> DefinitionJudgeVerdict:
    return DefinitionJudgeVerdict(
        assertion_a_id=a.assertion_id,
        assertion_b_id=b.assertion_id,
        verdict="uncertain",
        confidence=0.0,
        rationale=f"Definition judge degraded to uncertain: {reason}",
        evidence_spans=[],
    )


class DefinitionJudge(Protocol):
    """Anything that produces a DefinitionJudgeVerdict for a pair of definitions."""

    def judge(self, a: Assertion, b: Assertion) -> DefinitionJudgeVerdict: ...


class FixtureDefinitionJudge:
    """Returns canned verdicts keyed by the canonical assertion-id pair."""

    def __init__(self, fixtures: Mapping[tuple[str, str], DefinitionJudgeVerdict]) -> None:
        self._fixtures = dict(fixtures)

    def judge(self, a: Assertion, b: Assertion) -> DefinitionJudgeVerdict:
        key = (min(a.assertion_id, b.assertion_id), max(a.assertion_id, b.assertion_id))
        if key in self._fixtures:
            return self._fixtures[key]
        return definition_uncertain_fallback(a, b, reason="no fixture configured for pair")


class LLMDefinitionJudge:
    """Provider-backed judge with retry-on-malformed and degraded fallback."""

    def __init__(self, provider: DefinitionJudgeProvider, *, max_retries: int = 2) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self._provider = provider
        self._max_retries = max_retries

    def judge(self, a: Assertion, b: Assertion) -> DefinitionJudgeVerdict:
        system = render_definition_system_prompt()
        user = render_definition_user_prompt(a, b)
        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            try:
                payload = self._provider.request_payload(system, user)
                return DefinitionJudgeVerdict.from_payload(a, b, payload)
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)
                _log.warning(
                    "Definition judge attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    last_error,
                )
        return definition_uncertain_fallback(a, b, reason=last_error or "unknown error")
