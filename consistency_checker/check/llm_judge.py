"""Stage B LLM judge.

Orchestrates calling a provider with the structured prompt and translating the
validated :class:`JudgePayload` into a :class:`JudgeVerdict` for downstream
consumers. Includes retry-with-repair: on Pydantic validation failure the call
is retried up to ``max_retries`` times before degrading to an ``uncertain``
verdict with confidence 0 — never crashing the pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from consistency_checker.check.providers.base import (
    JudgePayload,
    JudgeProvider,
    JudgeVerdictLabel,
)
from consistency_checker.extract.schema import Assertion
from consistency_checker.logging_setup import get_logger

_log = get_logger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "judge_system.txt"
USER_PROMPT_PATH = PROMPTS_DIR / "judge_user.txt"


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """Verdict for one pair plus its provenance — outward-facing surface."""

    assertion_a_id: str
    assertion_b_id: str
    verdict: JudgeVerdictLabel
    confidence: float
    rationale: str
    evidence_spans: list[str] = field(default_factory=list)

    @classmethod
    def from_payload(cls, a: Assertion, b: Assertion, payload: JudgePayload) -> JudgeVerdict:
        return cls(
            assertion_a_id=a.assertion_id,
            assertion_b_id=b.assertion_id,
            verdict=payload.verdict,
            confidence=payload.confidence,
            rationale=payload.rationale,
            evidence_spans=list(payload.evidence_spans),
        )


def render_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def render_user_prompt(a: Assertion, b: Assertion, *, numeric_context: str | None = None) -> str:
    """Render the user prompt.

    If ``numeric_context`` is provided (a non-empty string of E3 disagreement
    lines), it's spliced in as a "## Numeric context" block between the
    assertions and the decision instruction. When ``None`` the placeholder
    collapses to an empty line so prose-only prompts stay golden-stable.
    """
    context_block = f"\n## Numeric context\n\n{numeric_context}\n" if numeric_context else ""
    template = USER_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template.replace("{doc_a_id}", a.doc_id)
        .replace("{doc_b_id}", b.doc_id)
        .replace("{assertion_a_text}", a.assertion_text)
        .replace("{assertion_b_text}", b.assertion_text)
        .replace("{numeric_context_block}", context_block)
    )


def uncertain_fallback(a: Assertion, b: Assertion, reason: str) -> JudgeVerdict:
    return JudgeVerdict(
        assertion_a_id=a.assertion_id,
        assertion_b_id=b.assertion_id,
        verdict="uncertain",
        confidence=0.0,
        rationale=f"Judge degraded to uncertain: {reason}",
        evidence_spans=[],
    )


class Judge(Protocol):
    """Anything that produces a JudgeVerdict for a pair of assertions."""

    def judge(
        self, a: Assertion, b: Assertion, *, numeric_context: str | None = None
    ) -> JudgeVerdict: ...


class FixtureJudge:
    """Returns canned verdicts keyed by the canonical assertion-id pair.

    Falls back to a low-confidence ``uncertain`` verdict when a pair is missing,
    so test fixtures only need to enumerate the interesting cases.
    """

    def __init__(self, fixtures: Mapping[tuple[str, str], JudgeVerdict]) -> None:
        self._fixtures = dict(fixtures)

    def judge(
        self, a: Assertion, b: Assertion, *, numeric_context: str | None = None
    ) -> JudgeVerdict:
        # Fixture judge ignores numeric_context — it's keyed only on assertion ids.
        del numeric_context
        key = self._canonical(a, b)
        if key in self._fixtures:
            return self._fixtures[key]
        return uncertain_fallback(a, b, reason="no fixture configured for pair")

    @staticmethod
    def _canonical(a: Assertion, b: Assertion) -> tuple[str, str]:
        return (
            min(a.assertion_id, b.assertion_id),
            max(a.assertion_id, b.assertion_id),
        )


class LLMJudge:
    """Provider-backed judge with retry-on-malformed and degraded fallback."""

    def __init__(self, provider: JudgeProvider, *, max_retries: int = 2) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self._provider = provider
        self._max_retries = max_retries

    def judge(
        self, a: Assertion, b: Assertion, *, numeric_context: str | None = None
    ) -> JudgeVerdict:
        system = render_system_prompt()
        user = render_user_prompt(a, b, numeric_context=numeric_context)
        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            try:
                payload = self._provider.request_payload(system, user)
                return JudgeVerdict.from_payload(a, b, payload)
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)
                _log.warning(
                    "Judge attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    last_error,
                )
        return uncertain_fallback(a, b, reason=last_error or "unknown error")
