"""Multi-party (triangle) judge — ADR-0006 F3.

Sibling of :mod:`consistency_checker.check.llm_judge`. Takes a three-assertion
:class:`Triangle` and asks the provider whether any subset is jointly
contradictory. The schema is :class:`MultiPartyJudgePayload`; on validation
failure we retry up to ``max_retries`` times and then degrade to an
``uncertain`` verdict so the pipeline never crashes.

The pair judge and multi-party judge stay decoupled: separate Protocol,
separate prompts, separate provider classes. The downstream audit table
(``multi_party_findings``) carries the multi-party verdict label without
touching the pair :class:`Finding` schema.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from consistency_checker.check.providers.base import (
    MultiPartyJudgePayload,
    MultiPartyJudgeProvider,
    MultiPartyVerdictLabel,
)
from consistency_checker.check.triangle import Triangle
from consistency_checker.logging_setup import get_logger

_log = get_logger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "judge_multi_system.txt"
USER_PROMPT_PATH = PROMPTS_DIR / "judge_multi_user.txt"

SUBSET_LABELS: tuple[str, str, str] = ("A", "B", "C")


@dataclass(frozen=True, slots=True)
class MultiPartyJudgeVerdict:
    """Verdict for one triangle plus its provenance — outward-facing surface."""

    assertion_ids: tuple[str, str, str]
    verdict: MultiPartyVerdictLabel
    confidence: float
    rationale: str
    contradicting_subset: tuple[str, ...] = field(default_factory=tuple)
    evidence_spans: list[str] = field(default_factory=list)

    @classmethod
    def from_payload(
        cls, triangle: Triangle, payload: MultiPartyJudgePayload
    ) -> MultiPartyJudgeVerdict:
        return cls(
            assertion_ids=triangle.assertion_ids,
            verdict=payload.verdict,
            confidence=payload.confidence,
            rationale=payload.rationale,
            contradicting_subset=tuple(payload.contradicting_subset),
            evidence_spans=list(payload.evidence_spans),
        )


def render_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def render_user_prompt(triangle: Triangle) -> str:
    """Substitute the triangle's three assertions into the user template.

    Labels A, B, C are assigned in the triangle's canonical (assertion-id
    sorted) order, so the same triangle always renders to the same prompt
    bytes — useful for caching and golden tests.
    """
    template = USER_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template.replace("{doc_a_id}", triangle.a.doc_id)
        .replace("{doc_b_id}", triangle.b.doc_id)
        .replace("{doc_c_id}", triangle.c.doc_id)
        .replace("{assertion_a_text}", triangle.a.assertion_text)
        .replace("{assertion_b_text}", triangle.b.assertion_text)
        .replace("{assertion_c_text}", triangle.c.assertion_text)
    )


def uncertain_fallback(triangle: Triangle, reason: str) -> MultiPartyJudgeVerdict:
    return MultiPartyJudgeVerdict(
        assertion_ids=triangle.assertion_ids,
        verdict="uncertain",
        confidence=0.0,
        rationale=f"Multi-party judge degraded to uncertain: {reason}",
        contradicting_subset=(),
        evidence_spans=[],
    )


class MultiPartyJudge(Protocol):
    """Anything that produces a verdict for a triangle of assertions."""

    def judge(self, triangle: Triangle) -> MultiPartyJudgeVerdict: ...


class FixtureMultiPartyJudge:
    """Returns canned verdicts keyed by the triangle's canonical assertion-id tuple.

    Falls back to a low-confidence ``uncertain`` verdict when a triangle is
    missing from the fixture map, so tests only enumerate the cases they care
    about. Mirrors :class:`FixtureJudge` for the pair stage.
    """

    def __init__(
        self,
        fixtures: Mapping[tuple[str, str, str], MultiPartyJudgeVerdict],
    ) -> None:
        self._fixtures = dict(fixtures)

    def judge(self, triangle: Triangle) -> MultiPartyJudgeVerdict:
        if triangle.assertion_ids in self._fixtures:
            return self._fixtures[triangle.assertion_ids]
        return uncertain_fallback(triangle, reason="no fixture configured for triangle")


class LLMMultiPartyJudge:
    """Provider-backed multi-party judge with retry-on-malformed and fallback."""

    def __init__(self, provider: MultiPartyJudgeProvider, *, max_retries: int = 2) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self._provider = provider
        self._max_retries = max_retries

    def judge(self, triangle: Triangle) -> MultiPartyJudgeVerdict:
        system = render_system_prompt()
        user = render_user_prompt(triangle)
        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            try:
                payload = self._provider.request_payload(system, user)
                return MultiPartyJudgeVerdict.from_payload(triangle, payload)
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)
                _log.warning(
                    "Multi-party judge attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    last_error,
                )
        return uncertain_fallback(triangle, reason=last_error or "unknown error")
