"""Provider-agnostic shapes for the definition-inconsistency judge.

Mirrors :mod:`consistency_checker.check.providers.base` — a strict Pydantic
schema plus a Protocol — but with the three-verdict vocabulary specific to the
definition detector. Keeping the surface narrow ensures an LLM in the
definition path can't accidentally claim a contradiction verdict and vice
versa.
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

DefinitionVerdictLabel = Literal[
    "definition_consistent",
    "definition_divergent",
    "uncertain",
]

#: Verdicts that count as a confirmed definition inconsistency in run totals
#: and reports. Parallel to ``CONTRADICTION_VERDICTS`` in the contradiction
#: detector.
DEFINITION_INCONSISTENCY_VERDICTS: frozenset[str] = frozenset({"definition_divergent"})


class DefinitionJudgePayload(BaseModel):
    """Strict schema every definition-judge provider must satisfy."""

    model_config = ConfigDict(extra="forbid")

    verdict: DefinitionVerdictLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    evidence_spans: list[str] = Field(default_factory=list)


class DefinitionJudgeProvider(Protocol):
    """Anything that can produce a validated :class:`DefinitionJudgePayload`."""

    def request_payload(self, system: str, user: str) -> DefinitionJudgePayload: ...
