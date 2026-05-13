"""Provider-agnostic shapes for the Stage B judge.

The :class:`JudgePayload` Pydantic model is the strict schema the judge expects
back from every provider. Implementations are responsible for forcing the LLM
into this shape via the provider-native structured-output mechanism (tool-use
for Anthropic, JSON-schema response_format for OpenAI) and validating the
result before returning it.
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

JudgeVerdictLabel = Literal["contradiction", "not_contradiction", "uncertain"]


class JudgePayload(BaseModel):
    """Strict schema every provider must satisfy before returning a verdict."""

    model_config = ConfigDict(extra="forbid")

    verdict: JudgeVerdictLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    evidence_spans: list[str] = Field(default_factory=list)


class JudgeProvider(Protocol):
    """Anything that can produce a validated :class:`JudgePayload` from a prompt."""

    def request_payload(self, system: str, user: str) -> JudgePayload: ...
