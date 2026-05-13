"""Provider-agnostic shapes for the Stage B judge.

The :class:`JudgePayload` Pydantic model is the strict schema the judge expects
back from every provider. Implementations are responsible for forcing the LLM
into this shape via the provider-native structured-output mechanism (tool-use
for Anthropic, JSON-schema response_format for OpenAI) and validating the
result before returning it.

Two related Literal types:

- :data:`LLMVerdictLabel` — what an LLM provider may return. Three values.
- :data:`JudgeVerdictLabel` — what downstream consumers (audit, report) see.
  Adds ``numeric_short_circuit`` (ADR-0005), emitted by ``pipeline.check`` when
  a deterministic sign-flip is detected and the LLM is bypassed.
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

LLMVerdictLabel = Literal["contradiction", "not_contradiction", "uncertain"]
JudgeVerdictLabel = Literal[
    "contradiction",
    "not_contradiction",
    "uncertain",
    "numeric_short_circuit",
]


class JudgePayload(BaseModel):
    """Strict schema every provider must satisfy before returning a verdict."""

    model_config = ConfigDict(extra="forbid")

    verdict: LLMVerdictLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    evidence_spans: list[str] = Field(default_factory=list)


class JudgeProvider(Protocol):
    """Anything that can produce a validated :class:`JudgePayload` from a prompt."""

    def request_payload(self, system: str, user: str) -> JudgePayload: ...
