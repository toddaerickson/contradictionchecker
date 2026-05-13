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

The multi-party (ADR-0006, F3) judge has its own sibling shapes —
:data:`MultiPartyVerdictLabel`, :class:`MultiPartyJudgePayload`,
:class:`MultiPartyJudgeProvider` — so the pairwise type surface stays narrow
and an LLM in the multi-party path can't claim a pairwise verdict.
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
MultiPartyVerdictLabel = Literal[
    "multi_party_contradiction",
    "not_contradiction",
    "uncertain",
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


class MultiPartyJudgePayload(BaseModel):
    """Strict schema for the triangle (multi-party) judge.

    ``contradicting_subset`` records the subset of input assertions the judge
    believes is jointly contradictory. Labels in the subset are short string
    handles (``"A"``, ``"B"``, ``"C"``) supplied by the prompt — the caller
    maps them back to assertion ids. When ``verdict != "multi_party_contradiction"``
    the subset is expected to be empty (validator enforces this so a stray
    subset doesn't sneak in via a non-contradiction verdict).
    """

    model_config = ConfigDict(extra="forbid")

    verdict: MultiPartyVerdictLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    contradicting_subset: list[str] = Field(default_factory=list)
    evidence_spans: list[str] = Field(default_factory=list)

    def model_post_init(self, _context: object, /) -> None:
        if self.verdict != "multi_party_contradiction" and self.contradicting_subset:
            raise ValueError(
                "contradicting_subset must be empty when verdict is not multi_party_contradiction"
            )


class MultiPartyJudgeProvider(Protocol):
    """Anything that can produce a validated :class:`MultiPartyJudgePayload`."""

    def request_payload(self, system: str, user: str) -> MultiPartyJudgePayload: ...
