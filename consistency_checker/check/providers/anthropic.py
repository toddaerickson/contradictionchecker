"""Anthropic provider for the Stage B judge.

Forces JSON output via tool-use: a ``record_verdict`` tool whose schema mirrors
:class:`JudgePayload`. ``tool_choice`` is set so the model must call the tool
on every turn. The tool_use block is then validated through Pydantic; a
malformed block raises :class:`ValidationError` and the caller decides whether
to retry.

The multi-party variant (ADR-0006, F3) reuses the tool-use mechanism with a
separate ``record_multi_party_verdict`` tool whose schema is
:class:`MultiPartyJudgePayload`. Sibling provider so the pair surface stays
narrow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from consistency_checker.check.providers.base import (
    JudgePayload,
    MultiPartyJudgePayload,
)
from consistency_checker.check.providers.definition_base import DefinitionJudgePayload

if TYPE_CHECKING:
    import anthropic

TOOL_NAME = "record_verdict"
TOOL_SCHEMA: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": "Record the contradiction verdict for the two given assertions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["contradiction", "not_contradiction", "uncertain"],
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string"},
            "evidence_spans": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["verdict", "confidence", "rationale"],
    },
}

MULTI_PARTY_TOOL_NAME = "record_multi_party_verdict"
MULTI_PARTY_TOOL_SCHEMA: dict[str, Any] = {
    "name": MULTI_PARTY_TOOL_NAME,
    "description": (
        "Record the joint-contradiction verdict for the three assertions "
        "labelled A, B, C and the subset that contradicts (if any)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": [
                    "multi_party_contradiction",
                    "not_contradiction",
                    "uncertain",
                ],
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string"},
            "contradicting_subset": {
                "type": "array",
                "description": (
                    "Subset of assertion labels (e.g. ['A','B','C']) that jointly "
                    "contradict. Must be empty when verdict is not "
                    "multi_party_contradiction."
                ),
                "items": {"type": "string"},
            },
            "evidence_spans": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["verdict", "confidence", "rationale"],
    },
}


def parse_tool_response(response: Any) -> JudgePayload:
    """Extract the ``record_verdict`` tool_use block from a Messages response."""
    blocks = getattr(response, "content", None) or []
    for block in blocks:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == TOOL_NAME:
            payload = getattr(block, "input", None) or {}
            return JudgePayload.model_validate(payload)
    raise ValueError(f"No tool_use block named {TOOL_NAME!r} found in response")


def parse_multi_party_tool_response(response: Any) -> MultiPartyJudgePayload:
    """Extract the ``record_multi_party_verdict`` tool_use block."""
    blocks = getattr(response, "content", None) or []
    for block in blocks:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == MULTI_PARTY_TOOL_NAME
        ):
            payload = getattr(block, "input", None) or {}
            return MultiPartyJudgePayload.model_validate(payload)
    raise ValueError(f"No tool_use block named {MULTI_PARTY_TOOL_NAME!r} found in response")


class AnthropicProvider:
    """Calls Anthropic Claude and returns a validated :class:`JudgePayload`."""

    def __init__(
        self,
        *,
        client: anthropic.Anthropic | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
    ) -> None:
        import anthropic

        self._client = client or anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def request_payload(self, system: str, user: str) -> JudgePayload:
        response = self._client.messages.create(  # type: ignore[call-overload]
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": TOOL_NAME},
        )
        return parse_tool_response(response)


DEFINITION_TOOL_NAME = "record_definition_verdict"
DEFINITION_TOOL_SCHEMA: dict[str, Any] = {
    "name": DEFINITION_TOOL_NAME,
    "description": (
        "Record whether two definitions of the same term are consistent or materially divergent."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": [
                    "definition_consistent",
                    "definition_divergent",
                    "uncertain",
                ],
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string"},
            "evidence_spans": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["verdict", "confidence", "rationale"],
    },
}


def parse_definition_tool_response(response: Any) -> DefinitionJudgePayload:
    """Extract the ``record_definition_verdict`` tool_use block."""
    blocks = getattr(response, "content", None) or []
    for block in blocks:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == DEFINITION_TOOL_NAME
        ):
            payload = getattr(block, "input", None) or {}
            return DefinitionJudgePayload.model_validate(payload)
    raise ValueError(f"No tool_use block named {DEFINITION_TOOL_NAME!r} found in response")


class AnthropicDefinitionProvider:
    """Anthropic provider for the definition-inconsistency judge."""

    def __init__(
        self,
        *,
        client: anthropic.Anthropic | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
    ) -> None:
        if client is None:
            import anthropic

            self._client: Any = anthropic.Anthropic()
        else:
            self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def request_payload(self, system: str, user: str) -> DefinitionJudgePayload:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[DEFINITION_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": DEFINITION_TOOL_NAME},
        )
        return parse_definition_tool_response(response)


class AnthropicMultiPartyProvider:
    """Multi-party (triangle) variant — same tool-use mechanism, separate tool."""

    def __init__(
        self,
        *,
        client: anthropic.Anthropic | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
    ) -> None:
        import anthropic

        self._client = client or anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def request_payload(self, system: str, user: str) -> MultiPartyJudgePayload:
        response = self._client.messages.create(  # type: ignore[call-overload]
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[MULTI_PARTY_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": MULTI_PARTY_TOOL_NAME},
        )
        return parse_multi_party_tool_response(response)
