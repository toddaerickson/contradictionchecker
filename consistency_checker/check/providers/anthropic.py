"""Anthropic provider for the Stage B judge.

Forces JSON output via tool-use: a ``record_verdict`` tool whose schema mirrors
:class:`JudgePayload`. ``tool_choice`` is set so the model must call the tool
on every turn. The tool_use block is then validated through Pydantic; a
malformed block raises :class:`ValidationError` and the caller decides whether
to retry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from consistency_checker.check.providers.base import JudgePayload

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


def parse_tool_response(response: Any) -> JudgePayload:
    """Extract the ``record_verdict`` tool_use block from a Messages response."""
    blocks = getattr(response, "content", None) or []
    for block in blocks:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == TOOL_NAME:
            payload = getattr(block, "input", None) or {}
            return JudgePayload.model_validate(payload)
    raise ValueError(f"No tool_use block named {TOOL_NAME!r} found in response")


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
