"""OpenAI provider for the Stage B judge.

Uses the SDK's structured-output ``parse`` helper (``client.beta.chat.completions.parse``)
which accepts a Pydantic model as ``response_format`` and returns the
validated instance directly. Available in ``openai>=1.40``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from consistency_checker.check.providers.base import JudgePayload

if TYPE_CHECKING:
    import openai


class OpenAIProvider:
    """Calls OpenAI Chat Completions with structured output and returns a JudgePayload."""

    def __init__(
        self,
        *,
        client: openai.OpenAI | None = None,
        model: str = "gpt-4o-2024-08-06",
    ) -> None:
        import openai

        self._client = client or openai.OpenAI()
        self._model = model

    def request_payload(self, system: str, user: str) -> JudgePayload:
        response: Any = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=JudgePayload,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI structured-output parse returned no validated payload")
        if isinstance(parsed, JudgePayload):
            return parsed
        return JudgePayload.model_validate(parsed)
