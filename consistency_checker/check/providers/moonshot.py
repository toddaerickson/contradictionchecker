"""Moonshot (Kimi) provider for the Stage B judge.

Uses openai.OpenAI SDK with Moonshot's API endpoint (https://api.moonshot.ai/v1).
Moonshot's API is OpenAI-compatible, supporting JSON schema structured output.

The pairwise judge uses response_format to enforce JSON matching JudgePayload schema.
The multi-party judge uses response_format to enforce MultiPartyJudgePayload schema.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from consistency_checker.check.providers.base import (
    JudgePayload,
    MultiPartyJudgePayload,
)

if TYPE_CHECKING:
    import openai


class MoonshotJudgeProvider:
    """Pairwise judge using Moonshot (Kimi) API with JSON schema structured output."""

    def __init__(self, api_key: str | None = None, model: str = "kimi-k2.6"):
        """Initialize the Moonshot judge provider.

        Args:
            api_key: Moonshot API key. If None, reads from MOONSHOT_API_KEY env var.
            model: Model name (default: kimi-k2.6)
        """
        import openai

        self.model = model
        api_key = api_key or os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError(
                "MOONSHOT_API_KEY not set. Set via env var, .env file, or pass to __init__"
            )

        self.client: openai.OpenAI = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.ai/v1",
        )

    def request_payload(self, system: str, user: str) -> JudgePayload:
        """Request a verdict from Moonshot for the given assertions.

        Args:
            system: System prompt
            user: User (assertion pair) prompt

        Returns:
            JudgePayload with verdict, confidence, rationale, evidence_spans
        """
        response: Any = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=JudgePayload,
        )

        # response.choices[0].message.parsed is a JudgePayload instance
        payload = response.choices[0].message.parsed
        if payload is None:
            raise ValueError("Moonshot API returned None payload")

        if isinstance(payload, JudgePayload):
            return payload
        return JudgePayload.model_validate(payload)


class MoonshotMultiPartyJudgeProvider:
    """Multi-party (triangle) judge using Moonshot API with JSON schema structured output."""

    def __init__(self, api_key: str | None = None, model: str = "kimi-k2.6"):
        """Initialize the Moonshot multi-party judge provider.

        Args:
            api_key: Moonshot API key. If None, reads from MOONSHOT_API_KEY env var.
            model: Model name (default: kimi-k2.6)
        """
        import openai

        self.model = model
        api_key = api_key or os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError(
                "MOONSHOT_API_KEY not set. Set via env var, .env file, or pass to __init__"
            )

        self.client: openai.OpenAI = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.ai/v1",
        )

    def request_payload(self, system: str, user: str) -> MultiPartyJudgePayload:
        """Request a multi-party verdict from Moonshot.

        Args:
            system: System prompt
            user: User (assertion triple) prompt

        Returns:
            MultiPartyJudgePayload with verdict, confidence, rationale, contradicting_subset
        """
        response: Any = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=MultiPartyJudgePayload,
        )

        payload = response.choices[0].message.parsed
        if payload is None:
            raise ValueError("Moonshot API returned None payload")

        if isinstance(payload, MultiPartyJudgePayload):
            return payload
        return MultiPartyJudgePayload.model_validate(payload)
