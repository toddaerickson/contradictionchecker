"""Tests for Moonshot (Kimi) judge provider."""

from __future__ import annotations

import pytest

from consistency_checker.check.providers.base import JudgePayload
from consistency_checker.check.providers.moonshot import MoonshotJudgeProvider


def test_moonshot_judge_provider_returns_valid_payload(monkeypatch):
    """Test that MoonshotJudgeProvider.request_payload returns a JudgePayload."""

    class MockClient:
        class Beta:
            class Chat:
                class Completions:
                    @staticmethod
                    def parse(**kwargs):
                        # Return an object that looks like a parsed response
                        parsed_payload = JudgePayload(
                            verdict="contradiction",
                            confidence=0.95,
                            rationale="The two assertions directly conflict.",
                            evidence_spans=["assertion A text", "assertion B text"],
                        )

                        class Message:
                            parsed = parsed_payload

                        class Choice:
                            message = Message()

                        class Response:
                            choices = [Choice()]

                        return Response()

                completions = Completions()

            chat = Chat()

        beta = Beta()

    # Create provider and test
    provider = MoonshotJudgeProvider(api_key="sk-test-key", model="kimi-k2.6")
    provider.client = MockClient()

    payload = provider.request_payload(system="You are a judge.", user="Do A and B contradict?")

    assert isinstance(payload, JudgePayload)
    assert payload.verdict == "contradiction"
    assert payload.confidence == 0.95


def test_moonshot_judge_provider_missing_api_key(monkeypatch):
    """Test that missing API key raises ValueError."""
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)

    with pytest.raises(ValueError, match="MOONSHOT_API_KEY not set"):
        MoonshotJudgeProvider()


def test_moonshot_judge_provider_with_explicit_api_key():
    """Test that explicit API key is used."""
    provider = MoonshotJudgeProvider(api_key="sk-test-explicit-key", model="kimi-k2.6")
    assert provider.model == "kimi-k2.6"
    assert provider.client is not None


def test_moonshot_judge_validates_payload():
    """Test that malformed responses raise ValidationError."""
    from pydantic import ValidationError

    class BadMockClient:
        class Beta:
            class Chat:
                class Completions:
                    @staticmethod
                    def parse(**kwargs):
                        class BadResponse:
                            class Choice:
                                class Message:
                                    parsed = {
                                        "verdict": "invalid_verdict",  # Not in enum
                                        "confidence": 0.5,
                                        "rationale": "test",
                                    }

                                message = Message()

                            choices = [Choice()]

                        return BadResponse()

                completions = Completions()

            chat = Chat()

        beta = Beta()

    provider = MoonshotJudgeProvider(api_key="sk-test")
    provider.client = BadMockClient()

    # This should raise during response.choices[0].message.parsed access
    # (OpenAI SDK validates via pydantic.parse internally)
    with pytest.raises((ValidationError, AttributeError, ValueError)):
        provider.request_payload("system", "user")
