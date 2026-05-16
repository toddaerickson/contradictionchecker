"""Tests for Moonshot (Kimi) judge provider."""

from __future__ import annotations

import pytest

from consistency_checker.check.providers.base import JudgePayload, MultiPartyJudgePayload
from consistency_checker.check.providers.moonshot import (
    MoonshotJudgeProvider,
    MoonshotMultiPartyJudgeProvider,
)


def test_moonshot_judge_provider_returns_valid_payload(monkeypatch):
    """Test that MoonshotJudgeProvider.request_payload returns a JudgePayload."""

    def make_response():
        parsed_payload = JudgePayload(
            verdict="contradiction",
            confidence=0.95,
            rationale="The two assertions directly conflict.",
            evidence_spans=["assertion A text", "assertion B text"],
        )

        class Message:
            pass

        msg = Message()
        msg.parsed = parsed_payload

        class Choice:
            pass

        choice = Choice()
        choice.message = msg

        class Response:
            pass

        resp = Response()
        resp.choices = [choice]
        return resp

    class MockClient:
        class Beta:
            class Chat:
                class Completions:
                    @staticmethod
                    def parse(**kwargs):
                        return make_response()

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

    def make_bad_response():
        class Message:
            pass

        msg = Message()
        msg.parsed = {
            "verdict": "invalid_verdict",  # Not in enum
            "confidence": 0.5,
            "rationale": "test",
        }

        class Choice:
            pass

        choice = Choice()
        choice.message = msg

        class BadResponse:
            pass

        resp = BadResponse()
        resp.choices = [choice]
        return resp

    class BadMockClient:
        class Beta:
            class Chat:
                class Completions:
                    @staticmethod
                    def parse(**kwargs):
                        return make_bad_response()

                completions = Completions()

            chat = Chat()

        beta = Beta()

    provider = MoonshotJudgeProvider(api_key="sk-test")
    provider.client = BadMockClient()

    # This should raise during response.choices[0].message.parsed access
    # (OpenAI SDK validates via pydantic.parse internally)
    with pytest.raises((ValidationError, AttributeError, ValueError)):
        provider.request_payload("system", "user")


@pytest.mark.live
def test_moonshot_judge_provider_real_api_pairwise():
    """Integration test: call real Moonshot API for pairwise judgment.

    Requires MOONSHOT_API_KEY env var. Marked @pytest.mark.live to skip in CI.
    """
    provider = MoonshotJudgeProvider(model="kimi-k2.6")

    system_prompt = (
        "You are a judge determining if two assertions contradict each other. "
        "Return a JSON verdict."
    )
    user_prompt = (
        "Assertion A: The Earth is round.\n"
        "Assertion B: The Earth is a perfect sphere.\n"
        "Do these assertions contradict? Respond with JSON."
    )

    payload = provider.request_payload(system_prompt, user_prompt)

    # Validate structure
    assert isinstance(payload, JudgePayload)
    assert payload.verdict in ["contradiction", "not_contradiction", "uncertain"]
    assert 0.0 <= payload.confidence <= 1.0
    assert len(payload.rationale) > 0


@pytest.mark.live
def test_moonshot_judge_provider_real_api_multi_party():
    """Integration test: call real Moonshot API for multi-party judgment.

    Requires MOONSHOT_API_KEY env var. Marked @pytest.mark.live to skip in CI.
    """
    provider = MoonshotMultiPartyJudgeProvider(model="kimi-k2.6")

    system_prompt = "You are a judge. Return a JSON verdict for three assertions."
    user_prompt = (
        "Assertion A: Water boils at 100°C.\n"
        "Assertion B: Water boils at higher temperatures at higher altitudes.\n"
        "Assertion C: Water boils at 100°C regardless of altitude.\n"
        "Determine if any subset jointly contradicts. Respond with JSON."
    )

    payload = provider.request_payload(system_prompt, user_prompt)

    # Validate structure
    assert isinstance(payload, MultiPartyJudgePayload)
    assert payload.verdict in ["multi_party_contradiction", "not_contradiction", "uncertain"]
    assert 0.0 <= payload.confidence <= 1.0
    assert len(payload.rationale) > 0

    # If verdict is multi_party_contradiction, subset should not be empty
    if payload.verdict == "multi_party_contradiction":
        assert len(payload.contradicting_subset) > 0
