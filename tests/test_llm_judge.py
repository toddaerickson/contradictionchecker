"""Tests for the Stage B LLM judge and its provider implementations."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from consistency_checker.check.llm_judge import (
    FixtureJudge,
    JudgeVerdict,
    LLMJudge,
    render_system_prompt,
    render_user_prompt,
)
from consistency_checker.check.providers.anthropic import (
    TOOL_NAME as ANTHROPIC_TOOL_NAME,
)
from consistency_checker.check.providers.anthropic import (
    AnthropicProvider,
    parse_tool_response,
)
from consistency_checker.check.providers.base import JudgePayload
from consistency_checker.check.providers.openai import OpenAIProvider
from consistency_checker.extract.schema import Assertion


def make_assertion(doc_id: str, text: str) -> Assertion:
    return Assertion.build(doc_id, text)


# --- JudgePayload schema ----------------------------------------------------


def test_judge_payload_accepts_valid_input() -> None:
    payload = JudgePayload.model_validate(
        {
            "verdict": "contradiction",
            "confidence": 0.85,
            "rationale": "Both about Q1 revenue with opposite signs.",
            "evidence_spans": ["grew 12%", "declined 5%"],
        }
    )
    assert payload.verdict == "contradiction"
    assert payload.confidence == 0.85


def test_judge_payload_rejects_unknown_verdict() -> None:
    with pytest.raises(ValidationError):
        JudgePayload.model_validate({"verdict": "maybe", "confidence": 0.5, "rationale": "..."})


def test_judge_payload_rejects_numeric_short_circuit_from_llm() -> None:
    """numeric_short_circuit is pipeline-emitted only — providers must not return it."""
    with pytest.raises(ValidationError):
        JudgePayload.model_validate(
            {
                "verdict": "numeric_short_circuit",
                "confidence": 1.0,
                "rationale": "...",
            }
        )


def test_judge_payload_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        JudgePayload.model_validate(
            {"verdict": "contradiction", "confidence": 1.5, "rationale": "..."}
        )


def test_judge_payload_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        JudgePayload.model_validate(
            {
                "verdict": "contradiction",
                "confidence": 0.5,
                "rationale": "...",
                "extra": "noise",
            }
        )


def test_judge_payload_requires_rationale() -> None:
    with pytest.raises(ValidationError):
        JudgePayload.model_validate(
            {"verdict": "contradiction", "confidence": 0.5, "rationale": ""}
        )


# --- Prompt rendering -------------------------------------------------------


def test_system_prompt_loads_and_mentions_structured_output() -> None:
    prompt = render_system_prompt()
    assert "verdict" in prompt
    assert "confidence" in prompt
    assert "rationale" in prompt


def test_user_prompt_substitutes_assertion_text() -> None:
    a = make_assertion("doc_a", "Revenue grew 12%.")
    b = make_assertion("doc_b", "Revenue declined 5%.")
    out = render_user_prompt(a, b)
    assert "Revenue grew 12%." in out
    assert "Revenue declined 5%." in out
    assert "doc_a" in out
    assert "doc_b" in out
    assert "{assertion_a_text}" not in out


# --- FixtureJudge -----------------------------------------------------------


def test_fixture_judge_returns_canned_verdict() -> None:
    a = make_assertion("doc_a", "Revenue grew 12%.")
    b = make_assertion("doc_b", "Revenue declined 5%.")
    expected = JudgeVerdict(
        assertion_a_id=min(a.assertion_id, b.assertion_id),
        assertion_b_id=max(a.assertion_id, b.assertion_id),
        verdict="contradiction",
        confidence=0.9,
        rationale="opposite signs at same scope",
    )
    judge = FixtureJudge(
        {(min(a.assertion_id, b.assertion_id), max(a.assertion_id, b.assertion_id)): expected}
    )
    assert judge.judge(a, b) == expected
    # Order-invariant: the canonical key holds in either direction.
    assert judge.judge(b, a) == expected


def test_fixture_judge_unknown_pair_returns_uncertain() -> None:
    a = make_assertion("doc_a", "X")
    b = make_assertion("doc_b", "Y")
    judge = FixtureJudge({})
    out = judge.judge(a, b)
    assert out.verdict == "uncertain"
    assert out.confidence == 0.0


# --- LLMJudge orchestration (mocked provider) -------------------------------


class _MockProvider:
    """Configurable mock satisfying JudgeProvider."""

    def __init__(self, *, payloads: list[Any]) -> None:
        # Each call pops the next payload; if it's an Exception, raise it.
        self._payloads = list(payloads)
        self.last_system: str | None = None
        self.last_user: str | None = None
        self.call_count = 0

    def request_payload(self, system: str, user: str) -> JudgePayload:
        self.last_system = system
        self.last_user = user
        self.call_count += 1
        item = self._payloads.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, JudgePayload):
            return item
        return JudgePayload.model_validate(item)


def test_llm_judge_returns_verdict_on_first_success() -> None:
    a = make_assertion("doc_a", "X")
    b = make_assertion("doc_b", "Y")
    payload = JudgePayload(verdict="contradiction", confidence=0.8, rationale="they conflict")
    provider = _MockProvider(payloads=[payload])
    judge = LLMJudge(provider, max_retries=2)
    verdict = judge.judge(a, b)
    assert verdict.verdict == "contradiction"
    assert verdict.assertion_a_id == a.assertion_id
    assert verdict.assertion_b_id == b.assertion_id
    assert provider.call_count == 1


def test_llm_judge_retries_on_validation_error() -> None:
    a = make_assertion("doc_a", "X")
    b = make_assertion("doc_b", "Y")
    good_payload = JudgePayload(
        verdict="not_contradiction", confidence=0.6, rationale="different scope"
    )
    provider = _MockProvider(
        payloads=[
            ValueError("malformed first attempt"),
            good_payload,
        ]
    )
    judge = LLMJudge(provider, max_retries=2)
    verdict = judge.judge(a, b)
    assert verdict.verdict == "not_contradiction"
    assert provider.call_count == 2


def test_llm_judge_falls_back_to_uncertain_after_retries() -> None:
    a = make_assertion("doc_a", "X")
    b = make_assertion("doc_b", "Y")
    provider = _MockProvider(
        payloads=[
            ValueError("attempt 1"),
            ValueError("attempt 2"),
            ValueError("attempt 3"),
        ]
    )
    judge = LLMJudge(provider, max_retries=2)
    verdict = judge.judge(a, b)
    assert verdict.verdict == "uncertain"
    assert verdict.confidence == 0.0
    assert "attempt 3" in verdict.rationale
    assert provider.call_count == 3


def test_llm_judge_rejects_negative_retries() -> None:
    with pytest.raises(ValueError):
        LLMJudge(provider=_MockProvider(payloads=[]), max_retries=-1)


def test_llm_judge_passes_rendered_prompts_to_provider() -> None:
    a = make_assertion("doc_a", "Revenue grew 12%.")
    b = make_assertion("doc_b", "Revenue declined 5%.")
    payload = JudgePayload(verdict="contradiction", confidence=0.9, rationale="opposing signs")
    provider = _MockProvider(payloads=[payload])
    LLMJudge(provider).judge(a, b)
    assert provider.last_system is not None
    assert provider.last_user is not None
    assert "Revenue grew 12%." in provider.last_user
    assert "Revenue declined 5%." in provider.last_user


# --- Anthropic provider parsing --------------------------------------------


class _FakeAnthropic:
    def __init__(self, payload: dict[str, Any] | Exception) -> None:
        self._payload = payload
        self.last_kwargs: dict[str, object] | None = None

    @property
    def messages(self) -> _FakeAnthropic:
        return self

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.last_kwargs = kwargs
        if isinstance(self._payload, Exception):
            raise self._payload
        block = SimpleNamespace(type="tool_use", name=ANTHROPIC_TOOL_NAME, input=self._payload)
        return SimpleNamespace(content=[block])


def test_anthropic_provider_parses_tool_use() -> None:
    fake = _FakeAnthropic(
        {
            "verdict": "contradiction",
            "confidence": 0.7,
            "rationale": "opposite signs",
            "evidence_spans": ["grew", "declined"],
        }
    )
    provider = AnthropicProvider(client=fake)  # type: ignore[arg-type]
    payload = provider.request_payload(system="sys", user="usr")
    assert payload.verdict == "contradiction"
    assert payload.confidence == 0.7
    # Check that tool_choice and tools wiring is right.
    assert fake.last_kwargs is not None
    assert fake.last_kwargs["tool_choice"] == {
        "type": "tool",
        "name": ANTHROPIC_TOOL_NAME,
    }


def test_anthropic_parse_tool_response_missing_block_raises() -> None:
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text="oops")])
    with pytest.raises(ValueError, match="No tool_use block"):
        parse_tool_response(response)


def test_anthropic_parse_tool_response_invalid_payload_raises() -> None:
    block = SimpleNamespace(
        type="tool_use",
        name=ANTHROPIC_TOOL_NAME,
        input={"verdict": "bogus", "confidence": 0.5, "rationale": "..."},
    )
    response = SimpleNamespace(content=[block])
    with pytest.raises(ValidationError):
        parse_tool_response(response)


# --- OpenAI provider parsing -----------------------------------------------


class _FakeOpenAI:
    def __init__(self, parsed: JudgePayload | None) -> None:
        self._parsed = parsed
        self.last_kwargs: dict[str, object] | None = None

    @property
    def beta(self) -> _FakeOpenAI:
        return self

    @property
    def chat(self) -> _FakeOpenAI:
        return self

    @property
    def completions(self) -> _FakeOpenAI:
        return self

    def parse(self, **kwargs: object) -> SimpleNamespace:
        self.last_kwargs = kwargs
        message = SimpleNamespace(parsed=self._parsed)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


def test_openai_provider_returns_validated_payload() -> None:
    payload = JudgePayload(
        verdict="not_contradiction",
        confidence=0.55,
        rationale="different time period",
    )
    fake = _FakeOpenAI(parsed=payload)
    provider = OpenAIProvider(client=fake)  # type: ignore[arg-type]
    out = provider.request_payload(system="sys", user="usr")
    assert out == payload
    assert fake.last_kwargs is not None
    assert fake.last_kwargs["response_format"] is JudgePayload


def test_openai_provider_raises_when_parsed_is_none() -> None:
    provider = OpenAIProvider(client=_FakeOpenAI(parsed=None))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no validated payload"):
        provider.request_payload(system="sys", user="usr")


# --- Live tests (gated) -----------------------------------------------------


@pytest.mark.live
def test_anthropic_provider_live_call() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    a = Assertion.build("doc_a", "Revenue grew 12% in fiscal 2025.")
    b = Assertion.build("doc_b", "Revenue declined 5% in fiscal 2025.")
    provider = AnthropicProvider()
    judge = LLMJudge(provider)
    verdict = judge.judge(a, b)
    assert verdict.verdict == "contradiction"


@pytest.mark.live
def test_openai_provider_live_call() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    a = Assertion.build("doc_a", "Revenue grew 12% in fiscal 2025.")
    b = Assertion.build("doc_b", "Revenue declined 5% in fiscal 2025.")
    provider = OpenAIProvider()
    judge = LLMJudge(provider)
    verdict = judge.judge(a, b)
    assert verdict.verdict == "contradiction"
