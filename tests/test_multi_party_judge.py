"""Tests for the multi-party (triangle) judge — ADR-0006 F3."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from consistency_checker.check.multi_party_judge import (
    FixtureMultiPartyJudge,
    LLMMultiPartyJudge,
    MultiPartyJudgeVerdict,
    render_system_prompt,
    render_user_prompt,
)
from consistency_checker.check.providers.anthropic import (
    MULTI_PARTY_TOOL_NAME as ANTHROPIC_MULTI_PARTY_TOOL_NAME,
)
from consistency_checker.check.providers.anthropic import (
    AnthropicMultiPartyProvider,
    parse_multi_party_tool_response,
)
from consistency_checker.check.providers.base import MultiPartyJudgePayload
from consistency_checker.check.providers.openai import OpenAIMultiPartyProvider
from consistency_checker.check.triangle import Triangle
from consistency_checker.extract.schema import Assertion


def _triangle(
    text_a: str = "All employees get four weeks vacation.",
    text_b: str = "Engineers are employees.",
    text_c: str = "Engineers get two weeks vacation.",
) -> Triangle:
    a = Assertion.build("doc_a", text_a)
    b = Assertion.build("doc_b", text_b)
    c = Assertion.build("doc_c", text_c)
    triangle_assertions = sorted([a, b, c], key=lambda x: x.assertion_id)
    return Triangle(
        a=triangle_assertions[0],
        b=triangle_assertions[1],
        c=triangle_assertions[2],
        edge_scores=(
            (
                triangle_assertions[0].assertion_id,
                triangle_assertions[1].assertion_id,
                0.82,
            ),
            (
                triangle_assertions[0].assertion_id,
                triangle_assertions[2].assertion_id,
                0.74,
            ),
            (
                triangle_assertions[1].assertion_id,
                triangle_assertions[2].assertion_id,
                0.91,
            ),
        ),
    )


# --- MultiPartyJudgePayload schema ----------------------------------------


def test_multi_party_payload_accepts_valid_contradiction() -> None:
    payload = MultiPartyJudgePayload.model_validate(
        {
            "verdict": "multi_party_contradiction",
            "confidence": 0.88,
            "rationale": "A says all employees get 4w; B says engineers are employees; C says engineers get 2w.",
            "contradicting_subset": ["A", "B", "C"],
            "evidence_spans": ["four weeks", "two weeks"],
        }
    )
    assert payload.verdict == "multi_party_contradiction"
    assert payload.contradicting_subset == ["A", "B", "C"]


def test_multi_party_payload_rejects_pair_label() -> None:
    """Plain 'contradiction' is reserved for the pair judge."""
    with pytest.raises(ValidationError):
        MultiPartyJudgePayload.model_validate(
            {"verdict": "contradiction", "confidence": 0.5, "rationale": "..."}
        )


def test_multi_party_payload_rejects_numeric_short_circuit() -> None:
    with pytest.raises(ValidationError):
        MultiPartyJudgePayload.model_validate(
            {"verdict": "numeric_short_circuit", "confidence": 1.0, "rationale": "..."}
        )


def test_multi_party_payload_rejects_subset_when_not_contradiction() -> None:
    """A non-contradiction verdict must come with an empty subset."""
    with pytest.raises(ValidationError):
        MultiPartyJudgePayload.model_validate(
            {
                "verdict": "uncertain",
                "confidence": 0.4,
                "rationale": "unclear scope",
                "contradicting_subset": ["A", "B"],
            }
        )


def test_multi_party_payload_allows_empty_subset_for_not_contradiction() -> None:
    payload = MultiPartyJudgePayload.model_validate(
        {
            "verdict": "not_contradiction",
            "confidence": 0.7,
            "rationale": "scopes don't overlap",
        }
    )
    assert payload.contradicting_subset == []


def test_multi_party_payload_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        MultiPartyJudgePayload.model_validate(
            {
                "verdict": "uncertain",
                "confidence": 0.5,
                "rationale": "...",
                "extra": "noise",
            }
        )


def test_multi_party_payload_requires_rationale() -> None:
    with pytest.raises(ValidationError):
        MultiPartyJudgePayload.model_validate(
            {"verdict": "uncertain", "confidence": 0.5, "rationale": ""}
        )


# --- Prompt rendering -----------------------------------------------------


def test_system_prompt_mentions_subset_and_three_labels() -> None:
    prompt = render_system_prompt()
    assert "multi_party_contradiction" in prompt
    assert "contradicting_subset" in prompt
    assert "A" in prompt and "B" in prompt and "C" in prompt


def test_user_prompt_substitutes_three_assertions() -> None:
    triangle = _triangle()
    out = render_user_prompt(triangle)
    assert triangle.a.assertion_text in out
    assert triangle.b.assertion_text in out
    assert triangle.c.assertion_text in out
    assert triangle.a.doc_id in out
    assert triangle.b.doc_id in out
    assert triangle.c.doc_id in out
    assert "{assertion_a_text}" not in out
    assert "{doc_c_id}" not in out


def test_user_prompt_is_deterministic() -> None:
    triangle = _triangle()
    assert render_user_prompt(triangle) == render_user_prompt(triangle)


# --- FixtureMultiPartyJudge -----------------------------------------------


def test_fixture_multi_party_judge_returns_canned_verdict() -> None:
    triangle = _triangle()
    expected = MultiPartyJudgeVerdict(
        assertion_ids=triangle.assertion_ids,
        verdict="multi_party_contradiction",
        confidence=0.9,
        rationale="A ∧ B ⇒ ¬C",
        contradicting_subset=("A", "B", "C"),
        evidence_spans=["four weeks", "two weeks"],
    )
    judge = FixtureMultiPartyJudge({triangle.assertion_ids: expected})
    assert judge.judge(triangle) == expected


def test_fixture_multi_party_judge_unknown_triangle_returns_uncertain() -> None:
    triangle = _triangle()
    judge = FixtureMultiPartyJudge({})
    out = judge.judge(triangle)
    assert out.verdict == "uncertain"
    assert out.confidence == 0.0
    assert out.contradicting_subset == ()


# --- LLMMultiPartyJudge orchestration -------------------------------------


class _MockProvider:
    """Configurable mock satisfying MultiPartyJudgeProvider."""

    def __init__(self, *, payloads: list[Any]) -> None:
        self._payloads = list(payloads)
        self.last_system: str | None = None
        self.last_user: str | None = None
        self.call_count = 0

    def request_payload(self, system: str, user: str) -> MultiPartyJudgePayload:
        self.last_system = system
        self.last_user = user
        self.call_count += 1
        item = self._payloads.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, MultiPartyJudgePayload):
            return item
        return MultiPartyJudgePayload.model_validate(item)


def test_llm_multi_party_judge_returns_verdict_on_first_success() -> None:
    triangle = _triangle()
    payload = MultiPartyJudgePayload(
        verdict="multi_party_contradiction",
        confidence=0.85,
        rationale="A ∧ B ⇒ ¬C",
        contradicting_subset=["A", "B", "C"],
    )
    provider = _MockProvider(payloads=[payload])
    judge = LLMMultiPartyJudge(provider, max_retries=2)
    verdict = judge.judge(triangle)
    assert verdict.verdict == "multi_party_contradiction"
    assert verdict.assertion_ids == triangle.assertion_ids
    assert verdict.contradicting_subset == ("A", "B", "C")
    assert provider.call_count == 1


def test_llm_multi_party_judge_retries_on_validation_error() -> None:
    triangle = _triangle()
    good_payload = MultiPartyJudgePayload(
        verdict="uncertain", confidence=0.3, rationale="scope unclear"
    )
    provider = _MockProvider(payloads=[ValueError("first attempt malformed"), good_payload])
    judge = LLMMultiPartyJudge(provider, max_retries=2)
    verdict = judge.judge(triangle)
    assert verdict.verdict == "uncertain"
    assert verdict.confidence == 0.3
    assert provider.call_count == 2


def test_llm_multi_party_judge_falls_back_after_retries() -> None:
    triangle = _triangle()
    provider = _MockProvider(payloads=[ValueError("a"), ValueError("b"), ValueError("c")])
    judge = LLMMultiPartyJudge(provider, max_retries=2)
    verdict = judge.judge(triangle)
    assert verdict.verdict == "uncertain"
    assert verdict.confidence == 0.0
    assert "c" in verdict.rationale
    assert provider.call_count == 3


def test_llm_multi_party_judge_rejects_negative_retries() -> None:
    with pytest.raises(ValueError):
        LLMMultiPartyJudge(provider=_MockProvider(payloads=[]), max_retries=-1)


def test_llm_multi_party_judge_passes_rendered_prompts() -> None:
    triangle = _triangle()
    payload = MultiPartyJudgePayload(
        verdict="not_contradiction", confidence=0.6, rationale="independent claims"
    )
    provider = _MockProvider(payloads=[payload])
    LLMMultiPartyJudge(provider).judge(triangle)
    assert provider.last_system is not None
    assert provider.last_user is not None
    assert triangle.a.assertion_text in provider.last_user
    assert triangle.c.assertion_text in provider.last_user


# --- Anthropic multi-party provider parsing -------------------------------


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
        block = SimpleNamespace(
            type="tool_use",
            name=ANTHROPIC_MULTI_PARTY_TOOL_NAME,
            input=self._payload,
        )
        return SimpleNamespace(content=[block])


def test_anthropic_multi_party_provider_parses_tool_use() -> None:
    fake = _FakeAnthropic(
        {
            "verdict": "multi_party_contradiction",
            "confidence": 0.8,
            "rationale": "chained conflict",
            "contradicting_subset": ["A", "B", "C"],
            "evidence_spans": ["four weeks", "two weeks"],
        }
    )
    provider = AnthropicMultiPartyProvider(client=fake)  # type: ignore[arg-type]
    payload = provider.request_payload(system="sys", user="usr")
    assert payload.verdict == "multi_party_contradiction"
    assert payload.contradicting_subset == ["A", "B", "C"]
    assert fake.last_kwargs is not None
    assert fake.last_kwargs["tool_choice"] == {
        "type": "tool",
        "name": ANTHROPIC_MULTI_PARTY_TOOL_NAME,
    }


def test_anthropic_multi_party_parse_missing_tool_raises() -> None:
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text="oops")])
    with pytest.raises(ValueError, match="No tool_use block"):
        parse_multi_party_tool_response(response)


def test_anthropic_multi_party_parse_invalid_payload_raises() -> None:
    block = SimpleNamespace(
        type="tool_use",
        name=ANTHROPIC_MULTI_PARTY_TOOL_NAME,
        input={"verdict": "bogus", "confidence": 0.5, "rationale": "..."},
    )
    response = SimpleNamespace(content=[block])
    with pytest.raises(ValidationError):
        parse_multi_party_tool_response(response)


# --- OpenAI multi-party provider parsing ----------------------------------


class _FakeOpenAI:
    def __init__(self, parsed: MultiPartyJudgePayload | None) -> None:
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


def test_openai_multi_party_provider_returns_validated_payload() -> None:
    payload = MultiPartyJudgePayload(
        verdict="multi_party_contradiction",
        confidence=0.9,
        rationale="chain",
        contradicting_subset=["A", "B", "C"],
    )
    fake = _FakeOpenAI(parsed=payload)
    provider = OpenAIMultiPartyProvider(client=fake)  # type: ignore[arg-type]
    out = provider.request_payload(system="sys", user="usr")
    assert out == payload
    assert fake.last_kwargs is not None
    assert fake.last_kwargs["response_format"] is MultiPartyJudgePayload


def test_openai_multi_party_provider_raises_when_parsed_is_none() -> None:
    provider = OpenAIMultiPartyProvider(client=_FakeOpenAI(parsed=None))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no validated payload"):
        provider.request_payload(system="sys", user="usr")


# --- Live tests (gated) ---------------------------------------------------


@pytest.mark.live
def test_anthropic_multi_party_live_call() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    triangle = _triangle()
    provider = AnthropicMultiPartyProvider()
    judge = LLMMultiPartyJudge(provider)
    verdict = judge.judge(triangle)
    assert verdict.verdict == "multi_party_contradiction"


@pytest.mark.live
def test_openai_multi_party_live_call() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    triangle = _triangle()
    provider = OpenAIMultiPartyProvider()
    judge = LLMMultiPartyJudge(provider)
    verdict = judge.judge(triangle)
    assert verdict.verdict == "multi_party_contradiction"
