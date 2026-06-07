"""Tests for the definition judge — prompts, payload routing, fixture + LLM backends."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from consistency_checker.check.definition_judge import (
    DefinitionJudgeVerdict,
    FixtureDefinitionJudge,
    LLMDefinitionJudge,
    definition_short_circuit_verdict,
    definition_uncertain_fallback,
    render_definition_system_prompt,
    render_definition_user_prompt,
)
from consistency_checker.check.providers.definition_base import (
    DEFINITION_CONSISTENT_AUTO,
    DefinitionJudgePayload,
)
from consistency_checker.check.providers.openai import OpenAIDefinitionProvider
from consistency_checker.extract.schema import Assertion


def test_openai_definition_provider_empty_choices_raises_clean_error() -> None:
    """An empty choices list must raise ValueError, not IndexError."""

    def parse(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(choices=[])

    fake = SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    )
    provider = OpenAIDefinitionProvider(client=fake)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no validated payload"):
        provider.request_payload(system="sys", user="usr")


def _mae_pair() -> tuple[Assertion, Assertion]:
    a = Assertion.build(
        "docA",
        '"MAE" means a material adverse effect on the Borrower\'s business.',
        kind="definition",
        term="MAE",
        definition_text="a material adverse effect on the Borrower's business",
    )
    b = Assertion.build(
        "docB",
        '"MAE" means an effect that materially impairs the Borrower\'s ability to perform.',
        kind="definition",
        term="MAE",
        definition_text="an effect that materially impairs the Borrower's ability to perform",
    )
    return a, b


def test_definition_system_prompt_explains_verdicts() -> None:
    s = render_definition_system_prompt()
    assert "definition_consistent" in s
    assert "definition_divergent" in s
    assert "uncertain" in s


def test_definition_user_prompt_renders_both_definitions() -> None:
    a, b = _mae_pair()
    user = render_definition_user_prompt(a, b)
    assert "MAE" in user
    assert "docA" in user
    assert "docB" in user
    assert "material adverse effect on the Borrower's business" in user
    assert "materially impairs" in user


def test_definition_user_prompt_requires_term() -> None:
    a = Assertion.build("docA", "no term", kind="definition")
    b = Assertion.build("docB", "no term", kind="definition")
    import pytest

    with pytest.raises(ValueError, match=r"requires both assertions to have a `term`"):
        render_definition_user_prompt(a, b)


def test_fixture_definition_judge_returns_canned_verdict() -> None:
    a, b = _mae_pair()
    verdict = DefinitionJudgeVerdict(
        assertion_a_id=min(a.assertion_id, b.assertion_id),
        assertion_b_id=max(a.assertion_id, b.assertion_id),
        verdict="definition_divergent",
        rationale="A scopes business; B scopes performance.",
        evidence_spans=["business", "ability to perform"],
    )
    judge = FixtureDefinitionJudge(
        {(min(a.assertion_id, b.assertion_id), max(a.assertion_id, b.assertion_id)): verdict}
    )
    out = judge.judge(a, b)
    assert out.verdict == "definition_divergent"


def test_fixture_definition_judge_falls_back_uncertain() -> None:
    a, b = _mae_pair()
    judge = FixtureDefinitionJudge({})
    out = judge.judge(a, b)
    assert out.verdict == "uncertain"


class _StubProvider:
    def __init__(self, payload: DefinitionJudgePayload) -> None:
        self.payload = payload
        self.calls = 0

    def request_payload(self, system: str, user: str) -> DefinitionJudgePayload:
        del system, user
        self.calls += 1
        return self.payload


def test_llm_definition_judge_round_trips_payload() -> None:
    a, b = _mae_pair()
    payload = DefinitionJudgePayload(
        verdict="definition_divergent",
        rationale="scope shift",
        evidence_spans=["business", "ability to perform"],
    )
    provider = _StubProvider(payload)
    judge = LLMDefinitionJudge(provider)
    out = judge.judge(a, b)
    assert out.verdict == "definition_divergent"
    assert provider.calls == 1


class _BadProvider:
    def __init__(self) -> None:
        self.calls = 0

    def request_payload(self, system: str, user: str) -> DefinitionJudgePayload:
        del system, user
        self.calls += 1
        raise ValueError("malformed payload")


def test_llm_definition_judge_retries_then_falls_back() -> None:
    a, b = _mae_pair()
    provider = _BadProvider()
    judge = LLMDefinitionJudge(provider, max_retries=2)
    out = judge.judge(a, b)
    assert out.verdict == "uncertain"
    assert "malformed payload" in out.rationale
    assert provider.calls == 3  # initial + 2 retries


def test_llm_definition_judge_rejects_negative_retries() -> None:
    import pytest

    with pytest.raises(ValueError, match=r"max_retries must be"):
        LLMDefinitionJudge(
            _StubProvider(  # type: ignore[arg-type]
                DefinitionJudgePayload(verdict="uncertain", rationale="x")
            ),
            max_retries=-1,
        )


def test_definition_uncertain_fallback_shape() -> None:
    a, b = _mae_pair()
    out = definition_uncertain_fallback(a, b, reason="missing API key")
    assert out.verdict == "uncertain"
    assert "missing API key" in out.rationale
    assert out.evidence_spans == []


def test_definition_short_circuit_verdict() -> None:
    a = Assertion.build(
        "docA",
        '"Board" means the board of directors.',
        kind="definition",
        term="Board",
        definition_text="the board of directors",
    )
    b = Assertion.build(
        "docB",
        '"Board" means the board of directors.',
        kind="definition",
        term="Board",
        definition_text="the board of directors",
    )
    v = definition_short_circuit_verdict(a, b)
    assert v.verdict == DEFINITION_CONSISTENT_AUTO
    assert v.assertion_a_id == a.assertion_id
    assert v.assertion_b_id == b.assertion_id
    assert "machine-resolved" in v.rationale


def test_system_prompt_has_equivalence_rule_and_examples() -> None:
    text = render_definition_system_prompt()
    assert "differ only in whitespace, punctuation, capitalization" in text
    assert "Examples:" in text
    assert "Affiliate" in text  # the divergent worked example
