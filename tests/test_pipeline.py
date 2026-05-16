"""Tests for the pipeline orchestration module.

Verifies that the pipeline can be configured with different judge providers
and that the factory functions wire up the correct provider instances.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from consistency_checker.check.providers.anthropic import AnthropicProvider
from consistency_checker.check.providers.moonshot import MoonshotJudgeProvider
from consistency_checker.check.providers.openai import OpenAIProvider
from consistency_checker.config import Config
from consistency_checker.pipeline import make_judge


@pytest.mark.e2e_fixture
def test_pipeline_with_moonshot_provider(tmp_path: Path, monkeypatch) -> None:
    """Test that pipeline factories work with moonshot provider configured.

    Verifies that:
    1. Config accepts "moonshot" as a judge_provider value
    2. make_judge() returns a MoonshotJudgeProvider instance
    3. The pipeline provider wiring doesn't crash on instantiation
    """
    # Patch MOONSHOT_API_KEY so the provider can initialize
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key-for-testing")

    # Create minimal config with moonshot provider
    config = Config(
        corpus_dir=tmp_path,
        judge_provider="moonshot",
    )

    # Verify that config accepted the moonshot provider
    assert config.judge_provider == "moonshot"

    # Verify that make_judge() returns a MoonshotJudgeProvider instance
    judge = make_judge(config)
    assert judge is not None
    # The judge is an LLMJudge that wraps the provider
    from consistency_checker.check.llm_judge import LLMJudge

    assert isinstance(judge, LLMJudge)
    # The underlying provider should be MoonshotJudgeProvider
    assert isinstance(judge._provider, MoonshotJudgeProvider)


@pytest.mark.e2e_fixture
def test_pipeline_with_anthropic_provider(tmp_path: Path, monkeypatch) -> None:
    """Verify that the anthropic provider still works with the factory."""
    # Mock API key (not needed for this test but good practice)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-for-testing")

    config = Config(
        corpus_dir=tmp_path,
        judge_provider="anthropic",
    )

    assert config.judge_provider == "anthropic"

    judge = make_judge(config)
    from consistency_checker.check.llm_judge import LLMJudge

    assert isinstance(judge, LLMJudge)
    assert isinstance(judge._provider, AnthropicProvider)


@pytest.mark.e2e_fixture
def test_pipeline_with_openai_provider(tmp_path: Path, monkeypatch) -> None:
    """Verify that the openai provider still works with the factory."""
    # Mock API key for OpenAI provider initialization
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-testing")

    config = Config(
        corpus_dir=tmp_path,
        judge_provider="openai",
    )

    assert config.judge_provider == "openai"

    judge = make_judge(config)
    from consistency_checker.check.llm_judge import LLMJudge

    assert isinstance(judge, LLMJudge)
    assert isinstance(judge._provider, OpenAIProvider)
