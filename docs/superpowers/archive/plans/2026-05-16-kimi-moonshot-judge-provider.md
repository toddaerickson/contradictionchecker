# Kimi (Moonshot) Judge Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Kimi (Moonshot AI) as an experimental judge provider using the OpenAI SDK pattern, supporting both pairwise and multi-party contradiction detection.

**Architecture:** Reuse the OpenAI SDK (`openai.OpenAI`) with Moonshot's API endpoint. Implement `MoonshotJudgeProvider` and `MoonshotMultiPartyJudgeProvider` classes that call the Moonshot API with JSON schema structured output. Update the judge factory in `pipeline.py` and config in `config.py` to support provider selection.

**Tech Stack:** Python 3.11+, openai SDK (already a dependency), pydantic (validation), pytest (testing)

---

## File Structure

| File | Status | Responsibility |
|------|--------|-----------------|
| `consistency_checker/check/providers/moonshot.py` | Create | `MoonshotJudgeProvider`, `MoonshotMultiPartyJudgeProvider` classes using OpenAI SDK |
| `consistency_checker/pipeline.py` | Modify | Add `"moonshot"` cases to `make_judge()` and `make_multi_party_judge()` factories |
| `consistency_checker/config.py` | Modify | Add `"moonshot"` to provider Literal and document the option |
| `tests/check/providers/test_moonshot.py` | Create | Unit tests (mocked) + integration test (`@pytest.mark.live`) |
| `docs/decisions/ADR-0007-moonshot-experimental-judge.md` | Create | Architecture Decision Record |

---

## Task 1: Implement MoonshotJudgeProvider (Pairwise)

**Files:**
- Create: `consistency_checker/check/providers/moonshot.py`
- Test: `tests/check/providers/test_moonshot.py`

- [ ] **Step 1: Write failing unit test for MoonshotJudgeProvider**

Create `tests/check/providers/test_moonshot.py`:

```python
import pytest
from consistency_checker.check.providers.base import JudgePayload
from consistency_checker.check.providers.moonshot import MoonshotJudgeProvider


def test_moonshot_judge_provider_returns_valid_payload(monkeypatch):
    """Test that MoonshotJudgeProvider.request_payload returns a JudgePayload."""
    # Mock the openai.OpenAI client to avoid real API calls
    mock_response = {
        "verdict": "contradiction",
        "confidence": 0.95,
        "rationale": "The two assertions directly conflict.",
        "evidence_spans": ["assertion A text", "assertion B text"],
    }
    
    class MockClient:
        class Beta:
            class Completions:
                @staticmethod
                def parse(**kwargs):
                    # Return an object that looks like a parsed response
                    class ParsedResponse:
                        verdict = "contradiction"
                        confidence = 0.95
                        rationale = "The two assertions directly conflict."
                        evidence_spans = ["assertion A text", "assertion B text"]
                    return ParsedResponse()
            
            completions = Completions()
        
        beta = Beta()
    
    # Create provider and test
    provider = MoonshotJudgeProvider(api_key="sk-test-key", model="kimi-k2.6")
    provider.client = MockClient()
    
    payload = provider.request_payload(
        system="You are a judge.",
        user="Do A and B contradict?"
    )
    
    assert isinstance(payload, JudgePayload)
    assert payload.verdict == "contradiction"
    assert payload.confidence == 0.95
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/terickson/contradictionchecker
pytest tests/check/providers/test_moonshot.py::test_moonshot_judge_provider_returns_valid_payload -v
```

Expected: `FAILED` with `ModuleNotFoundError: No module named 'consistency_checker.check.providers.moonshot'`

- [ ] **Step 3: Create the moonshot.py module with MoonshotJudgeProvider class**

Create `consistency_checker/check/providers/moonshot.py`:

```python
"""Moonshot (Kimi) provider for the Stage B judge.

Uses openai.OpenAI SDK with Moonshot's API endpoint (https://api.moonshot.ai/v1).
Moonshot's API is OpenAI-compatible, supporting JSON schema structured output.

The pairwise judge uses response_format to enforce JSON matching JudgePayload schema.
The multi-party judge uses response_format to enforce MultiPartyJudgePayload schema.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from consistency_checker.check.providers.base import (
    JudgePayload,
    JudgeProvider,
    MultiPartyJudgePayload,
    MultiPartyJudgeProvider,
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
                "MOONSHOT_API_KEY not set. Set via environment variable, .env file, or pass to __init__"
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
        response = self.client.beta.chat.completions.parse(
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

        return payload


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
                "MOONSHOT_API_KEY not set. Set via environment variable, .env file, or pass to __init__"
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
        response = self.client.beta.chat.completions.parse(
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

        return payload
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/check/providers/test_moonshot.py::test_moonshot_judge_provider_returns_valid_payload -v
```

Expected: `PASSED`

- [ ] **Step 5: Write additional unit tests for edge cases**

Add to `tests/check/providers/test_moonshot.py`:

```python
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
        beta = Beta()
    
    provider = MoonshotJudgeProvider(api_key="sk-test")
    provider.client = BadMockClient()
    
    # This should raise during response.choices[0].message.parsed access
    # (OpenAI SDK validates via pydantic.parse internally)
    with pytest.raises((ValidationError, AttributeError, ValueError)):
        provider.request_payload("system", "user")
```

- [ ] **Step 6: Run all unit tests for moonshot provider**

```bash
pytest tests/check/providers/test_moonshot.py -v
```

Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add consistency_checker/check/providers/moonshot.py tests/check/providers/test_moonshot.py
git commit -m "feat: add Moonshot (Kimi) judge provider with pairwise + multi-party support"
```

---

## Task 2: Add Moonshot to Provider Factory

**Files:**
- Modify: `consistency_checker/pipeline.py`

- [ ] **Step 1: Examine existing make_judge() factory**

```bash
grep -A 20 "def make_judge" /home/terickson/contradictionchecker/consistency_checker/pipeline.py
```

Note the current structure (likely has branches for "anthropic" and "openai").

- [ ] **Step 2: Add import for MoonshotJudgeProvider at the top of pipeline.py**

After other provider imports:

```python
from consistency_checker.check.providers.moonshot import (
    MoonshotJudgeProvider,
    MoonshotMultiPartyJudgeProvider,
)
```

- [ ] **Step 3: Add moonshot case to make_judge() function**

In `make_judge()`, add a branch like:

```python
elif config.provider == "moonshot":
    return MoonshotJudgeProvider(model="kimi-k2.6")
```

Place it after the openai branch and before any else/default.

- [ ] **Step 4: Add moonshot case to make_multi_party_judge() function**

In `make_multi_party_judge()`, add:

```python
elif config.provider == "moonshot":
    return MoonshotMultiPartyJudgeProvider(model="kimi-k2.6")
```

- [ ] **Step 5: Test that imports work**

```bash
python -c "from consistency_checker.pipeline import make_judge; print('Import successful')"
```

Expected: `Import successful` (no errors)

- [ ] **Step 6: Commit**

```bash
git add consistency_checker/pipeline.py
git commit -m "feat: add Moonshot provider to judge factory (make_judge and make_multi_party_judge)"
```

---

## Task 3: Update Config to Support Moonshot Provider

**Files:**
- Modify: `consistency_checker/config.py`

- [ ] **Step 1: Read the config file to understand structure**

```bash
grep -A 5 "provider" /home/terickson/contradictionchecker/consistency_checker/config.py | head -20
```

Identify where `provider` is defined as a Literal type.

- [ ] **Step 2: Add "moonshot" to the provider Literal**

Update the provider field (approximately line where provider is defined):

From:
```python
provider: Literal["anthropic", "openai"] = "anthropic"
```

To:
```python
provider: Literal["anthropic", "openai", "moonshot"] = "anthropic"
```

- [ ] **Step 3: Add documentation for the moonshot option**

Add a comment above or in the docstring:

```python
# provider: Choice of judge provider.
#   - "anthropic": Claude (Anthropic), tool-use structured output
#   - "openai": GPT-4 (OpenAI), JSON schema structured output
#   - "moonshot": Kimi (Moonshot AI), experimental, JSON schema via OpenAI SDK
```

- [ ] **Step 4: Test that config parses moonshot**

```bash
python -c "
from consistency_checker.config import Config
cfg = Config.from_yaml({'provider': 'moonshot'})
print(f'Provider: {cfg.provider}')
"
```

Expected: `Provider: moonshot`

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/config.py
git commit -m "chore: add moonshot to provider options in config"
```

---

## Task 4: Write Integration Test for Moonshot API

**Files:**
- Modify: `tests/check/providers/test_moonshot.py`

- [ ] **Step 1: Add live integration test (marked to skip by default)**

Add to `tests/check/providers/test_moonshot.py`:

```python
import pytest


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
```

- [ ] **Step 2: Run unit tests (should all pass)**

```bash
pytest tests/check/providers/test_moonshot.py -v -m "not live"
```

Expected: All non-live tests pass

- [ ] **Step 3: Run live integration test (optional, requires API key)**

```bash
pytest tests/check/providers/test_moonshot.py -v -m live
```

Expected: Both live tests pass (if MOONSHOT_API_KEY is set)

- [ ] **Step 4: Commit**

```bash
git add tests/check/providers/test_moonshot.py
git commit -m "test: add live integration tests for Moonshot provider (pairwise + multi-party)"
```

---

## Task 5: Create Architecture Decision Record (ADR)

**Files:**
- Create: `docs/decisions/ADR-0007-moonshot-experimental-judge.md`

- [ ] **Step 1: Check if docs/decisions/ directory exists**

```bash
ls -la /home/terickson/contradictionchecker/docs/decisions/
```

If not, create it:

```bash
mkdir -p /home/terickson/contradictionchecker/docs/decisions/
```

- [ ] **Step 2: Create ADR file**

Create `docs/decisions/ADR-0007-moonshot-experimental-judge.md`:

```markdown
# ADR-0007: Add Moonshot (Kimi) as Experimental Judge Provider

**Date:** 2026-05-16  
**Status:** Accepted

## Context

The consistency-checker pipeline currently supports two judge providers:
- Anthropic Claude (via tool-use structured output)
- OpenAI GPT-4 (via JSON schema structured output)

Moonshot AI's Kimi model is a strong alternative for contradiction detection, with potential cost advantages ($0.60/$2.50 per M tokens vs. Anthropic's higher rates). Before committing to Kimi as a production alternative (ADR-0008 will address this), we want to test it experimentally and compare its contradiction-detection quality.

## Decision

Add **MoonshotJudgeProvider** and **MoonshotMultiPartyJudgeProvider** as experimental judge options in the consistency-checker pipeline.

### Why Moonshot?

1. **Cost:** Significantly cheaper than Anthropic for equivalent quality (pending validation)
2. **API Compatibility:** Moonshot's API is OpenAI-compatible, reducing implementation burden
3. **Async Potential:** Long-horizon reasoning and extended context (256K tokens) suit deep contradiction analysis
4. **Research:** Wanted to understand Kimi's performance on a real task before committing

## Implementation

- Create `consistency_checker/check/providers/moonshot.py` with pairwise and multi-party providers
- Reuse openai SDK with Moonshot's API endpoint (https://api.moonshot.ai/v1)
- Use JSON schema structured output (same as OpenAI)
- Add "moonshot" as a provider option in config (selected at runtime via CLI or env var)
- Mark as experimental in naming and docs until quality validation is complete

## Testing

- Unit tests with mocked API responses (hermetic, CI-safe)
- Live integration tests (marked @pytest.mark.live, manual testing only)
- Quality comparison: Run on corpus of contradictions, compare verdicts to Anthropic/OpenAI

## Consequences

### Positive

- Low-risk addition (experimental mode, not default)
- Can evaluate Kimi's quality on real contradictions before production use
- Cost savings validated before commit
- Foundation for future cost-optimization work (ADR-0008)

### Negative

- Adds dependency on Moonshot's API stability
- Requires MOONSHOT_API_KEY env var setup (documented in .env setup)
- Integration tests require live API access (not hermetic)

## Related

- **ADR-0008 (future):** Cost-optimization via provider selection based on contradiction complexity
- **Task:** Kimi CLI extension auth setup (2026-05-16) — provides .env configuration reused here
- **Spec:** 2026-05-16-kimi-moonshot-judge-provider.md

## Notes for Future Reviewers

If Kimi's quality is insufficient (lower contradiction detection accuracy), deprecate this provider and remove it. If quality is comparable, proceed with ADR-0008 to make it a cost-optimization option (use Kimi for simple pairs, Anthropic for complex).
```

- [ ] **Step 3: Verify ADR is readable**

```bash
cat /home/terickson/contradictionchecker/docs/decisions/ADR-0007-moonshot-experimental-judge.md
```

- [ ] **Step 4: Commit**

```bash
git add docs/decisions/ADR-0007-moonshot-experimental-judge.md
git commit -m "docs: ADR-0007 -- Moonshot (Kimi) as experimental judge provider"
```

---

## Task 6: End-to-End Test (Pipeline Integration)

**Files:**
- Test: Verify `pipeline.check()` works with moonshot provider

- [ ] **Step 1: Write a simple hermetic end-to-end test**

Add to `tests/test_pipeline.py` or create new test file:

```python
def test_pipeline_with_moonshot_provider(tmp_path):
    """Test that pipeline.check() works with moonshot provider configured."""
    from consistency_checker.config import Config
    from consistency_checker.pipeline import check
    from consistency_checker.index.assertion_store import AssertionStore
    
    # Create minimal config with moonshot provider
    config = Config(
        provider="moonshot",
        nli_threshold=0.7,
    )
    
    # Create minimal assertion store
    store = AssertionStore(":memory:")
    
    # Add a test assertion pair (would fail to run without mocking,
    # so this just tests that the config is accepted)
    # In real test, mock the Moonshot API
    
    # This test primarily verifies that:
    # 1. Config accepts "moonshot"
    # 2. make_judge() returns a MoonshotJudgeProvider
    # 3. Pipeline doesn't crash on instantiation
    
    from consistency_checker.check.providers.moonshot import MoonshotJudgeProvider
    from consistency_checker.pipeline import make_judge
    
    judge = make_judge(config)
    assert isinstance(judge, MoonshotJudgeProvider)
```

- [ ] **Step 2: Run the integration test**

```bash
pytest tests/test_pipeline.py::test_pipeline_with_moonshot_provider -v
```

Expected: `PASSED`

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline.py  # or new test file
git commit -m "test: verify moonshot provider integrates with pipeline"
```

---

## Task 7: Verify All Tests Pass

**Files:**
- Test: Run full test suite

- [ ] **Step 1: Run all non-live tests**

```bash
cd /home/terickson/contradictionchecker
uv run pytest -m "not live" -v
```

Expected: All pass (no failures)

- [ ] **Step 2: Check code formatting**

```bash
uv run ruff check . --select=E,W,F
```

Expected: No errors related to moonshot provider files

- [ ] **Step 3: Run type checker**

```bash
uv run mypy consistency_checker/check/providers/moonshot.py
```

Expected: No type errors

- [ ] **Step 4: Verify build succeeds**

```bash
uv build
```

Expected: Wheel and sdist created without errors

- [ ] **Step 5: Final commit summary**

```bash
git log --oneline -5
```

Expected: Five new commits (one per task + this summary)

---

## Summary

After completing all tasks:
- ✅ `MoonshotJudgeProvider` and `MoonshotMultiPartyJudgeProvider` implemented
- ✅ Factory (`make_judge`, `make_multi_party_judge`) updated to support moonshot
- ✅ Config accepts `provider="moonshot"`
- ✅ Unit tests pass (mocked API)
- ✅ Live integration tests available (marked @pytest.mark.live)
- ✅ ADR documented decision
- ✅ Pipeline integration verified
- ✅ All tests pass, code type-checks, build succeeds

**Ready for testing:** Run a full consistency-check pipeline with `--provider moonshot` to evaluate Kimi's contradiction-detection quality on real data.
