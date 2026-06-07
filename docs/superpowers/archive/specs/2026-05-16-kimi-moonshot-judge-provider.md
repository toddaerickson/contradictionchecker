# Kimi (Moonshot) Judge Provider — Design Spec

**Date:** 2026-05-16  
**Goal:** Add Kimi (Moonshot AI) as an experimental judge provider, reusing the OpenAI SDK pattern with Moonshot's API endpoint.

**Scope:** Pairwise contradiction detection + multi-party (triangle) contradiction detection, matching the Anthropic/OpenAI provider feature set.

---

## Context

The consistency-checker pipeline currently supports two judge providers:
- **Anthropic:** Uses tool-use to enforce JSON output
- **OpenAI:** Uses JSON schema mode (`response_format`)

Both providers implement `JudgeProvider` and `MultiPartyJudgeProvider` protocols, returning validated `JudgePayload` and `MultiPartyJudgePayload` objects.

Moonshot AI's Kimi API is OpenAI-compatible: same REST structure, same JSON schema support, same model interaction pattern. This allows us to reuse the OpenAI SDK (`openai.OpenAI`) with only the base URL and API key configuration changed.

---

## Architecture

### Provider Implementation

**File:** `consistency_checker/check/providers/moonshot.py`

Create two classes:

1. **`MoonshotJudgeProvider`** — pairwise contradiction judge
   - Instantiate `openai.OpenAI(api_key=MOONSHOT_API_KEY, base_url="https://api.moonshot.ai/v1")`
   - Call `client.beta.chat.completions.parse()` with `response_format=JudgePayload` (OpenAI's structured output)
   - Return the parsed `JudgePayload` from the response

2. **`MoonshotMultiPartyJudgeProvider`** — three-way (triangle) judge
   - Same setup; call with `response_format=MultiPartyJudgePayload`
   - Return validated `MultiPartyJudgePayload`

Both classes implement the respective Protocol interfaces (duck-typing; no explicit inheritance needed).

### Configuration

**Where:** `consistency_checker/config.py`

Add a new provider option:
```python
provider: Literal["anthropic", "openai", "moonshot", ...] = "anthropic"
```

For experimental tracking, initially accept `"moonshot"` as a valid provider. No feature flag needed; the choice is explicit in config.

### Factory Pattern

**Where:** `consistency_checker/pipeline.py`, in the `make_judge()` and `make_multi_party_judge()` functions

Add branches:
```python
elif config.provider == "moonshot":
    return MoonshotJudgeProvider(
        api_key=get_env("MOONSHOT_API_KEY"),
        model="kimi-k2.6"  # or configurable
    )
```

### API Key Management

**Storage:** Use the `.env` file we created for Kimi CLI setup (from the Kimi extension auth setup task).

**Env var:** `MOONSHOT_API_KEY` — read from shell environment or `.env`

The `.env` file already exists at `/home/terickson/contradictionchecker/.env` and is gitignored, so secrets are safe.

---

## Data Flow

1. **Pipeline requests a judge:**
   ```python
   judge = make_judge(config)  # config.provider == "moonshot"
   ```

2. **Factory instantiates MoonshotJudgeProvider:**
   ```python
   provider = MoonshotJudgeProvider(api_key="sk-...", model="kimi-k2.6")
   ```

3. **Pipeline calls the judge:**
   ```python
   payload = judge.request_payload(system_prompt, user_prompt)
   ```

4. **Provider sends to Moonshot API:**
   - Endpoint: `POST https://api.moonshot.ai/v1/chat/completions`
   - Model: `kimi-k2.6` (configurable)
   - Messages: system + user
   - `response_format`: `JudgePayload` (structured output schema)

5. **Moonshot returns JSON-structured response:**
   - Kimi outputs a JSON object matching `JudgePayload` schema
   - No tool-use wrapper (unlike Anthropic); direct JSON response

6. **Provider validates and returns:**
   ```python
   return JudgePayload.model_validate(response)
   ```

---

## API Endpoint & Credentials

**Base URL:** `https://api.moonshot.ai/v1`

**Model:** `kimi-k2.6` (latest as of 2026-05-16; configurable in future)

**Authentication:** Bearer token in `Authorization` header
- Read from `MOONSHOT_API_KEY` env var
- Expected format: `sk-...` (Moonshot tokens start with `sk-`)

**Pricing (reference):**
- Input: $0.60 per million tokens
- Output: $2.50 per million tokens
- (Cheaper than Anthropic for testing; may vary by date)

---

## Testing Strategy

### Unit Tests

**File:** `tests/check/providers/test_moonshot.py`

1. **Mock Moonshot API responses** — no live API calls in default test suite
2. **Test pairwise judge:**
   - Mock `openai.OpenAI.beta.chat.completions.parse()` to return a valid response
   - Verify `MoonshotJudgeProvider.request_payload()` parses and returns `JudgePayload`
   - Test edge cases: malformed response, missing fields, confidence out of range

3. **Test multi-party judge:**
   - Same pattern with `MultiPartyJudgePayload`
   - Verify `contradicting_subset` validation (must be empty when verdict != "multi_party_contradiction")

4. **Test error handling:**
   - Missing API key → raises clear error
   - API rate limit → propagates exception
   - Invalid JSON response → `ValidationError` raised

### Integration Tests

**Mark:** `@pytest.mark.live` (requires `MOONSHOT_API_KEY` env var)

1. **Real API call (pairwise):**
   - Send a real prompt to Moonshot's API
   - Verify response structure matches `JudgePayload`
   - Assert verdict is one of `["contradiction", "not_contradiction", "uncertain"]`

2. **Real API call (multi-party):**
   - Send a three-assertion prompt
   - Verify response structure and verdict constraints

### Fixture Tests (Hermetic)

**Use case:** `FixtureJudge` (from existing test infrastructure) to keep tests hermetic

- Add `MoonshotFixtureJudge` to `conftest.py` if needed (but Moonshot may not need a fixture variant if we mock effectively)

---

## Error Handling

1. **Missing API key:**
   - Raise `ValueError("MOONSHOT_API_KEY not set in environment or .env")`
   - Provide clear setup instructions (link to .env setup doc)

2. **API errors (rate limit, auth failure, server error):**
   - Let `openai.APIError` and subclasses propagate
   - Caller (pipeline) decides retry logic (existing error handling covers this)

3. **Response validation failure:**
   - `JudgePayload.model_validate()` raises `ValidationError` if response doesn't match schema
   - Caller logs and decides whether to retry or skip the pair

---

## Configuration & Discovery

### Config Fields

In `consistency_checker/config.py`:

```python
class CheckConfig(BaseModel):
    provider: Literal["anthropic", "openai", "moonshot"] = "anthropic"
    # ... existing fields
```

Users can set via:
- CLI: `consistency-check check --provider moonshot`
- Env var: `CC_PROVIDER=moonshot`
- Config file: `provider: moonshot` in YAML
- Python API: `config.provider = "moonshot"`

### API Key Configuration

Users set `MOONSHOT_API_KEY` via:
1. Shell export: `export MOONSHOT_API_KEY="sk-..."`
2. `.env` file: `MOONSHOT_API_KEY=sk-...` (already gitignored)
3. Direct env var in CI/deployment

The `.env` file created in the Kimi extension auth setup task can be reused here.

---

## Files to Create/Modify

| File | Action | Responsibility |
|------|--------|-----------------|
| `consistency_checker/check/providers/moonshot.py` | Create | `MoonshotJudgeProvider`, `MoonshotMultiPartyJudgeProvider` classes |
| `consistency_checker/pipeline.py` | Modify | Add `moonshot` case to `make_judge()` and `make_multi_party_judge()` factories |
| `consistency_checker/config.py` | Modify | Add `"moonshot"` to `provider` Literal and document the option |
| `tests/check/providers/test_moonshot.py` | Create | Unit tests (mocked) + integration test (`@pytest.mark.live`) |
| `docs/decisions/ADR-0007-moonshot-experimental-judge.md` | Create | Document the decision to add Moonshot as an experimental provider |
| `.env` | Already exists | Reuse for `MOONSHOT_API_KEY` (created in auth setup task) |

---

## Dependencies

**New:** None. Reuse existing:
- `openai` (already a dependency for the OpenAI provider)
- `pydantic` (already a dependency for validation)

No additional pip packages needed.

---

## Success Criteria

- [ ] `MoonshotJudgeProvider` and `MoonshotMultiPartyJudgeProvider` instantiate without errors
- [ ] Unit tests pass with mocked API responses
- [ ] Integration test (live) passes with real Moonshot API
- [ ] Both pairwise and multi-party verdicts validate correctly
- [ ] Config accepts `provider="moonshot"` and factory creates the correct provider instance
- [ ] API key is read from `MOONSHOT_API_KEY` env var or `.env` file
- [ ] Error handling is clear (e.g., missing API key shows helpful message)
- [ ] Pipeline can run a full check with Moonshot as the judge without errors
- [ ] Contradictions detected by Moonshot judge are logged and reportable

---

## Limitations & Future Work

- **Model selection:** Hardcoded to `kimi-k2.6` for now. Can be made configurable later (e.g., `judge_model: "kimi-k2.6"` in config).
- **Definition judge:** Not included in this spec. Can be added in a follow-up if needed (same pattern as pairwise/multi-party).
- **Batch mode:** Moonshot API doesn't yet have a batch endpoint like OpenAI. Calls are individual.
- **Cancellation:** No support for cancelling in-flight requests (consistent with other providers).
- **Streaming:** Not supported; all responses are complete JSON objects (consistent with structured output requirement).

---

## Rollback Plan

If Moonshot provider has issues:
1. Set `provider: "anthropic"` or `"openai"` in config to revert to a tested provider
2. Delete `consistency_checker/check/providers/moonshot.py` and related test file
3. Revert changes to `pipeline.py` and `config.py`
4. Keep the `.env` setup (it's not harmful, just unused)

The pairwise and multi-party test suites for other providers remain unaffected.

---

## References

- [Moonshot AI API Docs](https://platform.moonshot.ai/)
- [OpenAI Python SDK (Structured Output)](https://platform.openai.com/docs/guides/structured-outputs)
- [Kimi K2.6 Model Info](https://kimi-app.com/api/)
