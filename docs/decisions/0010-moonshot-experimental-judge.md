# ADR-0010: Add Moonshot (Kimi) as Experimental Judge Provider

**Date:** 2026-05-16  
**Status:** Accepted

## Context

The consistency-checker pipeline currently supports two judge providers:
- Anthropic Claude (via tool-use structured output)
- OpenAI GPT-4 (via JSON schema structured output)

Moonshot AI's Kimi model is a strong alternative for contradiction detection, with potential cost advantages ($0.60/$2.50 per M tokens vs. Anthropic's higher rates). Before committing to Kimi as a production alternative (future ADR will address this), we want to test it experimentally and compare its contradiction-detection quality.

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
- Foundation for future cost-optimization work

### Negative

- Adds dependency on Moonshot's API stability
- Requires MOONSHOT_API_KEY env var setup (documented in .env setup)
- Integration tests require live API access (not hermetic)

## Related

- **Future ADR:** Cost-optimization via provider selection based on contradiction complexity
- **Task:** Kimi CLI extension auth setup (2026-05-16) — provides .env configuration reused here
- **Spec:** 2026-05-16-kimi-moonshot-judge-provider.md

## Notes for Future Reviewers

If Kimi's quality is insufficient (lower contradiction detection accuracy), deprecate this provider and remove it. If quality is comparable, proceed with future work to make it a cost-optimization option (use Kimi for simple pairs, Anthropic for complex).
