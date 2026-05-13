# ADR 0001 — LLM judge provider

**Status**: Accepted

## Context

Stage B of the contradiction pipeline (`check/llm_judge.py`) sends candidate pairs that survive the NLI gate to a large language model for structured verification. The judge's output (`verdict / rationale / confidence / evidence_spans`) is load-bearing for downstream report quality, so we need:

- Strict, schema-validated JSON output (the prior art at `datarootsio/knowledgebase_guardian` failed by parsing `.startswith("CONSISTENT")`).
- Reasonably priced inference, since gating still produces O(corpus) candidate pairs.
- Avoidance of vendor lock-in early in the project.

## Decision

Support **both Anthropic Claude and OpenAI** behind a `Judge` Protocol. Both providers expose first-class structured-output paths (Anthropic via tool-use, OpenAI via JSON-mode / structured outputs) — using both costs roughly a half-day extra versus picking one, in exchange for A/B comparability and vendor flexibility.

The interface lives in `consistency_checker/check/providers/base.py`. Concrete implementations: `providers/anthropic.py`, `providers/openai.py`. A `FixtureJudge` for hermetic tests rounds out the trio.

Default provider for new installs: Anthropic Claude (Sonnet for accuracy, Haiku for cheap regression runs). Configurable per `config.yml`.

## Consequences

- One extra SDK dependency (`anthropic` + `openai`), both lightweight.
- Tests must validate that both providers return schema-conforming output. A canonical hand-built contradiction pair is sent to each under the `live` pytest mark.
- Prompt files (`check/prompts/judge_*.txt`) must be written provider-neutrally; provider-specific argument shaping happens in each provider implementation, not in the prompt text.
- If a third provider is added later (Google, local llama.cpp), it is additive — implement the Protocol and register.
