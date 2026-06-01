# --max-cost ceiling + provider-aware pricing defaults

**Status**: planned
**Branch**: `feat/max-cost-ceiling`
**Date**: 2026-05-31
**Motivates**: commercial blocker #2 — hard cost ceiling that aborts a run before spending past N dollars; estimate-cost defaults to the actual provider's per-call pricing.

## Context

Today the `check` pipeline can spend an unbounded amount of money. The `estimate-cost` command projects spend with a fixed `$0.003-$0.010` per-call range (Anthropic/OpenAI tier), but:
- The default range silently miscalibrates for Moonshot/Kimi (the project's actual cheap-tier default since PR #60). Moonshot runs cost ~10-100× less per call; a user who reads the estimate at face value massively over-projects spend and may abort runs that would actually cost cents.
- There is no automatic ceiling. A user who passes `check` against a 50,000-assertion corpus with the definition detector engaged could rack up $50-$500 of judge calls without warning. The OOM-pre-flight (`max_memory_mb`) gates RAM but not money.

Two cases this PR fixes:
1. **Wrong default pricing** — Moonshot runs project at Anthropic/OpenAI prices, so users with `judge_provider: moonshot` see scary-but-fake totals.
2. **No hard ceiling** — `--max-cost <USD>` aborts a run BEFORE bootstrapping the judge if the (conservative, ceiling-style) pre-flight estimate exceeds the budget.

## Decision

- Add `Config.max_cost_usd: float | None = Field(default=None, ...)`. When set, `pipeline.check()` runs `estimate_cost()` as a pre-flight; if `est_cost_high > max_cost_usd`, it raises `CostCeilingExceeded(estimated_high=..., ceiling=...)` BEFORE the judge or NLI is initialised. The CLI catches it and exits 2 with a clean message.
- Add a `--max-cost <USD>` typer option to `check` (no `estimate-cost` flag — `estimate-cost` IS the estimator; there's no run to abort).
- Add provider-aware per-call cost defaults via a new `pipeline.default_per_call_costs(judge_provider) -> tuple[float, float]`:
  - `"anthropic"` and `"openai"`: `(0.003, 0.010)` — current defaults preserved.
  - `"moonshot"`: `(0.0001, 0.001)` — measured 10-100× cheaper than Anthropic.
  - `"fixture"`: `(0.0, 0.0)` — deterministic; no spend.
- CLI `estimate-cost` and `check`'s internal pre-flight both call `default_per_call_costs(cfg.judge_provider)` when the user did not override `--per-call-low`/`--per-call-high`. Override semantics unchanged.
- Why "ceiling-style" (high) for the gate: `estimate_cost`'s docstring says real spend is usually 30-70% lower than `est_cost_high`. Gating on the high end means false-positive aborts (a safe run rejected) but never false-negative (an over-budget run let through). That's the correct asymmetry for a budget guardrail.
- ADR-0016 documents the decision and the deferred work (per-call live tracking, OCR-time projection).

## Files

| File | Change |
|---|---|
| `consistency_checker/config.py` | Add `max_cost_usd: float \| None = Field(default=None, ge=0, ...)`. |
| `consistency_checker/pipeline.py` | New `CostCeilingExceeded` exception. New `default_per_call_costs(provider) -> tuple[float, float]`. `check()` runs pre-flight estimate against `cfg.max_cost_usd` when set. `estimate_cost()` defaults `per_call_low/high` from the provider when callers don't pass values explicitly. |
| `consistency_checker/cli/main.py` | `check`: add `--max-cost FLOAT` option; catch `CostCeilingExceeded` → `typer.echo(error)` + `raise typer.Exit(code=2)`. `estimate-cost`: leave the existing `--per-call-low`/`--per-call-high` options as overrides; when both are omitted, use provider defaults via `default_per_call_costs`. Update output formatting if the new defaults make the existing `:.3f` widths cramped (Moonshot lows are `$0.0001`, four decimals). |
| `docs/decisions/0016-max-cost-ceiling.md` | New ADR — context, decision, alternatives (live tracking, hard kill mid-run), consequences. |
| `tests/test_config.py` | `test_max_cost_usd_defaults_none`, `test_max_cost_usd_rejects_negative`. |
| `tests/test_pipeline.py` | `test_check_aborts_when_estimate_exceeds_max_cost` (raises `CostCeilingExceeded`, no judge init). `test_check_runs_when_estimate_under_max_cost`. `test_default_per_call_costs_moonshot` / `_anthropic` / `_fixture`. |
| `tests/test_estimate_cost.py` | `test_estimate_cost_uses_moonshot_defaults_when_provider_moonshot`. `test_estimate_cost_explicit_overrides_win`. |
| `tests/test_cli.py` | `test_check_max_cost_flag_aborts_when_exceeded`. `test_check_max_cost_flag_under_budget_runs`. `test_estimate_cost_moonshot_provider_uses_moonshot_defaults`. |
| `README.md` | Add `--max-cost` mention in CLI section; note that estimate-cost now defaults to the provider's pricing. |
| `CORPORATE_SETUP.md` | Note `--max-cost` as a budget guardrail in the operational hygiene section. |
| `config.example.yml` | Add `max_cost_usd: null` (commented) example with a one-line note about how to use it. |
| `futureplans.md` | Move blocker #2 to Completed with date + branch + ADR ref. |

## Tasks (TDD)

**Task 1 — Config field + ADR-0016**
Write `test_max_cost_usd_defaults_none` and `test_max_cost_usd_rejects_negative` first. Add the Field. Write ADR-0016 (context, decision, alternatives, consequences). Verify: `pytest -q tests/test_config.py`.

**Task 2 — Provider-aware defaults**
Write failing tests in `tests/test_pipeline.py`:
- `test_default_per_call_costs_anthropic` → `(0.003, 0.010)`
- `test_default_per_call_costs_openai` → `(0.003, 0.010)`
- `test_default_per_call_costs_moonshot` → `(0.0001, 0.001)`
- `test_default_per_call_costs_fixture` → `(0.0, 0.0)`
Add `default_per_call_costs(provider: JudgeProvider) -> tuple[float, float]` to `pipeline.py`. Update `estimate_cost()` so that when callers do NOT pass `per_call_low`/`per_call_high`, it uses `default_per_call_costs(config.judge_provider)`. Existing call sites that pass explicit values are unchanged. Verify: `pytest -q tests/test_pipeline.py tests/test_estimate_cost.py`.

**Task 3 — Pre-flight cost ceiling**
Write failing tests in `tests/test_pipeline.py`:
- `test_check_aborts_when_estimate_exceeds_max_cost` — config with `max_cost_usd=0.01`, fixture-mode store with assertions guaranteed to project > $0.01; `check()` raises `CostCeilingExceeded`. Critical: no NLI checker, no judge factory should be invoked. Use fixtures throughout — the pre-flight must run BEFORE bootstrap.
- `test_check_runs_when_estimate_under_max_cost` — same setup but `max_cost_usd=1000.0`; run completes normally.
- `test_check_raises_when_max_cost_zero_and_any_pairs_exist` — edge case: `max_cost_usd=0.0` with at least one definition pair → aborts.
Add `CostCeilingExceeded` exception and the pre-flight to `pipeline.check()`. Place the check AFTER the `pairwise_enabled + nli_checker` validation but BEFORE `nli_checker.release()` would run. Verify: `pytest -q tests/test_pipeline.py`.

**Task 4 — CLI wire-through**
Write failing tests in `tests/test_cli.py`:
- `test_check_max_cost_flag_aborts_when_exceeded` — `--max-cost 0.01`, fixture store with assertions; exit code 2; stderr has "exceeds" or "max-cost".
- `test_check_max_cost_flag_under_budget_runs` — `--max-cost 1000.0`; exit code 0.
- `test_estimate_cost_moonshot_provider_uses_moonshot_defaults` — config with `judge_provider: moonshot`, no per-call overrides; stdout contains `$0.0001` or similar Moonshot-tier number.
- `test_estimate_cost_explicit_overrides_win` — `--per-call-low 0.5 --per-call-high 0.6`; stdout reflects those numbers regardless of provider.
Add `--max-cost FLOAT` option to `check`. Catch `CostCeilingExceeded` and exit 2 with `"Estimated cost ${high:.2f} exceeds --max-cost ${ceiling:.2f}. Use --no-pairwise, narrow the corpus, or raise the ceiling."`. Apply provider-aware defaults in `estimate-cost` when overrides are omitted. Adjust format strings (`:.3f` → `:.4f` for the per-call display so Moonshot's `$0.0001` doesn't round to `$0.000`). Verify: `pytest -q tests/test_cli.py`.

**Task 5 — Docs sweep**
Update README, CORPORATE_SETUP, futureplans, config.example.yml. No code, no tests.

**Task 6 — Integration verification**
Full hermetic gate. Smoke: `consistency-check check --help` shows `--max-cost`. `consistency-check estimate-cost --corpus <existing>` with a Moonshot config shows sub-cent numbers.

## Non-goals

- Live per-call cost tracking with mid-run abort. v1 is pre-flight only. Live tracking would require every judge provider to return usage info; that's a bigger surface change that earns its own PR.
- OCR escalation time/cost projection. OCR is local compute (no $); the estimator works against a populated store where OCR has already run. A separate "what would ingest cost?" command could surface OCR time projections, but it's not the budget gate.
- Refactoring the `CostEstimate` dataclass to track per-provider breakdowns. The estimate is a single ceiling; the provider is recorded via the per-call defaults that produced it.

## Risks

- **Moonshot pricing drift**: pricing is hard-coded. Mitigation: ADR-0016 documents the source; if Moonshot raises prices, this is a one-line constant update.
- **Provider-default surprise**: a user who was reading `estimate-cost` with `judge_provider: moonshot` but seeing Anthropic numbers gets a 10-100× smaller number after this PR. That's the correct behavior; the docs sweep mentions it.
- **False-positive aborts on `--max-cost`**: the ceiling is conservative (`est_cost_high`). Users with tight budgets will hit it before they'd actually overspend. Acceptable — the workaround (`--no-pairwise`, narrower corpus, `--no-definitions`) is documented in the error message.
- **CI behavior**: hermetic tests use `judge_provider: fixture` (cost = 0); no provider-default test ever touches the network.

## Verification at end

```bash
uv run pytest -m "not slow and not live"
uv run ruff check .
uv run ruff format --check .
uv run mypy consistency_checker
uv build
```

Branch ready for `/review` + push when all four are green.
