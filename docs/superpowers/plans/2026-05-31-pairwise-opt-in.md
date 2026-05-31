# Pairwise contradiction detector → opt-in

**Status**: planned
**Branch**: `feat/pairwise-opt-in`
**Date**: 2026-05-31
**Motivates**: futureplans #3 — flip pairwise to opt-in (own eval 2026-05-21 showed near-zero useful yield on legal prose; the definition detector is the lever).

## Context

The check pipeline currently always runs two passes:

1. **Pairwise contradiction detector** — NLI gate (DeBERTa, ~800 MB download / ~600 MB RSS) → LLM judge on surviving pairs. Originally the centerpiece of the project.
2. **Definition-inconsistency detector** — term-grouped, bypasses NLI, fires the LLM judge on same-term assertion pairs. Added in ADR-0009.

Real-corpus measurement (2026-05-20, recorded in memory): on legal prose the pairwise pass produced **1 contradiction / 23 uncertain / 230 not_contra** from 254 NLI-flagged pairs — competent judge, anemic detector. The same corpus produces dozens of useful divergent-definition findings from the definition pass.

Cost of running pairwise by default:
- ~800 MB DeBERTa model download on first use.
- ~600 MB RSS while the model is resident.
- LLM judge calls on every NLI-flagged pair (the dominant cost item on corpora that *do* trigger many flags — common on technical docs where NLI happily mis-fires).
- Pre-flight `--max-memory-mb` gate exists precisely because of NLI's footprint.

A pairwise-by-default checker pays all of that cost on every run, on every corpus, even though the signal-to-noise on the corpora users actually feed it makes the pass net-negative. Flipping it to opt-in is the simplest change that recovers the user's time, memory, and money without removing the feature.

## Decision

- Add `Config.pairwise_enabled: bool = False`. Default off.
- Add `--pairwise/--no-pairwise` tri-state CLI flag on `check` and `estimate-cost` (mirrors `--ocr/--no-ocr`): omitted respects config, `--pairwise` forces on, `--no-pairwise` forces off.
- When pairwise is disabled, `pipeline.check()` skips the entire NLI-gate-and-judge loop AND the CLI/web entry points skip the NLI model load entirely (no download, no RSS, no `_preflight_memory` requirement).
- When pairwise is disabled, `pipeline.estimate_cost()` returns `n_candidate_pairs=0` (definition pairs still counted).
- The multi-party / triangle pass (`--deep`) keeps its current relationship to pairwise: it shares the strong-gate output. If a user passes `--deep` *without* `--pairwise`, the CLI rejects the combination — `--deep` requires pairwise gate output, so the combination is a config error. (Alternative considered: auto-enable pairwise when `--deep` is passed. Rejected as too implicit; users should know they're paying for NLI.)
- Document deprecation-by-default in a new ADR-0015. Don't remove the code path — the eval is corpus-specific and a user with a heavily numeric/specs corpus may want it on.

## Files

| File | Change |
|---|---|
| `consistency_checker/config.py` | Add `pairwise_enabled: bool = False` Field with description. |
| `consistency_checker/pipeline.py` | `check()` signature: `nli_checker: NliChecker \| None = None`. Skip pairwise loop + `nli_checker.release()` when `config.pairwise_enabled is False`. `estimate_cost()` gains the same gate — skip the candidate-pair scan when pairwise is off. |
| `consistency_checker/cli/main.py` | Add tri-state `--pairwise/--no-pairwise` to `check` and `estimate-cost`. Apply override before `cfg.pairwise_enabled` is read. Only load `TransformerNliChecker` when `cfg.pairwise_enabled` is True. Only run `_preflight_memory` when pairwise is on. Reject `--deep --no-pairwise` (or `--deep` with config-off) with a clear typer error. Adjust summary line so "gated/judged" suffixes only print when pairwise ran. |
| `consistency_checker/web/app.py` | `_run_check_in_background`: skip NLI construction when `config.pairwise_enabled` is False; pass `nli_checker=None` to `run_check`. Web form already has `deep`; if a future iteration adds a pairwise checkbox it lives here, but not required now (web defaults to config). |
| `consistency_checker/check/nli_checker.py` | No code change, but verify `FixtureNliChecker` is still importable from tests. |
| `tests/test_config.py` | Assert `Config(corpus_dir=...).pairwise_enabled is False`. |
| `tests/test_cli.py` | Tri-state tests for `check --pairwise / --no-pairwise / omitted`. Reject `--deep --no-pairwise`. `estimate-cost --no-pairwise` returns n_candidate_pairs=0. |
| `tests/test_pipeline.py` (or wherever `check()` is unit-tested) | Pipeline skips pairwise loop when `pairwise_enabled=False`; can pass `nli_checker=None`; definition pass still runs. |
| `docs/decisions/0015-pairwise-opt-in.md` | New ADR — eval data, decision, consequences, when to flip it back on. |
| `README.md` | Update "what runs by default" / install section — note NLI model is now downloaded only when `--pairwise` is used. |
| `futureplans.md` | Move item #3 (pairwise opt-in) to Completed section with this date. |
| `CORPORATE_SETUP.md` | If it mentions the NLI download as required, soften to "downloaded only when pairwise detector is enabled." |

## Tasks (TDD)

**Task 1 — Config field + ADR-0015**
Write `test_config_pairwise_enabled_default_false` first. Add the Field. Write ADR-0015 (eval data, decision, consequences). Verify: `pytest -q tests/test_config.py`.

**Task 2 — Pipeline gate**
Write failing tests:
- `test_check_skips_pairwise_when_disabled` — pass `nli_checker=None`, `config.pairwise_enabled=False`, assert n_pairs_gated=0, n_pairs_judged=0, definition findings still recorded.
- `test_estimate_cost_zero_candidate_pairs_when_pairwise_disabled` — config off → `result.n_candidate_pairs == 0`, definition_pairs unchanged.
Make them pass: change `pipeline.check()` and `pipeline.estimate_cost()` signatures/behavior. Verify: `pytest -q tests/test_pipeline.py`.

**Task 3 — CLI tri-state + lazy NLI load**
Write failing tests:
- `test_check_no_pairwise_flag_disables_pairwise` (config defaults; verify nli_checker not constructed, e.g. monkeypatch `TransformerNliChecker` to raise if instantiated).
- `test_check_pairwise_flag_overrides_config_off`.
- `test_check_omitted_respects_config` (set `pairwise_enabled=True` via env override; pairwise runs).
- `test_check_deep_without_pairwise_rejected` (typer should exit non-zero with helpful message).
- `test_estimate_cost_respects_pairwise_flag`.
Make them pass: add the typer option (mirroring `--ocr/--no-ocr`), gate the `TransformerNliChecker` construction, gate `_preflight_memory`, validate `--deep` combination. Verify: `pytest -q tests/test_cli.py`.

**Task 4 — Web background task**
Write failing test `test_web_run_check_skips_nli_when_pairwise_disabled` (mock the background path's NLI construction site). Make it pass: branch on `config.pairwise_enabled` in `_run_check_in_background`. Verify: `pytest -q tests/test_web.py` (or whichever test module owns the run-trigger surface).

**Task 5 — Docs sweep**
Update README "Default behavior" subsection, CORPORATE_SETUP NLI install note, futureplans Completed entry. No code, no tests required — but `mypy` + `ruff` must still pass.

**Task 6 — Integration verification**
Run the full hermetic gate (`uv run pytest -m "not slow and not live"`), then a tiny smoke: `uv run consistency-check check --help` shows `--pairwise/--no-pairwise`; `uv run consistency-check estimate-cost --no-pairwise --corpus <existing>` returns zero candidate pairs. Don't run a real `check` (no network/no NLI download required — that's the point).

## Non-goals

- Removing the pairwise code path. It stays as opt-in.
- Adding a hard cost ceiling. That's blocker #2, separate PR.
- Changing the definition detector. It already runs by default and is the headline value.
- Changing `--deep`'s behavior, except to require `--pairwise` when used.

## Risks

- **Existing config files without `pairwise_enabled`** — Pydantic `extra="forbid"` doesn't bite (this is a new field with a default). Existing `config.yml` files keep working; users get the new default automatically.
- **Web UI silent flip** — A web user who relied on pairwise findings will stop seeing them. Mitigation: ADR-0015 mentions this; if web becomes the dominant entry point, add a checkbox in a follow-up PR.
- **CI** — Hermetic tests don't actually load the NLI model (they use `FixtureNliChecker` / `HashEmbedder`), so the default flip should not change CI runtime measurably.
- **Cost-estimate accuracy** — A user who flips `--pairwise` back on and runs `estimate-cost --no-pairwise` will get an underestimate. Mitigation: the flag is opt-in symmetric (the same flag they used for the real run), so this is unlikely in practice.

## Verification at end

```bash
uv run pytest -m "not slow and not live"
uv run ruff check .
uv run ruff format --check .
uv run mypy consistency_checker
uv build
```

All four must stay green. Branch is ready for `/review` + push when they do.
