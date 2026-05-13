# Contributing

## Dev environment

```sh
uv sync
```

This creates `.venv/` and installs runtime + dev dependencies pinned in `uv.lock`. Python 3.11+ is required (pinned via `.python-version`).

## Local checks

Run these before opening a PR:

```sh
uv run pytest -q                       # unit tests (no network, no model downloads)
uv run ruff check .                    # lint
uv run ruff format --check .           # format check
uv run mypy consistency_checker        # strict type check
```

Slow / external tests are gated by markers and not run by default:

```sh
uv run pytest -m slow                  # downloads HF NLI model (~1.5GB; cached)
uv run pytest -m live                  # hits Anthropic / OpenAI APIs
uv run pytest -m e2e_fixture           # end-to-end with fixture judge (no network)
```

## Branching & PRs

Develop on a feature branch and open a PR against `main`. Squash-merge is the default. Branch protection settings may evolve as CI lands in Step 17 of the build plan.

PR titles should reference the build-plan step when applicable, e.g. `Step 5: Assertion schema + SQLite store`. PR bodies should include a verification block showing the commands run and their outputs.

## Adding a new module

The package layout in `consistency_checker/` is documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). When adding new functionality:

- Place tests alongside the module in `tests/test_<module>.py`.
- Fixtures go in `tests/fixtures/`.
- If a decision affects future code (model choice, provider abstraction, data layout), record it as an ADR in `docs/decisions/` following the existing template.
- New runtime dependencies: add to `pyproject.toml` `[project] dependencies`, then `uv lock`.

## Code style

- `ruff` enforces lint + format. Line length 100. Configured in `pyproject.toml`.
- `mypy --strict` is the contract. No untyped functions, no implicit `Any`.
- Default to writing no comments; only annotate non-obvious *why*. Identifiers should speak for themselves.
- Prefer dataclasses over dicts for structured data crossing module boundaries.

## Commit messages

Imperative subject, one blank line, body that explains *why* the change is being made (the *what* is in the diff). Keep subjects under 72 characters.
