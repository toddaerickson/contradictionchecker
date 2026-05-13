# contradictionchecker

Detects internal contradictions across a corpus of documents using a two-stage NLI + LLM pipeline.

> Status: under active construction. See [`/root/.claude/plans/propose-build-steps-start-lovely-eagle.md`](/root/.claude/plans/propose-build-steps-start-lovely-eagle.md) (when present) or the architecture decisions below for the build plan.

## Architecture decisions

Key decisions are recorded as ADRs in [`docs/decisions/`](docs/decisions/README.md):

- [0001 — LLM judge provider](docs/decisions/0001-llm-judge-provider.md)
- [0002 — Embedding model](docs/decisions/0002-embedding-model.md)
- [0003 — CONTRADOC benchmark integration timing](docs/decisions/0003-contradoc-integration.md)

## Development

```sh
uv sync
uv run pytest -q
uv run ruff check .
uv run mypy consistency_checker
```
