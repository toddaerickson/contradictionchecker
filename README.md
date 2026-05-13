# contradictionchecker

Scans a corpus of documents for **internal contradictions** using a two-stage NLI + LLM pipeline. Designed for symmetric pairwise scans across a static corpus, not for guarding an existing knowledge base against new documents.

> Status: under construction. Modules land one PR at a time per the build plan; this README and the [`docs/`](docs/) tree are kept current as the surface area grows.

## Why two stages

A single LLM-only check has been benchmarked at ~16% precision on pairwise contradiction detection in domain text (legal/financial). Adding an NLI gate (DeBERTa-class model) in front of the LLM judge lifts precision to ~89% while cutting LLM cost roughly an order of magnitude.

| Stage | Component | Role |
|------|-----------|------|
| A | NLI checker (`microsoft/deberta-v3-large-mnli` family) | Cheap bidirectional contradiction score. Gates candidate pairs to Stage B. |
| B | LLM judge (Anthropic Claude or OpenAI, structured JSON output) | Verifies with rationale, confidence, and evidence spans. |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full module breakdown.

## Quickstart

Not yet wired end-to-end — the CLI lands at Step 14 of the build plan. Once available:

```sh
uv sync
uv run consistency-check ingest path/to/corpus/
uv run consistency-check check
uv run consistency-check report --out report.md
uv run consistency-check export csv --out assertions.csv
```

The `export` command emits `(doc_id, assertion_id, assertion_text)` rows for downstream tooling.

## Development

```sh
uv sync
uv run pytest -q                 # unit tests
uv run pytest -m slow            # tests that download HF models (~1.5GB)
uv run pytest -m live            # tests that hit Anthropic / OpenAI APIs
uv run ruff check .
uv run mypy consistency_checker
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the dev loop and PR conventions.

## Architecture decisions

Recorded as ADRs in [`docs/decisions/`](docs/decisions/README.md):

- [0001 — LLM judge provider](docs/decisions/0001-llm-judge-provider.md): both Anthropic and OpenAI, behind a `Judge` Protocol.
- [0002 — Embedding model](docs/decisions/0002-embedding-model.md): `sentence-transformers/all-mpnet-base-v2` default.
- [0003 — CONTRADOC integration timing](docs/decisions/0003-contradoc-integration.md): in MVP scope.

## License

MIT. See [`LICENSE`](LICENSE).
