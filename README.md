# contradictionchecker

[![CI](https://github.com/toddaerickson/contradictionchecker/actions/workflows/ci.yml/badge.svg)](https://github.com/toddaerickson/contradictionchecker/actions/workflows/ci.yml)

Scans a corpus of documents for **internal contradictions** using a two-stage NLI + LLM pipeline. Designed for symmetric pairwise scans across a static corpus, not for guarding an existing knowledge base against new documents.

> v0.1.0 — first release. See [`CHANGELOG.md`](CHANGELOG.md) for what's in scope, [`futureplans.md`](futureplans.md) for what's next.

## Why two stages

A single LLM-only check has been benchmarked at ~16% precision on pairwise contradiction detection in domain text (legal/financial). Adding an NLI gate (DeBERTa-class model) in front of the LLM judge lifts precision to ~89% while cutting LLM cost roughly an order of magnitude.

| Stage | Component | Role |
|------|-----------|------|
| A | NLI checker (`microsoft/deberta-v3-large-mnli` family) | Cheap bidirectional contradiction score. Gates candidate pairs to Stage B. |
| B | LLM judge (Anthropic Claude or OpenAI, structured JSON output) | Verifies with rationale, confidence, and evidence spans. |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full module breakdown.

## Install

From PyPI (once published):

```sh
pip install consistency-checker
```

From source:

```sh
git clone https://github.com/toddaerickson/contradictionchecker
cd contradictionchecker
uv sync
```

## Quickstart

```sh
# 1. Copy and edit config
cp config.example.yml config.yml
# Set corpus_dir, choose judge_provider (anthropic | openai), API keys via env.

# 2. Set credentials for whichever provider you chose
export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY=...

# 3. Run
uv run consistency-check ingest path/to/corpus/
uv run consistency-check check
uv run consistency-check report --out report.md
uv run consistency-check export csv --out assertions.csv
```

The `export` command emits `(doc_id, assertion_id, assertion_text)` rows for downstream tooling. The first `check` run downloads a ~800 MB DeBERTa NLI model from Hugging Face; subsequent runs hit the cache.

### Other CLI commands

```sh
uv run consistency-check store stats             # row counts
uv run consistency-check store rebuild-index     # regenerate FAISS from SQLite
uv run consistency-check --help                  # all commands
```

## Benchmarks

`benchmarks/contradoc_harness.py` runs Stage A + Stage B against a normalised CONTRADOC dataset and reports precision / recall / F1. The dataset is not redistributed; see [`docs/benchmarks.md`](docs/benchmarks.md) for the input format and runbook.

```sh
uv run python -m benchmarks.contradoc_harness \
    --input contradoc.jsonl --output metrics.json --sample 50
```

## Development

```sh
uv sync
uv run pytest -m "not slow and not live"   # default CI gate
uv run pytest -m slow                      # downloads HF models (~800 MB - 1.5 GB)
uv run pytest -m live                      # hits Anthropic / OpenAI APIs
uv run ruff check .
uv run ruff format --check .
uv run mypy consistency_checker
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for branching, PR conventions, and the dev loop.

## Architecture decisions

Recorded as ADRs in [`docs/decisions/`](docs/decisions/README.md):

- [0001 — LLM judge provider](docs/decisions/0001-llm-judge-provider.md): both Anthropic and OpenAI, behind a `Judge` Protocol.
- [0002 — Embedding model](docs/decisions/0002-embedding-model.md): `sentence-transformers/all-mpnet-base-v2` default.
- [0003 — CONTRADOC integration timing](docs/decisions/0003-contradoc-integration.md): in MVP scope.

## Known limitations

Carried forward into the v0.2 roadmap in [`futureplans.md`](futureplans.md):

- PDF / DOCX loaders are stubbed (raise `NotImplementedError`).
- Chunk overlap `> 0` is unimplemented.
- No dedicated numeric/quantitative extractor.
- Three-document conditional contradictions (pairwise checks pass, the conjunction contradicts) out of scope.
- First `check` run downloads ~800 MB for the NLI model.

## License

MIT. See [`LICENSE`](LICENSE).
