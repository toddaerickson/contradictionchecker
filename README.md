# contradictionchecker

[![CI](https://github.com/toddaerickson/contradictionchecker/actions/workflows/ci.yml/badge.svg)](https://github.com/toddaerickson/contradictionchecker/actions/workflows/ci.yml)

Scans a corpus of documents for **internal contradictions** and **divergent definitions** of the same term. Designed for symmetric scans across a static corpus, not for guarding an existing knowledge base against new documents. The definition-inconsistency detector is on by default; the pairwise NLI + LLM detector is opt-in via `--pairwise` (ADR-0015).

> v0.3 — FastAPI + HTMX web UI, three-document conditional contradictions (graph triangles), numeric short-circuit, and PDF/DOCX loaders. See [`CHANGELOG.md`](CHANGELOG.md) for the full feature list, [`futureplans.md`](futureplans.md) for what's next.

## What runs by default

The **definition-inconsistency detector** runs by default: it groups assertions by canonical term and asks the LLM judge whether same-term definitions diverge across the corpus. This is the detector that carries the headline value on every corpus shape we've measured.

The **pairwise contradiction detector** (NLI gate → LLM judge on candidate pairs) is **off by default** as of ADR-0015 — own-corpus eval on legal prose showed near-zero useful yield at high compute cost. Enable it per run with `--pairwise`, or set `pairwise_enabled: true` in `config.yml`. When pairwise is enabled, the two-stage flow is:

| Stage | Component | Role |
|------|-----------|------|
| A | NLI checker (`microsoft/deberta-v3-large-mnli` family) | Cheap bidirectional contradiction score. Gates candidate pairs to Stage B. |
| B | LLM judge (Anthropic Claude or OpenAI, structured JSON output) | Verifies with rationale and evidence spans. |

A single LLM-only check has been benchmarked at ~16% precision on pairwise contradiction detection in domain text (legal/financial). The NLI gate lifts precision to ~89% while cutting LLM cost roughly an order of magnitude — so when pairwise *is* the right detector (numeric-/spec-heavy corpora), the two-stage flow is the cheap way to run it.

See [`docs/decisions/0015-pairwise-opt-in.md`](docs/decisions/0015-pairwise-opt-in.md) for the rationale, and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full module breakdown.

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

> **Corporate / sensitive-data users:** read [`CORPORATE_SETUP.md`](CORPORATE_SETUP.md) **first**. This tool sends every document chunk to a third-party LLM API — confirm that's allowed by your data-classification policy before running it.

## Quickstart

```sh
# 1. Copy and edit config
cp config.example.yml config.yml
# Set corpus_dir, choose judge_provider (anthropic | openai). The API key
# is read from the environment, NEVER from config.yml — config.yml is
# safe to commit; your key is not.

# 2. Set credentials for whichever provider you chose
# Prefer your corporate secret manager or a .gitignore'd .env file over
# baking the key into your shell rc. See CORPORATE_SETUP.md §3.
export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY=...

# 3a. Web UI flow (v0.3+)
uv run consistency-check serve --open    # browser opens to http://127.0.0.1:8000
# Drop files in the Ingest tab → click Run / Check now (toggle Deep for
# three-document conditional contradictions) → watch live counters on
# the Stats tab → drill into each finding from the Contradictions tab.

# 3b. CLI-only flow
uv run consistency-check ingest path/to/corpus/
uv run consistency-check estimate-cost               # rough API-spend ceiling before you commit; per-call defaults now follow your configured judge_provider (Moonshot/Kimi projects sub-cent — ~$0.0001–$0.001 per call — vs Anthropic/OpenAI ~$0.003–$0.010)
uv run consistency-check check                       # add --pairwise for the NLI gate (off by default — see ADR-0015); --deep for triangle pass (requires --pairwise); --no-definitions to skip the definition stage; --max-cost <USD> aborts before judge bootstrap when the projection exceeds the ceiling (ADR-0016)
uv run consistency-check report                      # writes data/store/reports/cc_report_<ts>_<run_id>.md
uv run consistency-check export csv                  # writes data/store/reports/cc_assertions_<ts>.csv
```

The `export` command emits `(doc_id, assertion_id, assertion_text)` rows for downstream tooling. `--out` is optional for both `report` and `export`; omit it and the file lands under `<data_dir>/reports/` with a unique descriptive name. The first `check --pairwise` run downloads a ~440 MB DeBERTa-base NLI model from Hugging Face (or ~1.5 GB if you opted up to `DeBERTa-v3-large` via `nli_model` in config); subsequent runs hit the cache. Default `check` runs (definition detector only) do **not** download or load the NLI model — same one-line download warning pattern as the OCR fallback (ADR-0014).

### Vendoring HTMX

The web UI ships with a placeholder `htmx.min.js`. After cloning, run once:

```sh
uv run python scripts/vendor_htmx.py
```

to download HTMX v1.9.12 into `consistency_checker/web/static/`. Tests use FastAPI's `TestClient` which doesn't execute JS, so CI doesn't need the real script.

### Other CLI commands

```sh
uv run consistency-check serve --host 127.0.0.1 --port 8000  # launch the web UI
uv run consistency-check store stats                          # row counts
uv run consistency-check store rebuild-index                  # regenerate FAISS from SQLite
uv run consistency-check --help                               # all commands
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
- [0004 — PDF/DOCX loader backend](docs/decisions/0004-pdf-docx-loaders.md): `unstructured` for both.
- [0005 — Numeric short-circuit before the LLM judge](docs/decisions/0005-numeric-short-circuit.md): deterministic sign-flip detector skips the judge.
- [0006 — Three-document conditional contradictions](docs/decisions/0006-three-doc-conditional.md): graph triangles on FAISS-similarity edges, opt-in via `--deep`.
- [0007 — Web UI](docs/decisions/0007-web-ui.md): FastAPI + HTMX, server-rendered, `cc_`-prefixed templates.
- [0015 — Pairwise contradiction detector becomes opt-in](docs/decisions/0015-pairwise-opt-in.md): pairwise off by default; enable per run with `--pairwise` or `pairwise_enabled: true`.
- [0016 — Pre-flight cost ceiling for `check`](docs/decisions/0016-max-cost-ceiling.md): `--max-cost <USD>` / `max_cost_usd` aborts before judge bootstrap when the conservative projection exceeds the ceiling; `estimate-cost` defaults per-call costs from the configured `judge_provider` (Moonshot ~10–100× cheaper than Anthropic/OpenAI).

## Supported formats

| Extension      | Loader                                        | Notes |
|----------------|-----------------------------------------------|-------|
| `.txt`, `.md`  | built-in plaintext loader                     | char spans round-trip exactly |
| `.pdf`, `.docx`| `unstructured` (`strategy="fast"`)             | body-content elements only; sidecar `element_spans` in `documents.metadata_json` |

> Scanned-image PDFs are auto-escalated to `unstructured`'s hi_res (OCR) strategy when fast extraction returns near-empty text. First OCR run downloads ~500 MB of layout + OCR models. Requires system Tesseract (`apt install tesseract-ocr` on Debian/Ubuntu, `brew install tesseract` on macOS).

Other formats can be added via the `LOADERS` registry in `consistency_checker/corpus/loader.py`.

## Known limitations

Carried forward into the v0.4+ roadmap in [`futureplans.md`](futureplans.md):

- Chunk overlap `> 0` is unimplemented.
- Three-document detection misses triangles whose edges fall below the FAISS gate threshold; v0.4 #6 adds an entity-NER cluster pass to catch these. Three-document detection requires `--pairwise` since it shares the NLI gate's output.
- First `check --pairwise` run downloads ~440 MB for the default NLI model (DeBERTa-v3-base). Switch to DeBERTa-v3-large via `nli_model` in config for higher recall at ~1.5 GB. Default `check` runs (pairwise off, see ADR-0015) skip the download entirely.
- OCR fallback is automatic for image-only PDFs (`--no-ocr` to disable); first use downloads ~500 MB and requires system Tesseract.
- `data_dir/uploads/<upload_id>/` grows without bound; v0.4 will add a GC pass.
- The web UI is single-user, localhost-only, with **no authentication or CSRF protection**. `serve` refuses to bind to a non-loopback host unless you pass `--unsafe-no-auth`; doing so exposes an **unauthenticated** file-upload and corpus-mutation surface to anyone who can reach the host, so only use it on a trusted, isolated network.

## License

MIT. See [`LICENSE`](LICENSE).
