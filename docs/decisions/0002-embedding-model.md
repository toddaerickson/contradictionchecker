# ADR 0002 — Embedding model

**Status**: Accepted

## Context

The assertion index (`index/assertion_store.py`) needs vector embeddings so the candidate-pair gate can retrieve top-k similar assertions across documents. Three classes of model were considered:

- **General-purpose sentence transformers** (e.g. `sentence-transformers/all-mpnet-base-v2`, 768-dim) — strong default with no domain assumption.
- **Domain-specific** (e.g. `yiyanghkust/finbert-tone`, `nlpaueb/legal-bert-base-uncased`) — incremental gains on in-domain text per the ContractNLI and FinBERT literature, but only worthwhile when the corpus is genuinely financial/legal.
- **Hosted embedding APIs** (OpenAI `text-embedding-3-*`, Cohere) — convenient but introduce another vendor dependency and per-call cost.

The MVP target is corpus-agnostic. We do not yet know what document type users will throw at it.

## Decision

Use **`sentence-transformers/all-mpnet-base-v2`** as the default embedding model. It is small enough to run locally without a GPU, has strong performance on STS / semantic-similarity benchmarks, and assumes nothing about corpus domain.

The embedder is configurable in `config.yml`. Domain models can be swapped in for users with known corpora without code changes.

## Consequences

- Adds `sentence-transformers` (and transitively `torch`) as a runtime dependency. Wheel size is non-trivial but manageable; CI caches it.
- 768-dimensional vectors → FAISS `IndexFlatIP` is fine up to ~1M assertions before IVF becomes necessary.
- No API key required to embed; entirely local. Air-gapped operation possible.
- Re-embedding after a model switch is supported via `consistency-check store rebuild-index` (the SQLite store is canonical; the FAISS index is derived).
