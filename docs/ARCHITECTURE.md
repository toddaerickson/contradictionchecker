# Architecture

## Goal

Scan a corpus of documents for **internal contradictions** — assertions in one document that conflict with assertions in another. Symmetric pairwise scan, not asymmetric KB-guard.

## Module map

```
consistency_checker/
├── corpus/          loader.py        load .txt / .md (PDFs/Docx stubbed)
│                    chunker.py       sentence-window chunks with char offsets
├── extract/         schema.py        Document & Assertion dataclasses
│                    atomic_facts.py  LLM decomposition into atomic claims
├── index/           assertion_store.py  SQLite — canonical store, exportable
│                    embedder.py         sentence-transformers wrapper
│                    faiss_store.py      FAISS sidecar (derived, rebuildable)
├── check/           gate.py          candidate pair generation (AllPairs / Ann)
│                    nli_checker.py   Stage A — DeBERTa MNLI
│                    llm_judge.py     Stage B — structured JSON verdict
│                    providers/       anthropic.py, openai.py, base.py (Protocol)
├── audit/           logger.py        SQLite-backed findings/run records
│                    report.py        Markdown report generation
└── cli/             main.py          typer entry point (consistency-check)
```

## Data flow

```
files on disk
   │
   ▼
corpus/loader     →  Document rows
   │
   ▼
corpus/chunker    →  chunks with (doc_id, char_start, char_end)
   │
   ▼
extract/atomic_facts (LLM)
   │
   ▼
index/assertion_store  ◄── canonical SQLite (documents + assertions)
   │
   ▼
index/embedder + faiss_store  ──►  FAISS index (derived)
   │
   ▼
check/gate (AnnGate top-k)  →  candidate pairs
   │
   ▼
check/nli_checker (Stage A)  →  p_contradiction
   │  (threshold filter)
   ▼
check/llm_judge (Stage B)  →  {verdict, rationale, confidence, evidence_spans}
   │
   ▼
audit/logger  ◄── findings + pipeline_runs
   │
   ▼
audit/report  →  report.md
```

## Storage layout

```
data/store/
├── assertions.db        SQLite — canonical. documents, assertions, findings, pipeline_runs.
└── assertions.faiss     FAISS index. Derived from assertions.db; rebuildable.
```

The SQLite database is the source of truth. The FAISS index is a derived view — `consistency-check store rebuild-index` regenerates it from SQLite. This keeps audit trails portable (plain SQL) and allows swapping the embedding model without losing data.

### Schema sketch

```sql
documents(doc_id PK, source_path, title, doc_date, doc_type, metadata_json, ingested_at)
assertions(assertion_id PK, doc_id FK, assertion_text, chunk_id, char_start, char_end, faiss_row UNIQUE, embedded_at, created_at)
pipeline_runs(run_id PK, started_at, finished_at, config_json, n_assertions, n_pairs_gated, n_pairs_judged, n_findings)
findings(finding_id PK, run_id FK, assertion_a_id FK, assertion_b_id FK, nli_p_contradiction, judge_verdict, judge_confidence, judge_rationale, evidence_spans_json, created_at)
```

`assertion_id = sha256(doc_id || assertion_text)[:16]` → re-running extraction on the same document is idempotent.

## Stage discipline

**Stage A (NLI)** is deliberately permissive — its job is to reduce the O(n²) pair space to something Stage B can afford. Default threshold `p_contradiction > 0.5`, tunable. Bidirectional scoring (A vs B and B vs A) because MNLI models are not symmetric in expectation.

**Stage B (LLM judge)** is the precision layer. Output is schema-validated JSON via SDK tool-use / structured-output features — never string parsing. A Pydantic validator catches malformed responses; retries with repair prompts on failure. Outputs include `evidence_spans` so the report can quote source text without a separate retrieval step.

## Provider abstraction

`check/providers/base.Judge` is a Protocol. Implementations:

- `AnthropicJudge` — uses tool-use for strict JSON.
- `OpenAIJudge` — uses JSON mode / structured outputs.
- `FixtureJudge` — returns canned responses keyed by pair hash. Required for hermetic CI.

Default provider configurable in `config.yml`. Both real providers are exercised under the `live` pytest mark.

## What's not in MVP

- PDF / DOCX loaders (stubbed, raise `NotImplementedError`).
- Numeric reasoning (`Revenue grew 12%` vs `Revenue declined 5%` — currently handled by NLI/LLM only, with no dedicated quantitative extractor).
- Cross-three-document conditional contradictions (only pairwise).
- Entity-resolution beyond what the judge can infer from doc metadata.
- Production deployment / scheduling / monitoring.

## References

- `datarootsio/knowledgebase_guardian` — walked during planning; not forked. Patterns borrowed: config.yml + paths.py shape, dual-logger split. Not reused: the single-stage LLM chain (fragile string-prefix parsing, asymmetric flow).
- FActScore — atomic-fact decomposition prompt template.
- LegalWiz / RAG-on-legal papers — hybrid NLI+LLM precision claim.
- SummaC — sentence-pair granularity for NLI.
