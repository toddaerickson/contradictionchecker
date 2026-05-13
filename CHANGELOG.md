# Changelog

All notable changes to this project will be documented in this file. Format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-05-13

First release. Implements the full 17-step plan from
`/root/.claude/plans/propose-build-steps-start-lovely-eagle.md`.

### Added

- **Foundation** — `uv`-managed Python 3.11+ project with `pytest`, `ruff`,
  `mypy --strict`, and a seven-module package tree (`corpus`, `extract`,
  `index`, `check`, `audit`, `cli`).
- **ADRs** capturing three load-bearing decisions:
  1. Both Anthropic Claude and OpenAI behind a `JudgeProvider` Protocol.
  2. `sentence-transformers/all-mpnet-base-v2` as the default embedder.
  3. CONTRADOC benchmark integrated in MVP scope.
- **Architecture & contributor docs** — `README.md`, `docs/ARCHITECTURE.md`,
  `CONTRIBUTING.md`, `docs/benchmarks.md`.
- **Config / paths / logging** — frozen Pydantic v2 config loaded from
  YAML with `CC_`-prefixed env overrides; idempotent dual-handler
  logger.
- **Assertion store** — SQLite with versioned migrations, foreign keys,
  CSV/JSONL export (default columns `doc_id, assertion_id,
  assertion_text` per the user-facing requirement).
- **Corpus loader & chunker** — `.txt` / `.md` ingest, sentence-window
  chunking via `pysbd`, char-span round-trip preserved.
- **Atomic-fact extractor** — `Extractor` Protocol with
  `FixtureExtractor` (hermetic) and `AnthropicExtractor` (tool-use,
  Pydantic-validated payload).
- **Embedder + FAISS sidecar** — sentence-transformers wrapper,
  L2-normalised vectors in `IndexFlatIP`, resumable `embed_pending`,
  `rebuild_index` for model swaps.
- **Candidate-pair gate** — `AllPairsGate` (baseline) and `AnnGate`
  (FAISS top-k, deduped, threshold-gated, intra-document filter).
- **Stage A NLI checker** — `TransformerNliChecker` against DeBERTa-v3
  MNLI, bidirectional scoring (takes max p_contradiction direction).
- **Stage B LLM judge** — strict JSON schema via tool-use (Anthropic)
  and `beta.chat.completions.parse` (OpenAI). Retry-with-repair on
  validation failure; degrades to `uncertain` rather than crashing.
- **Audit logger** — SQLite-backed `pipeline_runs` and `findings`
  tables; every judge verdict (contradiction / not_contradiction /
  uncertain) recorded so runs are reproducible from logged inputs.
- **Markdown report** — deterministic output, summary table sorted by
  confidence, per-pair findings with evidence spans and rationale.
- **CLI** — `consistency-check ingest <corpus_dir>`, `check`, `report
  --out`, `export csv|jsonl`, `store stats`, `store rebuild-index`.
- **End-to-end smoke test** — hermetic `e2e_fixture` plus a `live`
  variant requiring `ANTHROPIC_API_KEY`.
- **CONTRADOC benchmark harness** — JSONL-driven precision/recall/F1
  reporting, per-pair predictions for post-hoc threshold tuning.
- **CI + tooling** — GitHub Actions workflow (ruff, mypy, pytest, uv
  build + wheel smoke install), `pre-commit` config, this changelog.

### Risk areas (carry-forward)

The original plan flagged three critical-path steps. Status:

- **Assertion schema** — pinned in migration `0001_init.sql`; later
  audit tables added via `0002_audit.sql` without disturbing the
  original columns. ✅
- **Judge structured output** — Pydantic schema enforced via
  provider-native structured-output paths; retry-with-repair plus
  degraded fallback eliminate the failure mode from the prior art. ✅
- **NLI threshold** — default `p_contradiction >= 0.5`, tunable in
  config. The CONTRADOC harness emits per-pair predictions so the
  threshold can be re-swept post-hoc without re-running inference. ✅

### Known limitations

- PDF / DOCX loaders stub out with `NotImplementedError` (`.pdf` /
  `.docx`).
- Chunk overlap > 0 is unimplemented; raises explicitly.
- Numeric reasoning (qualitative sign-flips like "grew 12%" vs.
  "declined 5%") rides on the NLI + LLM stack; no dedicated
  quantitative extractor.
- Three-document conditional contradictions (pairwise checks pass but
  the conjunction contradicts) are out of scope.
- The CLI's `check` command opens a real `TransformerNliChecker` —
  expect a ~800 MB model download on first run.
