# Changelog

All notable changes to this project will be documented in this file. Format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] — 2026-05-14

The FastAPI + HTMX web UI lands. `consistency-check serve --open` boots a localhost server, opens the default browser, and offers the full ingest → check → review loop without leaving the page. See [`docs/plans/v0.3-block-g.md`](docs/plans/v0.3-block-g.md) for the step-by-step build.

### Added (Block G — web UI)

- **ADR-0007** — FastAPI + HTMX + Jinja2 server-rendered. `cc_`-prefixed template filenames (`cc_base.html`, `cc__pair_diff.html`, …) avoid mixed-app collisions; routes stay clean.
- **G0b — output naming helper.** `consistency_checker/audit/naming.py` produces unique descriptive filenames (`cc_report_<timestamp>_<run_id>.md`, `cc_assertions_<timestamp>.csv`) for every report and export. `consistency-check report --out` and `export --out` are now optional; default destinations live under `<data_dir>/reports/`.
- **G1 — Ingest tab + `POST /uploads`.** Multipart upload, files saved under `<data_dir>/uploads/<timestamp>_<hash>/`, ingest pipeline runs synchronously, success card with a Run / Check now button.
- **G2 — Contradictions tab as `GET /`.** Empty-state banner with a Check-now button when no run exists; populated state lists pair contradictions (verdict ∈ `{contradiction, numeric_short_circuit}`) and multi-document conditional contradictions sorted by confidence. Each row's Diff button opens a side-by-side card view in a base-layout `<dialog>` (2 cards for pair, 3 for triangle).
- **G3 — Documents + Assertions tabs.** Paginated tables (25 rows/page) with per-row detail partials. `AssertionStore.iter_documents()` + `limit/offset` on `iter_assertions()` keep the web layer on the public store API.
- **G4 — Stats tab.** While a run is in flight, the live counters panel polls itself every 2 s via HTMX (`hx-trigger="every 2s" hx-swap="outerHTML"`). On terminal status the polled endpoint returns the final-summary fragment, which has no polling attributes, so the loop stops without client state.
- **G5 — Background tasks + `serve` command.** Migration `0004_run_status.sql` adds `run_status` (`pending | running | done | failed`) to `pipeline_runs`. `pipeline.check` accepts an optional `run_id` to reuse a pre-created row; `AuditLogger` gains `update_run_status()`. `POST /runs` creates a pending row, schedules `_run_check_in_background` via FastAPI `BackgroundTasks`, returns `HX-Redirect: /tabs/stats?run_id=...` (HTMX) or a 303 (direct). New CLI: `consistency-check serve --host 127.0.0.1 --port 8000 [--open]`.
- **simplify pass.** Three-agent code review (reuse / quality / efficiency) drove a post-G5 cleanup: `RunStatus` Literal type, dropped `app.state.*_override` indirection, replaced full-table `iter_assertions` count with `store.stats()["assertions"]`, daemoned the serve `--open` timer, trimmed narrating docstrings.

### Runtime deps

`fastapi>=0.111`, `python-multipart>=0.0.9`, `jinja2>=3.1`, `uvicorn>=0.30`. Dev: `httpx>=0.27` for `TestClient`.

## [0.2.0] — 2026-05-14

Three-document conditional contradictions, numeric short-circuit, and PDF/DOCX loaders. See [`docs/plans/v0.2-build-plan.md`](docs/plans/v0.2-build-plan.md) for the step-by-step build.

### Added (Block D — loaders)

- **D0** — ADR-0004: `unstructured` as the single backend for non-plaintext formats.
- **D1** — `LOADERS` registry in `consistency_checker/corpus/loader.py` (was `if/elif` extension dispatch); `FileLoader` Protocol. `AuditLogger.most_recent_run()`; CLI `report` no longer reaches into `store._conn`.
- **D2** — `UnstructuredLoader` for `.pdf` and `.docx` via `unstructured.partition.auto(strategy="fast")`. Body-content elements (`NarrativeText`, `Title`, `Text`, `ListItem`, `Table`, `UncategorizedText`) concatenated with `\n\n`; sidecar `element_spans` (`element_index`, `element_type`, `char_start`, `char_end`) stored as JSON in `documents.metadata_json`. Runtime dep `unstructured[pdf,docx]`. Dev deps `reportlab` and `python-docx` for session-scope test fixtures that generate small valid PDF/DOCX files at test time.
- **D3** — Mixed-format end-to-end smoke test (`tests/test_e2e.py::test_end_to_end_mixed_format_corpus`) drives ingest + check across a 4-file corpus (`.txt` + `.md` + `.pdf` + `.docx`). README and ARCHITECTURE updated to reflect the new loader surface.

### Added (Block E — numeric short-circuit)

- **E0** — ADR-0005: deterministic sign-flip detector runs before the LLM judge. New verdict label `numeric_short_circuit` distinguishes the deterministic path from the LLM contradiction verdict in the audit DB.
- **E1** — `consistency_checker/extract/quantitative.py` with `QuantitativeTuple(metric, value, unit, polarity, scope)` and `extract_quantities()`. Direction-verb lexicon (`UP_VERBS` / `DOWN_VERBS`), `UNIT_CANON` for canonicalisation, scope regex that masks years inside scope phrases so they aren't read as values.
- **E2** — `pipeline.check` short-circuits sign-flip pairs (same metric / scope / unit, opposite polarity) before invoking the judge. `CONTRADICTION_VERDICTS = {"contradiction", "numeric_short_circuit"}` lets downstream consumers (reporter, web UI) treat both as contradictions.
- **E3** — Range-overlap "uncertain" hint: same-metric pairs that don't sign-flip but disagree by more than `Config.numeric_disagreement_threshold` (default 0.10) get a structured `numeric_context` block in the judge prompt. Prose-only pairs render an empty block, so golden prompts stay stable.

### Added (Block F — three-document conditional contradictions)

- **F0** — ADR-0006: graph-triangle detection on FAISS-similarity edges, opt-in via `--deep`. Sibling `multi_party_findings` table rather than overloading the pair `findings` shape. New verdict label `multi_party_contradiction`. Entity-NER cluster pass deferred to v0.4.
- **F1** — Migration `0003_multi_party.sql` + `AuditLogger.record_multi_party_finding` / `iter_multi_party_findings` / `get_multi_party_finding`. `finding_id` is a content hash over `(run_id, *sorted(assertion_ids))` so re-recording the same triangle within a run replaces the row (idempotent).
- **F2** — `consistency_checker/check/triangle.py` — `find_triangles(pairs, *, max_per_run=1000)` enumerates 3-cliques in the gate graph, dedupes by sorted assertion-id tuple, requires ≥ 2 distinct documents, ranks by min edge similarity desc, caps per run.
- **F3** — `MultiPartyJudge` Protocol + `MultiPartyJudgePayload` (`verdict ∈ {multi_party_contradiction, not_contradiction, uncertain}`, `contradicting_subset`). Sibling providers `AnthropicMultiPartyProvider` (separate `record_multi_party_verdict` tool) and `OpenAIMultiPartyProvider` (same `parse` helper, separate schema). New prompts `prompts/judge_multi_system.txt` + `judge_multi_user.txt`. `FixtureMultiPartyJudge` for hermetic tests.
- **F4** — Pipeline integration: `pipeline.check(..., multi_party_judge: MultiPartyJudge | None)` runs the triangle pass after the pairwise stage on the same gate output. New `Config.enable_multi_party` (default `False`) and `Config.max_triangles_per_run` (default `1000`). New CLI flag `consistency-check check --deep`. `CheckResult` gains `n_triangles_judged` / `n_multi_party_findings`. Report renderer appends a "## Multi-document conditional contradictions" section only when multi-party findings exist (pair-only reports stay byte-stable).

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
