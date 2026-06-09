# Changelog

All notable changes to this project will be documented in this file. Format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.1] — 2026-06-08

First-run polish for `pip`/`pipx` installs.

### Added

- **`consistency-check init`** — writes a starter `config.yml` and a `.env`
  template into the current directory so a fresh install runs without
  hand-authoring config. Skips existing files unless `--force` is passed. The
  "config.yml not found" error now points at it.

## [0.4.0] — 2026-06-06

First public PyPI release. Since 0.3.0 the product reorients around the
**definition-inconsistency detector** (pairwise contradiction is now opt-in), the
web UI is rebuilt as a **single page**, ingest moves to a non-blocking background
job, and the **Moonshot/Kimi** provider plus **OCR** for scanned PDFs land.

### Added

- **Single-page web UI (ADR-0017).** Replaces the 7-tab UI with a corpora
  sidebar + findings pane: New Corpus and Run Check modals, slide-over
  Assertions / Definitions / Stats drawers, inline reviewer verdicts with filter
  chips, a persistent estimated-cost gauge, and per-corpus SSE run progress.
  `GET /` serves it; `?legacy=1` and `/tabs/*` return 410.
- **Background corpus ingest (ADR-0019).** Uploads run as a tracked background
  job with a live "files done / total" progress bar in the sidebar, instead of
  blocking the request (a large or scanned upload used to freeze the whole UI).
  Adds an on-page "How it works" onboarding guide and button loading states, and
  surfaces files that yield no extractable text (scanned images / OCR
  unavailable) rather than reporting a silent 0-assertion success.
- **Corpus & run management.** Delete corpus (cascades documents / runs /
  findings); clear a stuck pending/running run so a crashed worker can't block a
  corpus.
- **Exports for analysis.** Findings CSV (honours the active filter), plus
  assertions and term→definition CSV exports; download files are named after the
  corpus (e.g. `atkins-assertions.csv`).
- **Moonshot (Kimi) provider** for the judge and extractor, wired through the
  provider abstraction; loads a gitignored `.env`. Now the default provider.
- **OCR fallback for scanned PDFs (ADR-0014).** Auto-escalates to `unstructured`
  hi-res when fast extraction returns near-empty text; `--no-ocr` to disable.
- **Corpus isolation.** Logical, single-DB corpora; `--corpus` is required on
  `ingest` / `check` / `estimate-cost` / `export`. Corpus-composition warning and
  opt-in, advisory org-grouping.
- **Pairwise contradiction detector is opt-in (ADR-0015):** `--pairwise`
  (default off); the definition-inconsistency detector is the default lever. The
  first pairwise check downloads the ~440 MB DeBERTa NLI model; the default check
  does not.
- **Cost ceiling (ADR-0016).** `--max-cost` aborts before judge/NLI bootstrap if
  the conservative estimate is exceeded; provider-aware `estimate-cost` defaults.
- **Definition-inconsistency detector** with a judge short-circuit, plus
  evaluation tooling under `benchmarks/definition_eval/`: a P/R/F1 harness, a pair
  miner, and a keyboard-driven labeler.
- **PyPI packaging.** Apache-2.0 license, OIDC Trusted-Publishing release
  workflow (no stored tokens), `docs/RELEASING.md`, and a `pipx` install path.

### Changed

- **Default NLI model → `DeBERTa-v3-base`** and weights are released after the
  pair loop (see the memory-hardening notes below).
- **PDF extraction junk filter** at the text and assertion stages.

### Fixed

- **Provider error-path hardening.** An empty `choices` list from an
  OpenAI-compatible API (Moonshot/OpenAI) now raises the clean `ValueError`
  callers already handle instead of an `IndexError`; the model-download cache
  probe debug-logs unexpected failures instead of swallowing them silently; and
  upload filename validation uses an explicit `400` guard rather than an
  `assert` that `python -O` strips.
- **Definition detector compares only real definitions.** An `is_definitional`
  gate keeps only assertions whose source reads `"Term" means / shall mean …`,
  dropping the extractor's mis-tagged *usages* of a capitalized defined term and
  *cross-references* — which were being paired against real definitions and
  reported as spurious divergences (on one real credit agreement ~half the
  tagged "definitions" were references).
- **Web UI: SSE reconnect flood + accessibility.** Idle corpora no longer open a
  reconnect-looping progress stream (gated on an active run; an SSE `retry:`
  directive parks reconnection on completion); dialogs gained
  `role`/`aria-modal`/`aria-labelledby`; the sidebar active-row highlight
  follows the selected corpus; and corpus-name validation is enforced
  client-side (a `pattern` valid under the RegExp `v` flag).

### Removed — judge confidence score (ADR-0018)

- **The LLM judge `confidence` score is gone, end to end.** It was the model's
  self-reported "subjective certainty", not a calibrated probability, and
  presenting it as a 0–1 figure implied a precision the system cannot produce.
  Removed from the judge schemas/prompts, the `JudgeVerdict` family, the
  `findings` / `multi_party_findings` tables (migration
  `0015_drop_judge_confidence.sql`), the markdown report, the web findings card
  + definitions drawer, and the findings CSV export.
- **Calibration tooling removed.** `audit/eval.py` loses
  `compute_calibration` / `CalibrationBin` and the `eval` command's calibration
  table; `report --min-confidence` and `eval --detector` are gone. The
  reviewer-verdict **precision** report (label-based) is kept.
- **Findings are no longer sorted by confidence** — the report orders by
  `finding_id` within document-pair groups; the web list by
  `(doc_a, doc_b, finding_id)`. The NLI contradiction probability is a separate
  classifier signal and is retained.

### Memory / OOM hardening — round 2

- **Default NLI model switched to `DeBERTa-v3-base`** (`MoritzLaurer/
  DeBERTa-v3-base-mnli-fever-anli`). Drops NLI RSS from ~1.5–2 GB to
  ~0.6 GB and first-run download from ~1.5 GB to ~440 MB. Stage B's LLM
  judge catches most of what the slightly lower gate recall lets through.
  Users who want max recall can pin large via
  `nli_model: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`.
- **`NliChecker.release()`** new Protocol method drops model weights /
  tokenizer / pipeline. `pipeline.check` calls it after the pair loop
  completes so the LLM-judge multi-party + definition passes don't compete
  with NLI for RSS. Idempotent. `FixtureNliChecker.release()` is a no-op.

### Memory / OOM hardening — round 1

- **`pipeline.check` streams candidate pairs** instead of materialising the
  full gate output. `_iter_candidates` now returns `Iterator[CandidatePair]`;
  the strong-key set used by the triangle pass is built inline during the
  pair-loop, and the triangle pass re-iterates the strong gate + chains the
  weak gate lazily via `itertools.chain`. Saves ~500 B per candidate × top_k
  × N assertions at peak.
- **`TransformerNliChecker` wraps the forward pass in `torch.inference_mode()`**
  — drops autograd-tape allocation on every score, meaningful peak-RSS
  reduction on the DeBERTa-large model.
- **`gc.collect()` between phases** in `pipeline.check` encourages the
  allocator to release per-pair scratch (NLI tensors, judge SDK response
  objects) before the triangle / definition passes allocate.
- **`Config.max_memory_mb`** opt-in pre-flight: when set, `consistency-check
  check` aborts before loading the NLI model if `MemAvailable` is below the
  threshold, with an actionable message instead of an OOM kill. A soft
  warning fires when MemAvailable is below the ~2.5 GB peak estimate even
  without the config field. Skips silently when `psutil` isn't installed.
- **Runtime dep:** `psutil>=5.9` for the pre-flight check.
- **[`docs/corporate-setup.md`](docs/corporate-setup.md) §4** documents the memory budget, devcontainer /
  WSL2 caps, and the smaller `DeBERTa-v3-base` fallback for tight envs.

## [0.3.0] — 2026-05-14

The FastAPI + HTMX web UI lands. `consistency-check serve --open` boots a localhost server, opens the default browser, and offers the full ingest → check → review loop without leaving the page. See ADR-0007 for the architectural decisions.

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

Three-document conditional contradictions, numeric short-circuit, and PDF/DOCX loaders. See ADR-0004, ADR-0005, and ADR-0006 for the architectural decisions.

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
