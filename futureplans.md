# Future plans

Forward-looking work beyond the v0.3 release. Items are sized roughly; pick from the top when starting a new milestone. When work is finished, move the entry to the **Completed** section below rather than deleting it — provenance matters when revisiting decisions.

## v0.4 — precision and provenance

Higher-effort items that require more design discussion. Some carry from the old v0.3 list because v0.3 ended up focused on the web UI (Block G) instead.

### 6. Three-document conditional contradictions — cluster-by-entity second pass

v0.2 shipped the graph-triangle approach (see [`docs/plans/v0.2-build-plan.md`](docs/plans/v0.2-build-plan.md) Block F): triangles enumerated on FAISS-similarity edges, opt-in via `--deep`. That catches conditional contradictions whose three assertions are pairwise similar enough to clear the gate threshold. It **misses** triangles whose edges fall below the gate — e.g. `A` and `C` referencing the same entity but in vocabulary that the embedder doesn't place near each other.

v0.4 adds a second pass that fires **after** the triangle pass and shares the same `multi_party_findings` table:

- Run a lightweight NER pass (see #7) to canonicalise entities across assertions.
- For every entity that appears in ≥ 3 assertions spanning ≥ 2 documents, build a cluster.
- For each cluster larger than a triangle, prompt the judge with up to N assertions and the structured question "does any subset jointly contradict the others?" Use the same `MultiPartyJudge` Protocol and prompt template; the only delta is the input cardinality and the cluster-source label in the finding's metadata.
- Dedupe against findings already produced by the triangle pass.

Depends on #7 (entity resolution). Keep #7 → #6 ordering so cluster construction has stable entity ids.

### 7. Entity resolution pass
Document A says "the Borrower", Document B says "ABC Corp". The current judge can sometimes infer they're the same; often it can't. A lightweight NER pass (`spaCy` `en_core_web_lg`) plus a fuzzy-match canonicaliser would let us pass `entity_ids` into the judge prompt, raising precision on entity-coreference contradictions. Audit findings should record the resolved entity id so reports can group by entity. This is the prerequisite for #6's second pass.

### 8. LanceDB migration option
SQLite + FAISS works at corpus sizes up to ~1M assertions. Beyond that, LanceDB consolidates store + vectors with native filtering. Worth prototyping as an alternative `Store` Protocol so users can choose. ADR-0002's "domain model" alternative path goes here too — once LanceDB is in place, domain embedders are configurable rather than code-bound.

### 9. Reviewer workflow
The audit DB has `findings.reviewer_verdict` columns reserved (or should — verify the schema). Build a tiny TUI (or web view) that walks reviewers through unreviewed findings and lets them set `reviewer_verdict` ∈ `{confirmed, false_positive, dismissed}`. Findings that reviewers confirm as false positives should feed back into prompt iteration. v0.3 Block G already provides the FastAPI/HTMX shell — #9 lives mostly in new templates + a `reviewer_verdict` column.

## v0.4+ — operational

### 10. Incremental scans
Today every `check` invocation re-scans the whole corpus. If only one new document was added since last run, only its assertions need to be paired against pre-existing ones. Use the `documents.ingested_at` timestamp to compute a "new assertions" set, then gate against the old set without re-pairing the old set with itself. Audit logger gains a `prior_run_id` link so report can show "new vs. carry-forward findings."

### 11. Web review surface (extends v0.3 Block G)
The FastAPI + HTMX app shell shipped in v0.3 ([`docs/plans/v0.3-block-g.md`](docs/plans/v0.3-block-g.md)). #11 adds the reviewer-specific routes once #9 lands: read-only mode for stakeholders, reviewer mode for analysts who can set `reviewer_verdict`.

### 12. Local-LLM judge provider
For air-gapped deployments: a `LocalLlamaProvider` against `llama.cpp` or vLLM. The `JudgeProvider` Protocol already exists — this is mostly prompt tuning and tooling, not architecture. Document the quality trade-off in a new ADR.

### 13. Metrics dashboard
The audit DB already records latency, token counts, and verdict distribution per run. A simple Streamlit page that reads the DB and plots:
- Token spend per run.
- Verdict distribution over time.
- Precision tracking against the rolling CONTRADOC benchmark.

### 14. Zero-CLI desktop launcher
v0.3 ships `consistency-check serve --open` which auto-launches the browser after binding (close to "double-click → tool opens" but still requires a terminal). v0.4 adds a true zero-CLI entry point:

- **Option A — `pipx run consistency-checker-launcher`**: a tiny separate package that depends on `consistency-checker` and exposes a single entrypoint script which runs the equivalent of `consistency-check serve --open` plus a graceful shutdown handler. Users install once via `pipx`; subsequent launches are one shell command from any cwd.
- **Option B — PyInstaller bundle**: a single-file executable per platform (macOS `.app`, Windows `.exe`, Linux AppImage) that bundles Python, the package, and the model caches. True double-click experience; larger artefacts (~200–400 MB once HuggingFace caches are baked in) and more release engineering. Decide whether to ship caches or download-on-first-run.

Pick A or B based on the target audience: A for technical users who already have Python; B for analyst-class users who don't. ADR-0008 captures the choice.

## Deferred from the v0.3 simplify pass

Flagged during the three-agent code review after G5 merged. Each is well-scoped but bigger than a one-line fix.

### 15. Bulk-fetch helpers on `AssertionStore` (N+1 fix)
`web/app.py::index` and `tab_assertions` issue per-row `get_assertion` / `get_document` queries. Add `get_assertions_bulk(ids: Sequence[str])` and `get_documents_bulk(ids)` with `WHERE id IN (...)` queries; the web routes collect all ids in a first pass, then build rows in a second. Cuts the index-page query count from `4 × n_findings + 2` to `4`.

### 16. Cache embedder dimension to avoid a SentenceTransformers load per run
`web/app.py::_run_check_in_background` calls `_open_stores()`, which constructs a real `SentenceTransformerEmbedder` (`pipeline.make_embedder`) purely to extract `.dim` for `FaissStore.open_or_create`. In production this triggers a multi-hundred-MB model load on every kick-off. Either persist `dim` to a config file / FAISS-file sidecar so `_open_stores` can skip the embedder when the index already exists on disk, or refactor `FaissStore.open_or_create` to read `dim` from the existing index header.

### 17. Extract `cc__run_button.html` partial
The `<form hx-post="/runs">` block appears in both `cc__upload_success.html` and `cc_contradictions.html` with minor style + label differences. Lift into a partial taking `button_label` / `inline` params and `{% include %}` from both sites.

### 18. Lift `audit_logger.begin_run` out of `pipeline.check`
v0.3 G5 added a `run_id` kwarg so the web layer could reuse a pending row. Cleaner: have all callers (CLI, web layer) call `begin_run` themselves and pass a required `run_id` into `pipeline.check`. Removes the dual-mode branch at the top of `check()` and keeps the function single-purpose.

## Known issues to fix opportunistically

- `consistency_checker/cli/main.py:140` reaches into `store._conn` to fetch the most recent run id. Add a `AuditLogger.most_recent_run()` method so the CLI doesn't touch a private attribute.
- The benchmark harness bypasses Stage 7 (atomic-fact extraction). Worth surfacing in `docs/benchmarks.md` more loudly so users don't compare CONTRADOC F1 to "full pipeline F1" and get confused.
- v0.1.0 ships with `embedder_model: "hash"` accepted by config validation only because Pydantic doesn't enforce the model name. A `Literal[...]` would catch typos at config-load time, but it would also lock out user-supplied HF model ids. Decide whether validation should be strict or permissive — currently permissive by accident.

## Completed

(Move items here as they ship, keep a one-line note on which release.)

- **v0.1.0** — full 17-step build plan: ingest → chunk → atomic-fact extraction → embed → gate → NLI → judge → audit → report → CLI → CONTRADOC harness → CI. See `CHANGELOG.md`.
- **v0.2.0** — Block D (PDF/DOCX loaders via `unstructured`), Block E (numeric short-circuit + range-overlap hint), Block F (three-document conditional contradictions via graph triangles, `--deep` flag). See [`docs/plans/v0.2-build-plan.md`](docs/plans/v0.2-build-plan.md).
  - Item #1 (PDF/DOCX) — D2 / ADR-0004.
  - Item #3 (numeric extractor) — E1–E3 / ADR-0005.
  - Items #6 partial (three-doc graph-triangle half) — Block F / ADR-0006. Cluster-by-entity second pass still pending in v0.4.
- **v0.3.0** — Block G web UI: FastAPI + HTMX, Contradictions / Documents / Assertions / Stats / Ingest tabs, Diff partials, HTMX self-polling, `consistency-check serve --open` CLI command, `run_status` migration + `BackgroundTasks` for check. See [`docs/plans/v0.3-block-g.md`](docs/plans/v0.3-block-g.md).
  - Output naming helper (`audit/naming.py`) + optional `--out` on `report` / `export` (G0b).
  - simplify pass: `RunStatus` Literal type, closure-captured overrides, dead-code cleanup.
