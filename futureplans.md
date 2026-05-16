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

### 9b. Reviewer workflow — Phase B (dedicated queue + small extensions)
Parked from the v0.4 Phase A build (item #9, shipped). Three pieces:

- **Dedicated "Review" tab** — focused per-finding queue with big buttons,
  skip/back navigation, optional batch-mode keyboard flow. The schema and
  setter API landed in Phase A; this is a new UI surface that uses them.
- **Note column UI** — `reviewer_verdicts.note` exists in the schema with
  no v1 UI. The queue page is the natural place to surface it.
- **Findings CSV export** with a `reviewer_verdict` column for downstream
  tooling.

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

Pick A or B based on the target audience: A for technical users who already have Python; B for analyst-class users who don't. A new ADR captures the choice when this is scheduled.

## v0.4.1 — embedding, matching, and semantic tuning

Four focused improvements to gate quality and triangle recall. These emerged from the precision audit and targeted-eval phase.

### 2. Entity Resolution

**Problem:** Assertions about "John Smith" and "J. Smith" are treated as separate entities across documents. This causes false negatives on contradictions that span variant names.

**Approach:** Implement entity linking to canonical forms. Use heuristics (prefix matching, edit distance under 2) as first pass; reserve ML-based entity clustering for v0.5.

**Impact:** Catch contradictions that would otherwise be silent due to name variance across corpus.

### 3. FAISS Triangle Recall

**Problem:** `find_triangles()` uses brute-force O(n³) search. For corpora with >10k assertions, the triangle finder becomes a bottleneck. Many conditional contradictions (A vs B, B vs C, therefore A vs C) are never formed because the search times out.

**Approach:** Index triangle candidates by assertion pairs in FAISS as a pre-filter. Query: given a strong pair (A, B), find all C that are similar to B. Rank by similarity and check top-k for transitive edges.

**Impact:** 10–100× speedup for large corpora; unlock triangle detection at scale.

### 4. Embedding Model

**Problem:** `sentence-transformers/all-MiniLM-L6-v2` (384 dims, 22M params) is small and fast but misses semantic distinctions. Semantic contradictions (e.g., "all employees have benefits" vs "some employees lack healthcare") rely on embedding quality.

**Approach:** Evaluate larger models: `all-mpnet-base-v2` (768 dims, 110M), `all-mpnet-base-v2-distilled` (384 dims, faster than MiniLM). Benchmark on synthetic contradiction pairs and real corpus. Make configurable in `config.py`.

**Impact:** Catch contradictions that fall below the similarity gate due to weak embeddings. Trade: slower ingestion (mitigated by batch processing).

### 5. MNLI Model

**Problem:** DeBERTa (`microsoft/deberta-v3-small`) is the NLI gate, but it's a general-purpose model. Specialized MNLI fine-tuning or task-specific calibration could reduce false negatives on domain-specific contradictions (e.g., financial/legal term precision).

**Approach:** Evaluate fine-tuned MNLI models or ONNX-accelerated variants. Add config field `nli_model_name`. Benchmark gate recall on real contradictions. If using a heavier model, prototype ONNX export for inference speed.

**Impact:** Higher-confidence gate verdicts; reduce hallucination-driven false contradictions. Trade: ingestion latency increase (partially offset by ONNX or quantization).

## Accessibility plan — making the tool usable by non-developers

Phased plan targeting analysts, paralegals, and other "normal tech users" who are
not comfortable with the CLI. Each phase is independently shippable.

### Phase 0 UI — surface the three Phase 0 eval signals in the web UI

See [`docs/plans/phase0-ui-plan.md`](docs/plans/phase0-ui-plan.md). Brings the
CLI-only outputs of PRs #51 (in-flight precision + calibration mining), #52
(targeted-eval on 120 pairs), and #53 (Claude-Projects baseline) into the
existing FastAPI + HTMX web UI so a non-programmer analyst never opens a
terminal to operate Phase 0. Four PRs (Blocks A, B, C1, C2). Sequenced before
the existing Phase 1–3 accessibility work because there is no point in
polishing terminology before the actual data signals are reachable.

### Phase 1 — Zero-effort wins (no architecture changes, ~1 session)

All five items touch only templates, CSS, and the stats route. No migration, no CLI
changes, no new dependencies.

**U1. Terminology audit across all templates.**
Replace ML jargon with plain language throughout the web UI:

| Current | Replacement |
|---------|-------------|
| Assertions | Statements |
| Pairs judged | Statement pairs checked |
| Pairs gated | Candidates screened |
| Multi-party | Cross-document |
| `uncertain` verdict label | (never shown; keep hidden) |

Files: every `cc_*.html` and `cc__*.html` template. No Python changes.

**U2. Surface run failure reason in Stats tab.**
When `run_status = "failed"`, the stats panel shows "Run failed" with no context.
Add an `error_message` column to `pipeline_runs` (migration 0006); populate it
in `_run_check_in_background`'s `except` block via `update_run_status(run_id, "failed",
error_message=str(exc))`; render it in `cc__stats_final.html` beneath the status
heading. Users see "Run failed — No API key found" instead of a silent dead end.

**U3. Redirect landing tab based on corpus state.**
The `index()` route currently always loads the Contradictions tab. When there are
zero documents in the store, redirect (HTMX-aware) to `/tabs/ingest` so new users
land on the upload form rather than a blank screen. One `if store.stats()["documents"] == 0`
guard in `index()` before the findings query.

**U4. "Start here" empty-state banner on Ingest tab.**
When zero documents are ingested, replace the current sparse Ingest empty state with
a three-step visual: **① Upload → ② Check → ③ Review results**. Uses existing
`.cc-banner` CSS. No new routes.

**U5. Run-failure details partial.**
Extract a `cc__stats_failed.html` partial (alongside the existing `cc__stats_live.html`
and `cc__stats_final.html`) that shows the error message from U2 and a "Try again"
button. Keeps the stats route switch statement clean.

---

### Phase 2 — Guided first-run flow (moderate effort, ~2 sessions)

Requires one new route and small pipeline changes. No new dependencies.

**U6. First-check confirmation dialog.**
Before firing `POST /runs`, pop a `<dialog>` that explains what's about to happen:
"This will analyse N statements across M documents using your configured LLM. Continue?"
Prevents accidental API charges and makes the action feel deliberate. Uses the existing
`cc-dialog` CSS class.

**U7. Post-upload "Check now" auto-flow.**
After a successful upload, the `cc__upload_success.html` partial currently shows a
static "Run / Check now" button. Change it to: if there are no prior runs, render a
prominent "Check for contradictions →" CTA that goes straight to the Stats tab after
firing. If there are prior runs, keep the current button. Uses the existing HTMX
redirect path in `POST /runs`.

**U8. Download-warning gate in CLI before first check.**
Before `pipeline.check()` instantiates `TransformerNliChecker` for the first time,
check whether the HF model cache already contains the NLI model weights. If not,
print a one-line warning: `"First run: downloading ~800 MB NLI model — this takes a
few minutes."` No progress bar needed; the warning alone removes the "it hung"
perception. Gate: `Path(huggingface_hub.constants.HF_HUB_CACHE) / <model_slug>`.

**U9. Inline check progress on Stats tab.**
The live-poll panel already shows "Run in progress" and counter tiles. Add a short
English sentence computed from the counters: *"Checked 42 of ~210 pairs — about 3
minutes remaining."* The estimate is rough (pairs_judged / elapsed × remaining) but
gives users a sense of pace. Computed server-side in `_live_counters()`, rendered in
`cc__stats_live.html`.

---

### Phase 3 — Packaging (deferred, high effort)

Prerequisite: Phases 1–2 shipped and validated. No code changes planned here yet;
entries are carried from the existing roadmap.

**U10 (= item #14 Option A) — pipx launcher.**
A tiny `consistency-checker-launcher` package on PyPI that runs `consistency-check serve --open`.
Technical users install once via `pipx`; subsequent launches are one command from any
directory. Avoids the `uv sync` / virtualenv friction entirely.

**U11 (= item #14 Option B) — PyInstaller bundle.**
Single-file executable per platform (macOS `.app`, Windows `.exe`). True double-click
experience. Bundles Python and the package; model weights download on first run.
~200–400 MB artefact. Requires a separate release pipeline. Decide A vs B via ADR
once Phase 2 is validated with real users.

---

## Deferred from the v0.3 simplify pass

Flagged during the three-agent code review after G5 merged. Each is well-scoped but bigger than a one-line fix.

### 15. Bulk-fetch helpers on `AssertionStore` (N+1 fix)
`web/app.py::index` and `tab_assertions` issue per-row `get_assertion` / `get_document` queries. Add `get_assertions_bulk(ids: Sequence[str])` and `get_documents_bulk(ids)` with `WHERE id IN (...)` queries; the web routes collect all ids in a first pass, then build rows in a second. Cuts the index-page query count from `4 × n_findings + 2` to `4`.

### 17. Extract `cc__run_button.html` partial
The `<form hx-post="/runs">` block appears in both `cc__upload_success.html` and `cc_contradictions.html` with minor style + label differences. Lift into a partial taking `button_label` / `inline` params and `{% include %}` from both sites.

## Persona-aware analysis and the detector family

Design shape locked in [ADR-0008](docs/decisions/0008-persona-aware-analysis.md) (status: Proposed). Build is gated on eval data; capture here so the design doesn't drift.

### 19. Persona-aware scoring + presentation
The same document set is read differently by different consumers — employee vs manager vs HR professional; lender vs credit analyst vs borrower vs borrower's counsel. Per ADR-0008, this is a **view layer**, not forked judge agents: a `Persona` config object (interests, scope assumptions, materiality weights) feeds (a) an impact scorer that re-ranks/filters the shared `findings` per persona, and (b) an optional `persona_context` block spliced into the judge prompt the way E3's `numeric_context` already works. Core detector, audit trail, and cache stay single and shared. Ship the report/web persona *filter* first; add the prompt hook only once eval shows borderline scope calls actually flip per persona.

### 20. Consistency-detector family (gap / ambiguity detectors)
The contradiction judge can only return verdicts about two assertions that both exist — so it structurally cannot find `X` vs silence (a **gap**) or `X` vs vague (an **ambiguity**). A borrower's counsel reading a loan package cares about exactly those: an obligation promised in the term sheet but unaddressed in the credit agreement; a "material adverse change" clause loose enough to litigate. These are *new detectors* that share CrossCheck's ingest / atomic-fact / embedding / audit infrastructure but ask a different question of the assertion graph. CrossCheck's contradiction detector is the first of the family; the **definition-inconsistency detector** (ADR-0009, flavor A) shipped in v0.4 as the second. Each new detector gets its own ADR. Personas (#19) then map to *which detectors run*, not just how findings rank.

### 20a. Definition ↔ usage drift (flavor B of the definition detector)
Parked from the v0.4 definition-inconsistency build (ADR-0009). Shape: `(definition assertion, usage assertion, verdict)`. Requires a "usage extraction" pass — finding every occurrence of a defined term in a context that is not itself the definition. Larger candidate set than flavor A; storage shape stays pair-isomorphic so no `findings` schema change is anticipated. Build once flavor A has produced real findings on user corpora and we have a sense of usage-vs-definition signal vs noise.

## Known issues to fix opportunistically

- The benchmark harness bypasses Stage 7 (atomic-fact extraction). Worth surfacing in `docs/benchmarks.md` more loudly so users don't compare CONTRADOC F1 to "full pipeline F1" and get confused.
- v0.1.0 ships with `embedder_model: "hash"` accepted by config validation only because Pydantic doesn't enforce the model name. A `Literal[...]` would catch typos at config-load time, but it would also lock out user-supplied HF model ids. Decide whether validation should be strict or permissive — currently permissive by accident.

## Completed

(Move items here as they ship, keep a one-line note on which release.)

- **f4-fixups** — `AuditLogger.most_recent_run()` + CLI private-attr fix (already done on branch); `FaissStore.open_or_create` made `dim`-optional (reads from existing index header); `_run_check_in_background` no longer loads the ~800 MB embedder model for check runs; `pipeline.check` now requires a pre-created `run_id` — callers own `begin_run` (items #16, #18, and the `main.py:140` known issue).

- **v0.1.0** — full 17-step build plan: ingest → chunk → atomic-fact extraction → embed → gate → NLI → judge → audit → report → CLI → CONTRADOC harness → CI. See `CHANGELOG.md`.
- **v0.2.0** — Block D (PDF/DOCX loaders via `unstructured`), Block E (numeric short-circuit + range-overlap hint), Block F (three-document conditional contradictions via graph triangles, `--deep` flag). See [`docs/plans/v0.2-build-plan.md`](docs/plans/v0.2-build-plan.md).
  - Item #1 (PDF/DOCX) — D2 / ADR-0004.
  - Item #3 (numeric extractor) — E1–E3 / ADR-0005.
  - Items #6 partial (three-doc graph-triangle half) — Block F / ADR-0006. Cluster-by-entity second pass still pending in v0.4.
- **v0.3.0** — Block G web UI: FastAPI + HTMX, Contradictions / Documents / Assertions / Stats / Ingest tabs, Diff partials, HTMX self-polling, `consistency-check serve --open` CLI command, `run_status` migration + `BackgroundTasks` for check. See [`docs/plans/v0.3-block-g.md`](docs/plans/v0.3-block-g.md).
  - Output naming helper (`audit/naming.py`) + optional `--out` on `report` / `export` (G0b).
  - simplify pass: `RunStatus` Literal type, closure-captured overrides, dead-code cleanup.
- **v0.4 (definition-inconsistency detector)** — flavor A of item #20: divergent definitions of the same canonical term across the corpus. New migrations `0007_assertion_kind.sql` (`assertions.kind/term/definition_text`) and `0008_finding_detector_type.sql` (`findings.detector_type`); definitions extracted alongside atomic facts via the existing tool-use schema; new `DefinitionChecker` skips the NLI gate in favour of canonical-term grouping; new `DefinitionJudge` provider surface (Anthropic + OpenAI); audit, report, web UI, and CLI extended; `--no-definitions` opt-out. ADR-0009. Flavor B (definition ↔ usage) parked as item #20a.
- **v0.4 (reviewer workflow, Phase A)** — item #9: inline verdict buttons on Contradictions / Definitions / Cross-document tabs; content-keyed `reviewer_verdicts` table (migration 0009) keyed by `(pair_key, detector_type)` so verdicts survive re-runs; hide-by-default with per-section "Show reviewed" toggle; persistent undo toast (no auto-dismiss); keyboard shortcuts C/F/D when a finding row has focus; markdown report filters `false_positive` and tags surviving findings with `**Reviewer:** Real issue/Dismissed/Pending review`. Phase B (dedicated queue, note column UI, findings CSV) parked as item #9b.
