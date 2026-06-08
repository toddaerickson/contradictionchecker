# Future plans

Forward-looking work beyond the v0.3 release. Items are sized roughly; pick from the top when starting a new milestone. When work is finished, move the entry to the **Completed** section below rather than deleting it — provenance matters when revisiting decisions.

## Eval findings & next levers (2026-05-21)

Real-corpus eval (3 nonprofit-bylaws PDFs + an earlier loan/partnership corpus, Moonshot/Kimi judge) produced three decisions that should steer the next milestone pick. No ground truth — these are operating-point observations, not precision/recall.

1. **Pairwise contradiction detector low-yield on legal/contract prose — flipped to opt-in (shipped 2026-05-31, see Completed).** Original observation retained for provenance: full funnel on a real partnership/loan corpus (3,416 assertions → 901 candidate pairs → 254 NLI-flagged) yielded **1** borderline, non-reproducing "contradiction" (kimi non-deterministic; 22/24 borderline pairs flipped verdict across re-runs). Adding containing-sentence context made the judge confidently find **zero**. The judge is competent; the detector just earns ~nothing on prose at high compute cost. Lean into definition-inconsistency + the obligation/date-conflict and cross-reference detectors (items #20/#20a) instead.

2. **PDF junk filter shipped (PR #61) and validated — but it only fixed the extraction-noise slice.** It removed dot-leaders / page-numbers / cross-reference "definitions" (bylaws corpus: 158→84 definitions, 93→24 same-term pairs, zero junk fragments reaching the judge). Definition-divergent rate fell only **84%→75%** because the residual is NOT extraction noise.

3. **The residual divergent noise split into one shipped fix and one still-open corpus problem:**
   - **Definition-judge identical-text precision shipped.** The high-confidence subcase (identical definitions flagged as divergent, e.g. *"the authorized number of directors of the Corporation"* vs itself) is now handled by the deterministic short-circuit recorded under Completed below.
   - **Corpus composition.** Comparing 3 *unrelated organizations'* bylaws makes every shared term ("Director", "Quorum") "diverge" by construction. Meaningful detection needs a *single* entity's governing docs. Worth a UI/doc warning when a corpus spans unrelated sources, and/or entity-grouping (cf. item #7).
     Implementation: see ADR-0012 (`docs/decisions/0012-corpus-org-warning.md`).
     Default is advisory-only warning; cross-org suppression is opt-in via
     `--org-scope`. Backfill via `consistency-check store reidentify-orgs`.
     §9 measurement (bylaws-corpus divergent-rate delta) pending.

## v0.4 — precision and provenance

Higher-effort items that require more design discussion. Some carry from the old v0.3 list because v0.3 ended up focused on the web UI (Block G) instead.

### 6. Three-document conditional contradictions — cluster-by-entity second pass

v0.2 shipped the graph-triangle approach (see ADR-0006): triangles enumerated on FAISS-similarity edges, opt-in via `--deep`. That catches conditional contradictions whose three assertions are pairwise similar enough to clear the gate threshold. It **misses** triangles whose edges fall below the gate — e.g. `A` and `C` referencing the same entity but in vocabulary that the embedder doesn't place near each other.

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

### Corpus archive (deferred)

Bundle a corpus + its runs + verdicts into a portable artifact (tarball +
manifest, optional cloud upload) for review and off-machine retention.
Companion to ADR-0013; will be drafted post-isolation-merge. Once a corpus
is a real isolation unit, the export artifact is well-defined: all
`corpora`, `documents`, `assertions`, `pipeline_runs`, `findings`, and
`reviewer_verdicts` rows for the corpus, plus the FAISS sub-index sliced
to that corpus's assertion-id set.

### 10. Incremental scans
Today every `check` invocation re-scans the whole corpus. If only one new document was added since last run, only its assertions need to be paired against pre-existing ones. Use the `documents.ingested_at` timestamp to compute a "new assertions" set, then gate against the old set without re-pairing the old set with itself. Audit logger gains a `prior_run_id` link so report can show "new vs. carry-forward findings."

### 11. Web review surface (extends v0.3)
The FastAPI + HTMX app shell shipped in v0.3. #11 adds the reviewer-specific routes once #9 lands: read-only mode for stakeholders, reviewer mode for analysts who can set `reviewer_verdict`.

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
- **Extractor under-tags informal definitions (2026-06-08).** On a hand-built test corpus the live Moonshot extractor classified plain `"Quorum means a majority of the directors…"` sentences as `kind=claim` (term `None`), not `kind=definition` — so the **default definition-inconsistency detector had no definition pairs to compare and surfaced nothing**, even though the two docs defined the same term incompatibly. The contradictory *claims* were still catchable by the pairwise detector (NLI gate), but that is opt-in. Worth checking whether the extraction prompt should recognize the bare `X means Y` definitional pattern (the fixture extractor's tests assume it does; the real model is more conservative). Affects the headline detector's recall on informally-phrased definitions.
- **Stats drawer timestamp ordering looks wrong (2026-06-08).** The web Stats drawer rendered a run as "Started …T12:08:21 · finished …T07:08:22" — the start displays *after* the finish. Almost certainly a timezone/format mismatch between the two values (one local, one UTC, or one formatted differently). Cosmetic but confusing; reconcile the two timestamps to a single zone/format in the stats template / `_render_stats`.
- **Truncated (chunk-split) definitions still reach the judge (2026-06-08).** The `is_definitional` gate now drops reference/cross-ref mis-tags, but long definitions split across chunks still get extracted as a partial head, so the detector compares a truncated fragment against the full definition and the judge can read it as a divergence. A naive "one side is a prefix of the other → drop" heuristic was tried and rejected: it can't distinguish a chunking truncation from a *genuine append-style amendment* (B = the original definition + a new material clause), which is exactly a finding we want. Fix at the source (don't split a definition across chunks — the chunker could keep a `"Term" means …` clause whole) or let the judge decide with the truncation flagged in the prompt. Until then, the human labeler skips truncated candidates in the eval set.

### Architecture/debug audit triage (2026-06-07) — deferred items

A 10-finding architecture audit was triaged before the PyPI release. The
safe, low-risk batch shipped on branch `fix/pre-pypi-hardening` (empty
`choices[]` guard across all 7 provider/extractor sites; narrowed the bare
`except` in `_warn_if_model_download_needed` to debug-log unexpected errors;
replaced an `assert file.filename` with a `_require_filename` 400-guard that
survives `python -O`). Two findings were **false positives** and need no
work: the CLI *does* handle `CostCeilingExceeded` cleanly
(`cli/main.py` `check`), and a CLI cost-ceiling test already exists
(`tests/test_cli.py::test_check_max_cost_flag_aborts_when_exceeded`).

The rest were deferred as too large/risky for a pre-release batch:

- **Ingest logic duplication (#1).** `pipeline.ingest()` and
  `web.app._ingest_uploaded_paths()` mirror each other (~30 lines; the
  docstring even says "Mirrors pipeline.ingest"). Extract a shared core that
  takes a `LoadedDocument` iterator + optional `progress_cb`, so the
  directory-walk path and the explicit-paths path stop drifting. Refactor
  with regression risk — do it test-first, on its own branch.
- **Provider-factory dispatch repeated 4× (#3).** `make_judge`,
  `make_multi_party_judge`, `make_definition_judge`, `make_extractor` each
  re-switch on `judge_provider`. A provider-registry map would collapse them,
  but it touches every judge-construction path — pure churn risk, low value.
  Only worth it alongside a new provider (#12 local-LLM).
- **No rollback on partial ingest (#6).** `_ingest_uploaded_paths` commits
  documents/assertions incrementally; a mid-loop loader failure leaves a
  half-ingested corpus. Content-hash idempotency already makes re-ingest
  safe, so this is a UX nicety, not a correctness bug. Wrapping the loop in a
  single transaction is a behavior change worth its own design pass.
- **`store._conn` accessed directly across web routes (#7).** 20+ sites in
  `web/app.py` reach into the private SQLite connection instead of a public
  `AssertionStore` API. Real encapsulation smell, pre-existing, no functional
  bug. Large mechanical refactor — pairs naturally with #15 (bulk-fetch
  helpers on `AssertionStore`).
- **OCR fallback not covered in CI (#9).** The `hi_res` path is `slow`-marked
  because it needs the model download, so default CI skips it. A small mocked
  test of the escalation decision (without the real OCR engine) would close
  the gap cheaply.

## Completed

(Move items here as they ship, keep a one-line note on which release.)

- **UI collapse: single-page shell replaces the 7-tab UI (ADR-0017, Phase 6, 2026-06-02)**
  Branch: `feat/ui-collapse-phase-6`. The ADR-0017 single-page shell
  (`cc_single.html` — sidebar + findings + slide-over drawers + inline
  verdicts + cost gauge) is now the default `GET /` response; the legacy
  7-tab UI (`cc_base.html` + the `cc_contradictions`/`cc_ingest`/`cc_process`/
  `cc_action_items`/`cc_documents` templates, the pair/multi-party diff
  partials, and the `POST /uploads`, `GET /documents/{id}`, diff, and
  `POST /runs` routes) was deleted. `?legacy=1` and every `/tabs/*` path now
  return 410 Gone. Security dividend: the dead `consistency_checker/web/api/
  runs.py` module (sole consumer was the deleted `cc_process.html` SSE view)
  was removed. Closes two ADR-0011 gaps tracked in MEMORY: the **Action Items
  query** (the never-wired `/tabs/action_items` placeholder is gone — its role
  is served by the shell's filter chips) and **`/uploads` unification** (the
  single corpus-scoped upload path is now `POST /corpora/new`). Still open:
  **Findings CSV export** (item 9b above) and the `api/corpora.py`/
  `findings_router`/`models/ui.py` decommission (separate follow-up).

- **Pre-flight cost ceiling + provider-aware estimate-cost defaults (commercial blocker #2, 2026-05-31)**
  Branch: `feat/max-cost-ceiling`. ADR-0016. Plan:
  `docs/superpowers/archive/plans/2026-05-31-max-cost-ceiling.md`.
  `Config.max_cost_usd: float | None` defaults to `None`; `--max-cost <USD>`
  on `check` runs `estimate_cost()` as a pre-flight and raises
  `CostCeilingExceeded` (CLI exit 2) BEFORE any NLI or judge bootstrap when
  `est_cost_high > max_cost_usd` — no spend, no model download on an
  over-budget run. New `pipeline.default_per_call_costs(judge_provider)`
  returns `(0.003, 0.010)` for anthropic/openai, `(0.0001, 0.001)` for
  moonshot, `(0.0, 0.0)` for fixture; `estimate_cost()` and the `check`
  pre-flight both default `per_call_low/high` from the provider when
  callers don't pass explicit overrides. CLI `estimate-cost` output widened
  to four decimals so Moonshot's sub-cent numbers don't round to `$0.000`.
  Conservative gate (uses `est_cost_high`, not `est_cost_low`) — false
  positives possible, never false negatives. Resolves the "Moonshot
  projections off by 1–2 orders of magnitude" and "no automatic ceiling"
  failure modes recorded in ADR-0016's Context.

- **Pairwise contradiction detector flipped to opt-in (2026-05-31)**
  Branch: `feat/pairwise-opt-in`. ADR-0015. Plan:
  `docs/superpowers/archive/plans/2026-05-31-pairwise-opt-in.md`.
  `Config.pairwise_enabled` defaults to `False`; tri-state CLI
  `--pairwise / --no-pairwise` on `check` and `estimate-cost`; pipeline
  and web entrypoints skip the NLI model load entirely when pairwise is
  off (no ~800 MB download, no ~600 MB RSS, `max_memory_mb` pre-flight
  skipped); `--deep` without `--pairwise` rejected as a config error;
  `estimate_cost` reports `n_candidate_pairs=0` when pairwise is off.
  Resolves the "low-yield on legal/contract prose" finding recorded in
  the Eval findings & next levers (2026-05-21) section above. Code path
  retained — flipping the default back is one line if a future eval on a
  numeric/spec corpus reverses the call.

- **OCR fallback for scanned PDFs (2026-05-31)**
  Auto-escalates fast-strategy PDF extraction to hi_res (Tesseract via
  unstructured) when the extracted text is near-empty on a multi-page
  non-trivial PDF. Disable with `--no-ocr` / `ocr_enabled: false`.
  Spec: inline in `docs/superpowers/archive/plans/2026-05-31-ocr-fallback.md`.
  ADR-0014. Resolves the Atkins-corpus silent-drop failure mode recorded
  in `project_atkins_corpus_2026-05-30.md`.

- **Corpus isolation (item: retention gap, 2026-05-25)**
  Spec: `do../superpowers/archive/specs/2026-05-25-corpus-isolation-design.md`.
  Plan: `docs/superpowers/archive/plans/2026-05-25-corpus-isolation.md`.
  ADR-0013. Migration 0014 (additive). --corpus required on ingest /
  check / estimate-cost / export / store reidentify-orgs (interactive
  picker on TTY, hard error in scripts). FAISS gate post-filter so
  the shared index can't leak cross-corpus pairs. Companion archive
  spec deferred (see below).

- **Corpus-composition warning + opt-in org grouping (item #2, 2026-05-24)**
  Spec: `docs/superpowers/archive/specs/2026-05-24-corpus-org-warning-design.md`.
  Plan: `docs/superpowers/archive/plans/2026-05-24-corpus-org-warning.md`.
  ADR-0012. Default-on advisory warning; `--org-scope` suppression with
  audit-logged `findings.suppressed=1` rows.
  Measurement (§9 of the spec): divergent-rate delta on the bylaws corpus =
  `<fill in after the post-ship rerun>` — placeholder filled by a follow-up
  commit once the post-ship rerun is collected.

- **Definition-judge identical-text short-circuit + prompt tightening (item #1, 2026-05-21)**
  — deterministic `definitions_equivalent` short-circuit at the checker layer
  (mirrors `_try_numeric_short_circuit`, ADR-0005) emitting a distinguishable
  `definition_consistent_auto` verdict (no migration; free-text `judge_verdict`);
  tightened `definition_judge_system.txt`; `n_definition_short_circuited` in
  `CheckResult`; labeled regression set under `benchmarks/definition_eval/`.
  Provenance: PR #62 and `docs/decisions/0005-numeric-short-circuit.md`.
  **Deferred:** the `canonicalize_term` rewrite (gated on eval showing real
  distinct-term over-merge; use a recall-safe casefold key, NOT case-sensitive)
  and alias-aware grouping. Item #2 (org grouping) resumes next.
- **PDF extraction junk filter (PR #61, 2026-05-21)** — deterministic, conservative, audited two-stage filter (`consistency_checker/corpus/junk_filter.py`): `is_junk_line` drops TOC dot-leaders / page-numbers / non-alpha rows in `UnstructuredLoader`; `is_junk_assertion` drops cross-reference / near-empty "definitions" via a `JunkFilteringExtractor` wrapper in `make_extractor`; gated by `config.junk_filter_enabled` (default on); drops audited to `<data_dir>/junk_drops.jsonl`. Spec + plan in `docs/superpowers/`. Measured effect: see "Eval findings & next levers (2026-05-21)" above.
- **Moonshot (Kimi) provider end-to-end + safe `.env` loading (PR #60, 2026-05-21)** — `config.yml` defaults to `judge_provider: moonshot`, `kimi-k2.6`; `MoonshotExtractor` + `MoonshotDefinitionProvider` so extraction, judge, and definition-judge all run on one Kimi key; `load_local_env()` reads `MOONSHOT_API_KEY` from a gitignored `.env` at CLI/web startup. Extraction disables Kimi reasoning (`thinking:disabled`, ~50× faster; the `moonshot-v1-*` models can't do structured output). Bundled the API-500 info-disclosure hardening (generic detail messages across 12 endpoints).

- **f4-fixups** — `AuditLogger.most_recent_run()` + CLI private-attr fix (already done on branch); `FaissStore.open_or_create` made `dim`-optional (reads from existing index header); `_run_check_in_background` no longer loads the ~800 MB embedder model for check runs; `pipeline.check` now requires a pre-created `run_id` — callers own `begin_run` (items #16, #18, and the `main.py:140` known issue).

- **v0.1.0** — full 17-step build plan: ingest → chunk → atomic-fact extraction → embed → gate → NLI → judge → audit → report → CLI → CONTRADOC harness → CI. See `CHANGELOG.md`.
- **v0.2.0** — Block D (PDF/DOCX loaders via `unstructured`), Block E (numeric short-circuit + range-overlap hint), Block F (three-document conditional contradictions via graph triangles, `--deep` flag). See ADR-0004, ADR-0005, and ADR-0006.
  - Item #1 (PDF/DOCX) — D2 / ADR-0004.
  - Item #3 (numeric extractor) — E1–E3 / ADR-0005.
  - Items #6 partial (three-doc graph-triangle half) — ADR-0006. Cluster-by-entity second pass still pending in v0.4.
- **v0.3.0** — Block G web UI: FastAPI + HTMX, Contradictions / Documents / Assertions / Stats / Ingest tabs, Diff partials, HTMX self-polling, `consistency-check serve --open` CLI command, `run_status` migration + `BackgroundTasks` for check. See ADR-0007.
  - Output naming helper (`audit/naming.py`) + optional `--out` on `report` / `export` (G0b).
  - simplify pass: `RunStatus` Literal type, closure-captured overrides, dead-code cleanup.
- **v0.4 (definition-inconsistency detector)** — flavor A of item #20: divergent definitions of the same canonical term across the corpus. New migrations `0007_assertion_kind.sql` (`assertions.kind/term/definition_text`) and `0008_finding_detector_type.sql` (`findings.detector_type`); definitions extracted alongside atomic facts via the existing tool-use schema; new `DefinitionChecker` skips the NLI gate in favour of canonical-term grouping; new `DefinitionJudge` provider surface (Anthropic + OpenAI); audit, report, web UI, and CLI extended; `--no-definitions` opt-out. ADR-0009. Flavor B (definition ↔ usage) parked as item #20a.
- **v0.4 (reviewer workflow, Phase A)** — item #9: inline verdict buttons on Contradictions / Definitions / Cross-document tabs; content-keyed `reviewer_verdicts` table (migration 0009) keyed by `(pair_key, detector_type)` so verdicts survive re-runs; hide-by-default with per-section "Show reviewed" toggle; persistent undo toast (no auto-dismiss); keyboard shortcuts C/F/D when a finding row has focus; markdown report filters `false_positive` and tags surviving findings with `**Reviewer:** Real issue/Dismissed/Pending review`. Phase B (dedicated queue, note column UI, findings CSV) parked as item #9b.
