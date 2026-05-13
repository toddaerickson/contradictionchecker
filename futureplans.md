# Future plans

Forward-looking work beyond the v0.1.0 release. Items are sized roughly; pick from the top when starting a new milestone. When work is finished, move the entry to the **Completed** section below rather than deleting it — provenance matters when revisiting decisions.

## v0.2 — fill the obvious gaps

These were explicitly out of scope in the 17-step plan but are the highest-leverage additions. A detailed PR-by-PR build sequence lives in [`docs/plans/v0.2-build-plan.md`](docs/plans/v0.2-build-plan.md); summaries below.

### 1. PDF and DOCX loaders
Replace the `NotImplementedError` stubs in `consistency_checker/corpus/loader.py`.
- PDF: `pypdfium2` for layout-preserving text, fall back to `pdfminer.six` for tricky files. Preserve page numbers as `metadata_json` on the `Document` row.
- DOCX: `python-docx` for body text; ignore headers/footers/footnotes for v0.2.
- Both must maintain the char-span round-trip invariant (`text[chunk.char_start:chunk.char_end] == chunk.text`).
- Test fixtures: one short PDF, one short DOCX in `tests/fixtures/sample_docs/`.

### 2. Real chunk overlap
The chunker currently raises `NotImplementedError` when `overlap_chars > 0`. Implement a sliding-sentence window: after emitting a chunk, retain the last sentences whose total length is up to `overlap_chars` and use them as the start of the next chunk. Keep the char-span invariant. Add a property-style test that the union of overlapping chunks covers every character at least once.

### 3. Numeric / quantitative extractor
NLI models routinely miss "grew 12%" vs. "declined 5%" as a contradiction unless the wording is unambiguous. A small auxiliary extractor (`consistency_checker/extract/quantitative.py`?) that pulls `(metric, value, unit, polarity, scope)` tuples out of each assertion would let us run a deterministic check before falling back to the LLM judge. Specifics:
- Regex + `pint` for units (`%`, `million`, `bps`).
- Same-scope tuples with opposite-sign polarity → automatic contradiction with `confidence=1.0`, skip the judge.
- Same-scope tuples with adjacent ranges → flag as `uncertain` and pass to the judge with the numeric mismatch in the prompt context.

### 4. OpenAI atomic-fact extractor
`AnthropicExtractor` exists; the OpenAI counterpart was deferred. The Anthropic implementation in `consistency_checker/extract/atomic_facts.py` is the reference shape; mirror it in `OpenAIExtractor` using `client.beta.chat.completions.parse` with a Pydantic schema.

### 5. CONTRADOC fetch + cache helper
Today `benchmarks/contradoc_harness.py` expects a pre-normalised JSONL. Add a thin loader that:
- Fetches the original CONTRADOC release from the canonical source.
- Normalises to our JSONL shape and caches under `~/.cache/contradoc/`.
- Skips re-download if the cached file exists. SHA-256 verification on the download.
- Exposes `python -m benchmarks.contradoc_harness fetch` as a subcommand.

## v0.3 — precision and provenance

Higher-effort items that require more design discussion.

### 6. Three-document conditional contradictions
The current pipeline is strictly pairwise. CONTRADOC includes cases where assertions A, B, C are individually consistent but `A ∧ B ⇒ ¬C`. Likely shape:
- After Stage B, cluster contradiction candidates by shared entities.
- For each entity cluster, prompt the judge with up to N assertions plus a structured question ("do these jointly contradict any third assertion in the cluster?").
- Keep the pairwise pipeline for performance; this becomes an opt-in `--deep` flag.

### 7. Entity resolution pass
Document A says "the Borrower", Document B says "ABC Corp". The current judge can sometimes infer they're the same; often it can't. A lightweight NER pass (`spaCy` `en_core_web_lg`) plus a fuzzy-match canonicaliser would let us pass `entity_ids` into the judge prompt, raising precision on entity-coreference contradictions. Audit findings should record the resolved entity id so reports can group by entity.

### 8. LanceDB migration option
SQLite + FAISS works at corpus sizes up to ~1M assertions. Beyond that, LanceDB consolidates store + vectors with native filtering. Worth prototyping as an alternative `Store` Protocol so users can choose. ADR-0002's "domain model" alternative path goes here too — once LanceDB is in place, domain embedders are configurable rather than code-bound.

### 9. Reviewer workflow
The audit DB has `findings.reviewer_verdict` columns reserved (or should — verify the schema). Build a tiny TUI (or web view) that walks reviewers through unreviewed findings and lets them set `reviewer_verdict` ∈ `{confirmed, false_positive, dismissed}`. Findings that reviewers confirm as false positives should feed back into prompt iteration.

## v0.4 — operational

### 10. Incremental scans
Today every `check` invocation re-scans the whole corpus. If only one new document was added since last run, only its assertions need to be paired against pre-existing ones. Use the `documents.ingested_at` timestamp to compute a "new assertions" set, then gate against the old set without re-pairing the old set with itself. Audit logger gains a `prior_run_id` link so report can show "new vs. carry-forward findings."

### 11. Web review surface
Once #9 lands, wrap it in a FastAPI + a few HTMX templates. Read-only mode for stakeholders; reviewer mode for analysts.

### 12. Local-LLM judge provider
For air-gapped deployments: a `LocalLlamaProvider` against `llama.cpp` or vLLM. The `JudgeProvider` Protocol already exists — this is mostly prompt tuning and tooling, not architecture. Document the quality trade-off in a new ADR.

### 13. Metrics dashboard
The audit DB already records latency, token counts, and verdict distribution per run. A simple Streamlit page that reads the DB and plots:
- Token spend per run.
- Verdict distribution over time.
- Precision tracking against the rolling CONTRADOC benchmark.

## Known issues to fix opportunistically

- `consistency_checker/cli/main.py:140` reaches into `store._conn` to fetch the most recent run id. Add a `AuditLogger.most_recent_run()` method so the CLI doesn't touch a private attribute.
- The benchmark harness bypasses Stage 7 (atomic-fact extraction). Worth surfacing in `docs/benchmarks.md` more loudly so users don't compare CONTRADOC F1 to "full pipeline F1" and get confused.
- v0.1.0 ships with `embedder_model: "hash"` accepted by config validation only because Pydantic doesn't enforce the model name. A `Literal[...]` would catch typos at config-load time, but it would also lock out user-supplied HF model ids. Decide whether validation should be strict or permissive — currently permissive by accident.

## Completed

(Move items here as they ship, keep a one-line note on which release.)

- **v0.1.0** — full 17-step build plan: ingest → chunk → atomic-fact extraction → embed → gate → NLI → judge → audit → report → CLI → CONTRADOC harness → CI. See `CHANGELOG.md`.
