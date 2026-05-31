# ADR 0014 — OCR fallback for scanned PDFs

**Status**: Accepted

## Context

ADR-0004 wired `unstructured` in with `strategy="fast"` as the default — a rule-based, model-free path that keeps ingest hermetic and cheap. The escape hatch flagged at the time was that image-only PDFs would need `strategy="hi_res"` and a model download; the assumption was that users would notice and opt in.

The Atkins corpus run on 2026-05-30 falsified that assumption. Three of five PDFs were scanned image dumps; `strategy="fast"` returned `""` for each without raising, so the loader handed empty text to the chunker, which produced the empty-content sentinel and moved on. The user only noticed because the resulting assertion count was implausibly low for the page count. There was no log line, no warning, no audit row — the failure was indistinguishable from "this PDF really is blank."

For the legal/contract corpora this project is most likely to see — recorded deeds, scanned bylaws, ordinance amendments captured from microfilm — the population of image-only PDFs is plausibly 30–60% of any given corpus. A silent skip on those documents corrupts every downstream measurement (corpus composition, judge cost, finding counts) and is not acceptable as the default.

Alternatives considered:

- **Always use `strategy="hi_res"`**. Forces the DeBERTa-class layout-and-OCR model load on every ingest, including text-native PDFs that don't need it. 10–100× slower per page. Rejected — punishes the common case to fix the uncommon one.
- **OCRmyPDF as a separate preprocessing step**. Runs OCR up front, produces a text-layered PDF, then hands the result to the existing fast loader. Keeps the loader untouched and parallelizes well for overnight batches. Trade-off is a second process invocation, an intermediate file, and a new step in the operator's runbook. Deferred — auto-escalation inside the loader is a smaller surface change and doesn't require teaching users a new command.
- **Cloud OCR (AWS Textract, Google Vision, Azure Form Recognizer)**. Highest accuracy on degraded scans. Contradicts the local-first framing — documents would leave the operator's machine — and introduces per-page billing the rest of the pipeline doesn't have. Out of scope.

## Decision

Auto-escalate `strategy="fast"` to `strategy="hi_res"` inside `UnstructuredLoader` when a pure detection predicate decides the fast result is empty in a way that suggests an image-only source. Implementation notes:

- The predicate lives in `consistency_checker/corpus/ocr.py::needs_ocr` and takes the extracted text, the PDF page count, and the on-disk file size — all primitives, kwargs-only. It returns `True` only when fast extracted fewer than 100 alpha characters AND the PDF has ≥ 2 pages AND the file is ≥ 100 KB on disk. The single-page and tiny-file guards exist so legitimate cover sheets and placeholder PDFs don't pay the slow path. Pure and fixture-free, so it unit-tests with primitive inputs.
- Escalation is **one-shot**. If the `hi_res` retry also returns empty, the loader records the failure and proceeds with the empty result — never a loop, never a third strategy.
- Escalation happens inside `consistency_checker/corpus/loader.py::UnstructuredLoader` so callers (the registry, ingest, tests) see the same loader interface. The element list returned to the chunker is the post-escalation list; `full_text` is rebuilt from that list, so the char-span invariant (`text[chunk.char_start:chunk.char_end] == chunk.text`) survives by construction.
- A new `Config.ocr_enabled: bool = True` flag and a `--no-ocr` CLI flag let operators opt out for the "I know this corpus is text-only and I never want a model download" case. Default-on because the silent-skip failure mode is worse than a one-line "OCR engaged" warning.
- Every escalation writes an audit row to `<data_dir>/ocr_events.jsonl` via `OcrAudit` (also in `ocr.py`). The row records a UTC timestamp, the event kind (`escalated` when the hi_res retry fires; `ocr_failed` when hi_res returned near-empty; `ocr_error` when the hi_res call itself raised — Tesseract missing, model-download failure, OOM, decode error), the source path, and the PDF page count. The hi_res call is wrapped in a broad `except Exception` so a missing system Tesseract or a corrupt image PDF can never abort an ingest walk mid-corpus — it records `ocr_error`, logs a warning pointing to the Tesseract install path, and continues with the fast-pass empty result. This is the surface that lets a future reviewer answer "did OCR fire on this document, and did it recover anything?" without re-running ingest. Element counts and wall-clock duration are deliberately not recorded — the file/event/page-count tuple is enough to triage failure modes; richer telemetry can land in a follow-up if anyone needs it.
- The DeBERTa OCR model download is gated by use — pulled only the first time `hi_res` runs. A one-line warning prints before the download starts so the user understands why the first OCR'd document takes ~5 minutes longer than the rest.

System Tesseract becomes a **soft dependency**: `unstructured`'s `hi_res` path shells out to it. The package is documented in README and CORPORATE_SETUP as required only when OCR escalation is enabled; the hermetic test suite (`pytest -m "not slow and not live"`) does not depend on it because escalation paths are exercised via the `FixtureExtractor`-style seam.

The implementation plan that produced this decision is at [`docs/superpowers/plans/2026-05-31-ocr-fallback.md`](../superpowers/plans/2026-05-31-ocr-fallback.md).

## Consequences

- Image-only PDFs ingest content instead of collapsing into the empty sentinel. The Atkins-style "3 of 5 PDFs silently empty" failure mode is closed at the loader layer.
- First OCR run on a fresh environment downloads ~500 MB of model weights. A one-line warning fires before the download starts so users don't think the CLI is hung — same pattern as the DeBERTa NLI download on first `check` run.
- System Tesseract becomes a soft dependency. Documented in the README install section and in CORPORATE_SETUP. Hermetic CI does not require it because the OCR path is reachable only through `strategy="hi_res"` and is exercised in tests via fixtures.
- The char-span invariant is preserved because `UnstructuredLoader` rebuilds `full_text` from the post-escalation element list before handing off to the chunker. The same concatenation rules from ADR-0004 apply unchanged.
- A new per-corpus audit artifact appears at `<data_dir>/ocr_events.jsonl`. Append-only, line-delimited JSON, safe to `tail -f` during long ingests.
- One extra runtime config knob (`ocr_enabled`) and one extra CLI flag (`--no-ocr`). Both default to the safe-for-the-common-case behavior; nobody who wasn't paying attention has to change anything.
- OCRmyPDF preprocessing remains an option for very large corpora where overnight batch OCR makes more sense than per-ingest escalation. Revisit if anyone runs a corpus where escalation latency dominates the run; the loader-level escalation does not preclude that path.
