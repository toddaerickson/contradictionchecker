# ADR 0004 — `unstructured` as the PDF/DOCX loader backend

**Status**: Accepted

## Context

v0.1.0 ships only `.txt` and `.md` loaders. The `STUB_EXTENSIONS` set in `consistency_checker/corpus/loader.py` lists `.pdf` and `.docx` and raises `NotImplementedError` for both. v0.2 Block D fills in those loaders.

Candidates considered:

- **`pypdfium2` + `python-docx`** — two separate dependencies, one per format. Both are small, native-bound, and give precise character offsets. Two registry entries, two test fixtures, two char-span bookkeeping paths.
- **`unstructured`** — one dependency, one `partition()` entry point, one registry entry. Handles `.pdf`, `.docx`, `.html`, `.epub`, and a long tail of other formats out of the box. Trade-off: heavy install footprint (pulls in pdfminer, magic, optional OCR / layout inference models).
- **`pdfminer.six` + `python-docx`** — pure-Python both. Slower than pypdfium2 on PDF; char offsets are noisier on tricky layouts.

The corpora the project is most likely to see — board minutes, contracts, financial filings — frequently arrive in PDF and DOCX. The marginal value of supporting `.html` / `.epub` later via the same library (rather than adding another dep per format) is significant.

## Decision

Use **`unstructured`** as the single loader backend for non-plaintext formats. Implementation notes:

- Call `unstructured.partition.auto.partition(file_path)` and accept the resulting `list[Element]`.
- Concatenate `element.text` for body-content element types only — `NarrativeText`, `Title`, `ListItem`, `Table`. Skip `Header`, `Footer`, `PageBreak`, `Footnote`, and other navigational element types. Deferring those keeps v0.2 scope tight; they can be lifted in v0.3 if a reviewer asks.
- Element separator is `"\n\n"` between elements in document order.
- `Table` elements are flattened **row-major**, `"\t"` between cells and `"\n"` between rows. This matches what a hand-rolled `python-docx` loader would have done and keeps report goldens diffable.
- Build a sidecar `[(element_index, element_type, char_start, char_end), …]` list and store it as JSON in `documents.metadata_json["element_spans"]`. No new SQLite columns; the existing `metadata_json TEXT` is enough.
- Char-span invariant — `text[chunk.char_start:chunk.char_end] == chunk.text` — survives because the text the chunker sees is exactly the concatenation we built. Element-span offsets are computed against the same concatenated string.
- Use `strategy="fast"` (rule-based, no model inference) by default to keep the loader hermetic. Surface `unstructured`'s `strategy` parameter as a future config knob if a user needs OCR or layout-aware parsing; that path is opt-in and marked `slow`.

Install footprint is managed by pinning the minimal `unstructured` extras needed for `.pdf` + `.docx`. Resolve precisely after a `uv lock` pass; document the resolved set in `pyproject.toml` rather than the broad `unstructured[all-docs]`.

## Consequences

- One runtime dependency added (`unstructured`) instead of two. Two fewer registry entries and one fewer char-span bookkeeping path to test.
- The loader registry pattern (Step D1) routes both `.pdf` and `.docx` through the same `UnstructuredLoader` instance — minus the file-extension dispatch, the loader is one function.
- Future formats (`.html`, `.epub`, `.rtf`, `.odt`) become additive registry entries pointing at the same loader.
- Install weight grows. CI cache must include the `unstructured` wheel and its transitive deps; first cold install will be noticeably slower than today's. Document this in the README's Install section.
- Layout-aware / OCR parsing is not available in the v0.2 default code path. Users with image-only PDFs will need to swap `strategy="hi_res"` and accept the model download. ADR-0008 (or whatever lands after) will record that escape hatch when it's wired.
- If `unstructured` turns out to be too heavy in practice, the registry pattern means a swap to `pypdfium2` + `python-docx` is two replacement loaders behind the same interface — not a rewrite.
