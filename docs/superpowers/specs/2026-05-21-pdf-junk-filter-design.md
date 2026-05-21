# PDF Extraction Junk Filter — Design

**Date:** 2026-05-21
**Status:** Approved (brainstorming) — pending implementation plan
**Author:** Claude Code session (with Todd Erickson)

## Problem

Running the definition-inconsistency detector on real nonprofit-bylaws PDFs produced an
84% "divergent" rate that, on inspection, was mostly noise rather than real conflicts.
Two distinct junk sources, observed in `data/eval/nonprofit_definition_findings.jsonl`:

1. **Dirty extracted text** — `unstructured` mislabels table-of-contents dot-leaders
   (`Expenses ........ 15`), bare page numbers, and similar formatting as body
   `Title`/`Text` elements, so they survive the loader's existing element-type filter
   and become chunks → assertions.
2. **Junk "definitions"** — the extractor turns cross-references into definitions
   (e.g. term `Agent` → `"as defined in this Article 11"`), and captures near-empty /
   dot-leader fragments as assertions.

These pollute the assertion store and inflate false-positive findings. The fix is a
deterministic junk filter applied at two pipeline stages.

## Non-goals (YAGNI)

- No ML/LLM-based junk classification — junk here is structural, so deterministic
  heuristics are faster, free, reproducible, and testable.
- No per-document tuning UI or configurable rule sets beyond a single on/off flag.
- No change to the loader's existing element-type allowlist (`BODY_TYPES`), which
  already drops `Header`/`Footer`/`PageBreak`/`Footnote`.
- Not a standalone package or Claude skill — an in-repo module with a clean,
  dependency-light interface that can be lifted to a package later if reuse materializes.

## Architecture (Approach A: pure predicates + two shared seams)

A new module exposing two **pure** functions and a small audit sink, wired at the two
seams every ingest path already funnels through (the loader and `make_extractor`), so
CLI, web upload, and any headless path inherit both filters with no per-path code.

```text
load_path / UnstructuredLoader ──(text seam)──> drop junk body elements
        │
        ▼
   chunk_document
        │
        ▼
make_extractor → JunkFilteringExtractor(inner) ──(assertion seam)──> drop junk assertions
        │
        ▼
   store.add_assertions
```

### Module: `consistency_checker/corpus/junk_filter.py`

Pure, stdlib-`re` only, no project imports (keeps it reuse-ready):

```python
def is_junk_line(text: str) -> str | None: ...
def is_junk_assertion(text: str, *, kind: str) -> str | None: ...
```

- Returns a **reason string** when the input is junk, `None` when clean.
- **Total functions:** never raise; on unexpected input return `None`
  (**fail-open** — a filter bug must never delete real content).
- Returning the reason (not a bool) powers the audit trail and later tuning.

### Junk taxonomy (conservative; each rule traces to an observed example)

`is_junk_line` (operates on one `unstructured` body element's text):
- `dot_leader` — title text followed by ≥5 dot-leaders, optional trailing page number
  (`...........`, `Expenses ........ 15`).
- `page_number` — line is only a number / `Page N` / `- N -`, no alphabetic content.
- `mostly_non_alpha` — below ~15% alphabetic characters over a minimum length
  (catches dot/underscore rows).

`is_junk_assertion` (operates on an extracted assertion/definition):
- `cross_reference` — starts with `as defined in|as set forth in|as described in|`
  `as provided in|see ` and points at `Article/Section/Bylaws/herein/above/below`
  with no substantive clause (`"as defined in this Article 11"`).
- `dot_fragment` — same non-alpha test as `mostly_non_alpha`.
- `near_empty` — fewer than 10 alphabetic characters.

Thresholds are deliberately high-confidence. The audit log is the mechanism for
loosening/tightening later, rather than guessing exact thresholds now.

### Audit sink

- Each drop appended to `<data_dir>/junk_drops.jsonl`:
  `{ts, stage, reason, doc_id, text_snippet}`.
- INFO-level summary count emitted per load / per ingest.
- Audit-write failures log a WARNING and never abort ingest.

### Wiring

- **Text seam:** `UnstructuredLoader.__init__` gains `drop_junk_lines: bool = True`
  and an optional audit sink. In the existing element loop, after the `BODY_TYPES`
  check, `if is_junk_line(element_text): record drop; continue`. The char-span
  invariant (`text[chunk.char_start:chunk.char_end] == chunk.text`) is preserved
  because `full_text` is built only from kept elements.
- **Assertion seam:** `JunkFilteringExtractor(inner: Extractor, audit)` — defined in
  `consistency_checker/extract/atomic_facts.py` alongside the `Extractor` protocol and
  the other extractors — implements the `Extractor` protocol; `extract()` calls
  `inner.extract()` then drops assertions where `is_junk_assertion` fires, recording
  each. `make_extractor` wraps the constructed Anthropic/Moonshot extractor (not
  `FixtureExtractor`).
- **Config:** `junk_filter_enabled: bool = True` added to `Config` so the filter can be
  A/B-tested or disabled.

## Error handling

- Predicates are total and fail-open (`None` on anomaly).
- Audit-write errors are caught, logged at WARNING, and swallowed — ingest proceeds.
- No new external dependency.

## Testing (all hermetic — no API/model calls)

- **Predicate unit tests:** for every reason, a junk example (drawn from the real
  findings) AND a clean counter-example that must NOT be dropped (false-positive guard):
  e.g. a genuine short definition, a clause containing the word "Section", a numbered
  list item.
- **Loader test:** a junk element is dropped while a real one is kept, and the
  `text[char_start:char_end] == chunk.text` invariant still holds.
- **`JunkFilteringExtractor` test:** junk assertions dropped, clean ones kept, drops
  recorded to the audit sink.
- **Config test:** `junk_filter_enabled=False` disables both seams.

## Files touched

- New: `consistency_checker/corpus/junk_filter.py`, `tests/test_junk_filter.py`.
- Edit: `consistency_checker/corpus/loader.py` (text seam),
  `consistency_checker/pipeline.py` (`make_extractor` wrap),
  `consistency_checker/config.py` (flag),
  `consistency_checker/extract/atomic_facts.py` (home for `JunkFilteringExtractor`,
  or a new small file — decide in the plan).

## Reuse path

Because `is_junk_line` / `is_junk_assertion` are pure and import-free, lifting them into
a standalone package later is a file move plus a packaging shim — no rework of callers.
