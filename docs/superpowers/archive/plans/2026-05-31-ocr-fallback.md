# OCR fallback for scanned PDFs — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop silently dropping scanned-image PDFs during ingest. When the default `strategy="fast"` path extracts near-empty text from a PDF that clearly has content (multi-page, non-trivial file size), automatically escalate to `strategy="hi_res"` (OCR-capable) and surface what happened to the user.

**Why this matters:** the Atkins corpus run on 2026-05-30 silently lost 3 of 5 PDFs because pypdf returns `""` for scanned-image PDFs with no error. The user only noticed because the assertion count was suspiciously low. For the most likely commercial buyer (legal review of contracts + amendments), image PDFs are not an edge case — they are 30–60% of the corpus.

**Architecture:** a pure detection predicate (`needs_ocr(text, page_count)`) decides whether a fast-path extraction looks empty; a one-shot retry inside `UnstructuredLoader.__call__` re-partitions the same file with `strategy="hi_res"` when the predicate fires; an `OcrAudit` sink records which docs were escalated and which still came back empty; the existing `element_spans` sidecar continues to work because `unstructured` returns the same element shape regardless of strategy.

**Tech stack:** Python 3.11, `unstructured[pdf]>=0.15` (already a dep — `hi_res` strategy adds Tesseract + a layout model on first use; both are pulled by the `[pdf]` extra), pypdf for page count (already transitive), pytest.

**Out of scope (deferred to a follow-up plan):**
- OCRmyPDF as a preprocessing alternative (decouples OCR from ingest; better for very large corpora).
- Cloud OCR providers (Textract, Google Vision, Azure Form Recognizer) — contradicts the local-first framing; revisit if cloud becomes a hard requirement.
- Per-page progress bars during `hi_res` runs.
- `.docx` parity (DOCX never needs OCR; this plan is PDF-only).

**Spec:** inline below (no separate design doc — this is a focused, mostly-additive change).
**ADR:** to be drafted as `docs/decisions/0014-ocr-fallback.md` in Task 7 once the implementation is committed.

---

## Design summary

### Detection — when does fast-path "look empty"?

Three signals combine:

| Signal | Threshold | Why |
|---|---|---|
| Total alpha chars in extracted text | `< 100` | A multi-paragraph PDF page should always exceed this. |
| PDF page count | `>= 2` | Single-page PDFs can legitimately be near-empty (cover sheets); skip the heuristic. |
| File size on disk | `>= 100 KB` | Avoids escalating tiny placeholder PDFs. |

All three must trigger. Encoded as a pure function `needs_ocr(text: str, page_count: int, file_size: int) -> bool`. No project imports.

### Escalation — how does the loader retry?

`UnstructuredLoader.__call__` currently calls `partition(filename=..., strategy=self._strategy)` once. The new path:

1. Run the existing fast-path body.
2. After building `full_text`, evaluate `needs_ocr(full_text, page_count, file_size)`.
3. If `True` *and* `self._ocr_enabled`: re-run `partition(..., strategy="hi_res")`, rebuild `full_text` and `element_spans` from the new element list, and continue. Record the escalation in the audit.
4. If still near-empty after escalation: record an `ocr_failed` entry and let the document through with whatever text was extracted (could be empty). Caller decides what to do — `add_document` already has the cross-corpus duplicate guard; an additional "extracted to empty" log keeps the user informed without aborting ingest.

The escalation is one-shot — never a loop.

### Why not always use `hi_res`?

- `hi_res` is 10–100× slower than `fast` on text PDFs.
- `hi_res` downloads ~500 MB of layout-detection + OCR models on first use.
- `hi_res` requires system Tesseract (`tesseract-ocr` apt package on Debian/Ubuntu); fast-path has no system deps.

Auto-escalation gives the right cost/quality trade-off: pay the slow path only when the fast path failed.

### Char-span invariant

The existing invariant `text[chunk.char_start:chunk.char_end] == chunk.text` is preserved because the loader rebuilds `full_text` from the (new) element list using the same concatenation logic. The `element_spans` sidecar in `documents.metadata_json` is recomputed against the new text. Downstream chunker / extractor are unaffected.

### Failure modes the plan addresses

| Failure | Before this plan | After |
|---|---|---|
| Scanned PDF, fast returns "" | Silent — doc collapses to `sha256("")` sentinel row | Auto-escalate to hi_res; warn if still empty |
| Tesseract not installed | Cryptic unstructured stack trace | Caught at first use; clear error pointing to `apt install tesseract-ocr` |
| hi_res model download (~500 MB) | Surprise — looks hung | One-line warning before download starts |
| User wants to disable | Not possible | `--no-ocr` CLI flag + `ocr_enabled: false` config field |

### Failure modes deliberately left for a future plan

- A PDF that *partially* OCRs (some pages text, others image) — fast-path returns partial text; the heuristic only fires on near-empty totals. A page-level escalation pass is more complex and waits for evidence it matters.
- OCR quality auditing (the OCR'd text could be garbage). Out of scope; the existing junk filter catches most of the gibberish at the assertion stage anyway.

---

## Task 1: Detection predicate (pure function + audit sink)

**Files:**
- Create: `consistency_checker/corpus/ocr.py`
- Test: `tests/test_ocr.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ocr.py
"""Tests for the OCR detection predicate and audit sink."""

from __future__ import annotations

import json
from pathlib import Path

from consistency_checker.corpus.ocr import OcrAudit, needs_ocr


# --- needs_ocr: positive cases (image PDFs the heuristic must catch) --------
def test_needs_ocr_empty_text_multi_page_large_file() -> None:
    assert needs_ocr(text="", page_count=10, file_size=500_000) is True


def test_needs_ocr_short_text_multi_page_large_file() -> None:
    assert needs_ocr(text="Page 1\n\nPage 2", page_count=10, file_size=500_000) is True


# --- needs_ocr: negative cases (must NOT fire) ------------------------------
def test_needs_ocr_short_text_single_page_skipped() -> None:
    # single-page cover sheets are legitimately short
    assert needs_ocr(text="", page_count=1, file_size=500_000) is False


def test_needs_ocr_short_text_tiny_file_skipped() -> None:
    # placeholder PDFs under 100 KB don't deserve a slow re-pass
    assert needs_ocr(text="", page_count=10, file_size=10_000) is False


def test_needs_ocr_substantive_text_skipped() -> None:
    long_text = "The Board shall consist of no fewer than three Directors. " * 20
    assert needs_ocr(text=long_text, page_count=10, file_size=500_000) is False


# --- OcrAudit ---------------------------------------------------------------
def test_ocr_audit_records_escalation_and_failure(tmp_path: Path) -> None:
    audit = OcrAudit(tmp_path / "ocr_events.jsonl")
    audit.record(event="escalated", doc_id="docA", path="a.pdf", page_count=10)
    audit.record(event="ocr_failed", doc_id="docB", path="b.pdf", page_count=5)
    assert audit.counts == {"escalated": 1, "ocr_failed": 1}
    lines = (tmp_path / "ocr_events.jsonl").read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["event"] == "escalated" and rec["doc_id"] == "docA"


def test_ocr_audit_none_path_is_memory_only(tmp_path: Path) -> None:
    audit = OcrAudit(None)
    audit.record(event="escalated", doc_id=None, path="a.pdf", page_count=3)
    assert audit.counts == {"escalated": 1}


def test_ocr_audit_write_failure_is_swallowed(tmp_path: Path) -> None:
    bad = tmp_path / "afile"
    bad.write_text("x")
    audit = OcrAudit(bad / "nested" / "events.jsonl")
    audit.record(event="escalated", doc_id=None, path="a.pdf", page_count=3)  # no exception
    assert audit.counts == {"escalated": 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ocr.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'consistency_checker.corpus.ocr'`

- [ ] **Step 3: Write the module**

```python
# consistency_checker/corpus/ocr.py
"""Detection predicate + audit sink for the OCR-fallback path.

`needs_ocr` is a pure predicate that decides whether a fast-strategy
extraction looks empty enough to warrant a one-shot retry with
`strategy="hi_res"`. `OcrAudit` mirrors `JunkAudit`: in-memory counts plus
optional JSONL persistence, write failures swallowed so ingest never aborts
on telemetry trouble.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

_log = logging.getLogger(__name__)

_MIN_ALPHA_CHARS = 100
_MIN_PAGE_COUNT = 2
_MIN_FILE_SIZE = 100_000  # bytes


def _alpha_count(text: str) -> int:
    return sum(1 for c in text if c.isalpha())


def needs_ocr(*, text: str, page_count: int, file_size: int) -> bool:
    """True iff fast-path extraction looks empty for a non-trivial PDF.

    All three guards must trigger:
      * < 100 alpha chars in the extracted body text
      * >= 2 pages in the PDF (single-page PDFs are often legit-short)
      * >= 100 KB on disk (placeholder PDFs aren't worth a slow retry)
    """
    if page_count < _MIN_PAGE_COUNT:
        return False
    if file_size < _MIN_FILE_SIZE:
        return False
    return _alpha_count(text) < _MIN_ALPHA_CHARS


class OcrAudit:
    """Records OCR-fallback events: in-memory counts always, JSONL when path is set."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._counts: dict[str, int] = {}

    def record(
        self,
        *,
        event: str,
        doc_id: str | None,
        path: str,
        page_count: int,
    ) -> None:
        self._counts[event] = self._counts.get(event, 0) + 1
        if self._path is None:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "doc_id": doc_id,
            "path": path,
            "page_count": page_count,
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            _log.warning("ocr audit write failed (%s): %s", self._path, exc)

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ocr.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Lint, format, type-check**

Run: `uv run ruff check consistency_checker/corpus/ocr.py tests/test_ocr.py && uv run ruff format consistency_checker/corpus/ocr.py tests/test_ocr.py && uv run mypy consistency_checker/corpus/ocr.py`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add consistency_checker/corpus/ocr.py tests/test_ocr.py
git commit -m "feat: OCR detection predicate + audit sink"
```

---

## Task 2: Config field `ocr_enabled`

**Files:**
- Modify: `consistency_checker/config.py` (add field after `junk_filter_enabled`)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_config.py
def test_ocr_enabled_defaults_true(tmp_path: Path) -> None:
    yml = write_yaml(tmp_path / "c.yml", {"corpus_dir": str(tmp_path / "corpus")})
    cfg = Config.from_yaml(yml, env={})
    assert cfg.ocr_enabled is True


def test_ocr_enabled_can_disable(tmp_path: Path) -> None:
    yml = write_yaml(
        tmp_path / "c.yml",
        {"corpus_dir": str(tmp_path / "corpus"), "ocr_enabled": False},
    )
    cfg = Config.from_yaml(yml, env={})
    assert cfg.ocr_enabled is False
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config.py -k ocr_enabled -q`
Expected: FAIL (`ValidationError: extra fields not permitted`)

- [ ] **Step 3: Add the field**

In `consistency_checker/config.py`, immediately after `junk_filter_enabled`:

```python
    ocr_enabled: bool = Field(
        default=True,
        description=(
            "Re-run image-only PDFs with unstructured's hi_res (OCR) strategy "
            "when the fast strategy extracts near-empty text. Requires system "
            "Tesseract; first use downloads ~500 MB of layout/OCR models."
        ),
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_config.py -k ocr_enabled -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/config.py tests/test_config.py
git commit -m "feat: add ocr_enabled config flag (default on)"
```

---

## Task 3: `UnstructuredLoader` auto-escalation

**Files:**
- Modify: `consistency_checker/corpus/loader.py`
- Test: `tests/test_corpus.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_corpus.py
from consistency_checker.corpus.ocr import OcrAudit


@_skip_unstructured_on_win
def test_loader_escalates_to_hi_res_when_fast_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fast partition returns nothing; loader re-runs with hi_res and uses that text."""
    calls: list[str] = []

    def fake_partition(*, filename: str, strategy: str, **kwargs: object) -> list[object]:
        calls.append(strategy)
        if strategy == "fast":
            return []  # simulate image-only PDF
        return [_named("OCR'd narrative content recovered from images.", "NarrativeText")]

    import unstructured.partition.auto as auto

    monkeypatch.setattr(auto, "partition", fake_partition)
    monkeypatch.setattr(
        "consistency_checker.corpus.loader._pdf_page_count", lambda _: 10
    )
    pdf = tmp_path / "scanned.pdf"
    pdf.write_bytes(b"x" * 200_000)  # >= 100 KB so heuristic fires
    audit = OcrAudit(None)
    loaded = UnstructuredLoader(ocr_enabled=True, ocr_audit=audit)(pdf)
    assert "OCR'd narrative content" in loaded.text
    assert calls == ["fast", "hi_res"]
    assert audit.counts == {"escalated": 1}


@_skip_unstructured_on_win
def test_loader_does_not_escalate_when_fast_already_has_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def fake_partition(*, filename: str, strategy: str, **kwargs: object) -> list[object]:
        calls.append(strategy)
        return [_named("The Board shall consist of three Directors. " * 5, "NarrativeText")]

    import unstructured.partition.auto as auto

    monkeypatch.setattr(auto, "partition", fake_partition)
    monkeypatch.setattr(
        "consistency_checker.corpus.loader._pdf_page_count", lambda _: 10
    )
    pdf = tmp_path / "text.pdf"
    pdf.write_bytes(b"x" * 200_000)
    audit = OcrAudit(None)
    loaded = UnstructuredLoader(ocr_enabled=True, ocr_audit=audit)(pdf)
    assert "Board shall consist" in loaded.text
    assert calls == ["fast"]
    assert audit.counts == {}


@_skip_unstructured_on_win
def test_loader_records_ocr_failed_when_hi_res_also_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_partition(*, filename: str, strategy: str, **kwargs: object) -> list[object]:
        return []  # both passes return nothing

    import unstructured.partition.auto as auto

    monkeypatch.setattr(auto, "partition", fake_partition)
    monkeypatch.setattr(
        "consistency_checker.corpus.loader._pdf_page_count", lambda _: 10
    )
    pdf = tmp_path / "bad.pdf"
    pdf.write_bytes(b"x" * 200_000)
    audit = OcrAudit(None)
    loaded = UnstructuredLoader(ocr_enabled=True, ocr_audit=audit)(pdf)
    assert loaded.text == ""
    assert audit.counts == {"escalated": 1, "ocr_failed": 1}


@_skip_unstructured_on_win
def test_loader_disabled_never_escalates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def fake_partition(*, filename: str, strategy: str, **kwargs: object) -> list[object]:
        calls.append(strategy)
        return [] if strategy == "fast" else [_named("recovered", "NarrativeText")]

    import unstructured.partition.auto as auto

    monkeypatch.setattr(auto, "partition", fake_partition)
    monkeypatch.setattr(
        "consistency_checker.corpus.loader._pdf_page_count", lambda _: 10
    )
    pdf = tmp_path / "scanned.pdf"
    pdf.write_bytes(b"x" * 200_000)
    loaded = UnstructuredLoader(ocr_enabled=False)(pdf)
    assert loaded.text == ""
    assert calls == ["fast"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_corpus.py -k "escalates or does_not_escalate or ocr_failed or disabled_never_escalates" -q`
Expected: FAIL (`TypeError: __init__() got an unexpected keyword argument 'ocr_enabled'`)

- [ ] **Step 3: Add the page-count helper + auto-escalation to `UnstructuredLoader`**

In `consistency_checker/corpus/loader.py`, add imports near the top:

```python
from consistency_checker.corpus.ocr import OcrAudit, needs_ocr
```

Add a small helper above `UnstructuredLoader` (kept module-level so tests can monkey-patch it):

```python
def _pdf_page_count(path: Path) -> int:
    """Page count for a PDF; returns 0 on any error (treated as "skip OCR")."""
    if path.suffix.lower() != ".pdf":
        return 0
    try:
        from pypdf import PdfReader  # transitive dep of unstructured[pdf]

        return len(PdfReader(str(path)).pages)
    except Exception:  # pypdf raises a zoo of errors; treat all as "unknown"
        return 0
```

Extend `UnstructuredLoader.__init__` to accept the new options (keeping all existing options intact):

```python
    def __init__(
        self,
        *,
        strategy: str = "fast",
        drop_junk_lines: bool = True,
        audit: JunkAudit | None = None,
        ocr_enabled: bool = True,
        ocr_audit: OcrAudit | None = None,
    ) -> None:
        self._strategy = strategy
        self._drop_junk_lines = drop_junk_lines
        self._audit = audit
        self._ocr_enabled = ocr_enabled
        self._ocr_audit = ocr_audit
```

Extend `with_options` accordingly:

```python
    def with_options(
        self,
        *,
        drop_junk_lines: bool,
        audit: JunkAudit | None,
        ocr_enabled: bool | None = None,
        ocr_audit: OcrAudit | None = None,
    ) -> UnstructuredLoader:
        return UnstructuredLoader(
            strategy=self._strategy,
            drop_junk_lines=drop_junk_lines,
            audit=audit,
            ocr_enabled=self._ocr_enabled if ocr_enabled is None else ocr_enabled,
            ocr_audit=ocr_audit if ocr_audit is not None else self._ocr_audit,
        )
```

Refactor the body of `__call__` so the element-list → `full_text` build is a private helper, then wrap it with the escalation:

```python
    def __call__(self, path: Path) -> LoadedDocument:
        from unstructured.partition.auto import partition

        # First pass: configured strategy (default "fast").
        elements = partition(filename=str(path), strategy=self._strategy)
        full_text, element_spans = self._build_text_and_spans(elements, path)

        # OCR fallback: re-partition with hi_res when fast looks empty.
        if (
            self._ocr_enabled
            and self._strategy == "fast"
            and path.suffix.lower() == ".pdf"
        ):
            page_count = _pdf_page_count(path)
            try:
                file_size = path.stat().st_size
            except OSError:
                file_size = 0
            if needs_ocr(text=full_text, page_count=page_count, file_size=file_size):
                if self._ocr_audit is not None:
                    self._ocr_audit.record(
                        event="escalated",
                        doc_id=None,
                        path=str(path),
                        page_count=page_count,
                    )
                _log.warning(
                    "Fast extraction returned near-empty text on %s — "
                    "retrying with hi_res (OCR). First use downloads ~500 MB.",
                    path,
                )
                elements = partition(filename=str(path), strategy="hi_res")
                full_text, element_spans = self._build_text_and_spans(elements, path)
                if needs_ocr(text=full_text, page_count=page_count, file_size=file_size):
                    if self._ocr_audit is not None:
                        self._ocr_audit.record(
                            event="ocr_failed",
                            doc_id=None,
                            path=str(path),
                            page_count=page_count,
                        )
                    _log.warning(
                        "hi_res extraction also returned near-empty text on %s — "
                        "document will be ingested with whatever text was recovered.",
                        path,
                    )

        metadata = make_metadata_json({"element_spans": element_spans})
        document = Document.from_content(
            full_text,
            source_path=str(path),
            title=path.stem,
            metadata_json=metadata,
        )
        return LoadedDocument(document=document, text=full_text)

    def _build_text_and_spans(
        self, elements: list[object], path: Path
    ) -> tuple[str, list[dict[str, object]]]:
        text_parts: list[str] = []
        element_spans: list[dict[str, object]] = []
        char_offset = 0
        for index, element in enumerate(elements):
            element_type = type(element).__name__
            if element_type not in self.BODY_TYPES:
                continue
            element_text = (getattr(element, "text", "") or "").strip()
            if not element_text:
                continue
            if self._drop_junk_lines:
                reason = is_junk_line(element_text)
                if reason is not None:
                    if self._audit is not None:
                        self._audit.record(
                            stage="text", reason=reason, doc_id=str(path), text=element_text
                        )
                    continue
            if text_parts:
                text_parts.append(self.ELEMENT_SEPARATOR)
                char_offset += len(self.ELEMENT_SEPARATOR)
            start = char_offset
            text_parts.append(element_text)
            char_offset += len(element_text)
            element_spans.append(
                {
                    "element_index": index,
                    "element_type": element_type,
                    "char_start": start,
                    "char_end": char_offset,
                }
            )
        return "".join(text_parts), element_spans
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_corpus.py -q`
Expected: PASS (new + existing loader tests green)

- [ ] **Step 5: Lint, format, type-check**

Run: `uv run ruff check consistency_checker/corpus/loader.py tests/test_corpus.py && uv run ruff format consistency_checker/corpus/loader.py tests/test_corpus.py && uv run mypy consistency_checker/corpus/loader.py`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add consistency_checker/corpus/loader.py tests/test_corpus.py
git commit -m "feat: auto-escalate PDF loader to hi_res when fast returns near-empty"
```

---

## Task 4: Thread OCR audit through `load_path` / `load_corpus` / ingest

**Files:**
- Modify: `consistency_checker/corpus/loader.py` (`load_path`, `load_corpus`)
- Modify: `consistency_checker/pipeline.py` (`ingest`)
- Modify: `consistency_checker/web/app.py` (`_ingest_uploaded_paths`)
- Test: `tests/test_corpus.py`, `tests/test_pipeline.py`

- [ ] **Step 1: Extend `load_path` / `load_corpus`**

In `consistency_checker/corpus/loader.py`:

```python
def load_path(
    path: Path | str,
    *,
    junk_filter_enabled: bool = True,
    junk_audit: JunkAudit | None = None,
    ocr_enabled: bool = True,
    ocr_audit: OcrAudit | None = None,
) -> LoadedDocument:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Document path does not exist: {p}")
    ext = p.suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported extension: {ext!r}. "
            f"Registered: {sorted(LOADERS)}; stubbed: {sorted(STUB_EXTENSIONS)}."
        )
    if isinstance(loader, UnstructuredLoader):
        loader = loader.with_options(
            drop_junk_lines=junk_filter_enabled,
            audit=junk_audit,
            ocr_enabled=ocr_enabled,
            ocr_audit=ocr_audit,
        )
    return loader(p)


def load_corpus(
    corpus_dir: Path | str,
    *,
    junk_filter_enabled: bool = True,
    junk_audit: JunkAudit | None = None,
    ocr_enabled: bool = True,
    ocr_audit: OcrAudit | None = None,
) -> Iterator[LoadedDocument]:
    # ... existing body unchanged through the loop ...
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        loader = LOADERS.get(ext)
        if loader is None:
            _log.debug("Skipping %s — extension %s not registered", path, ext)
            continue
        if ext in STUB_EXTENSIONS and _is_stub(loader):
            _log.warning("Skipping %s — %s loader not yet implemented", path, ext)
            continue
        yield load_path(
            path,
            junk_filter_enabled=junk_filter_enabled,
            junk_audit=junk_audit,
            ocr_enabled=ocr_enabled,
            ocr_audit=ocr_audit,
        )
```

- [ ] **Step 2: Wire ingest in `pipeline.py`**

Add import:

```python
from consistency_checker.corpus.ocr import OcrAudit
```

In `ingest()`, build the audit alongside the junk audit and pass it into `load_corpus`:

```python
    ocr_audit = OcrAudit(config.data_dir / "ocr_events.jsonl") if config.ocr_enabled else None
    for loaded in load_corpus(
        config.corpus_dir,
        junk_filter_enabled=config.junk_filter_enabled,
        junk_audit=junk_audit,
        ocr_enabled=config.ocr_enabled,
        ocr_audit=ocr_audit,
    ):
        ...
```

After the loop, log a summary alongside the existing junk-summary line:

```python
    if ocr_audit is not None and ocr_audit.counts:
        _log.info("OCR fallback: %s", ocr_audit.counts)
```

- [ ] **Step 3: Wire the web upload path**

In `consistency_checker/web/app.py`, `_ingest_uploaded_paths`: build `OcrAudit` once before the loop and pass it through `load_path`. Mirror the junk-audit pattern already in that function. Add the import: `from consistency_checker.corpus.ocr import OcrAudit`.

- [ ] **Step 4: Run the loader + ingest + web suites**

Run: `uv run pytest tests/test_corpus.py tests/test_pipeline.py tests/test_web*.py -q`
Expected: PASS

- [ ] **Step 5: Lint, format, type-check**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy consistency_checker`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add consistency_checker/corpus/loader.py consistency_checker/pipeline.py consistency_checker/web/app.py
git commit -m "feat: thread OCR audit through load_path / load_corpus / ingest paths"
```

---

## Task 5: CLI `--no-ocr` flag + ingest summary

**Files:**
- Modify: `consistency_checker/cli/main.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_cli.py
def test_ingest_no_ocr_flag_disables_ocr(monkeypatch, tmp_path) -> None:
    """`--no-ocr` propagates to config.ocr_enabled=False before ingest runs."""
    captured: dict[str, object] = {}

    def fake_ingest(config, store, faiss_store, embedder, extractor, corpus_id):
        captured["ocr_enabled"] = config.ocr_enabled
        from consistency_checker.pipeline import IngestResult

        return IngestResult(docs=0, chunks=0, assertions=0, newly_embedded=0)

    monkeypatch.setattr("consistency_checker.cli.main.ingest", fake_ingest)
    # ... rest of harness mirrors existing CLI tests (build temp config.yml, run via Typer's CliRunner)
    # invoke: `consistency-check ingest <corpus> --config <yml> --corpus default --no-ocr`
    # assert captured["ocr_enabled"] is False
```

- [ ] **Step 2: Add the flag**

In `consistency_checker/cli/main.py`, find the `ingest` Typer command and add an `--ocr / --no-ocr` option:

```python
@app.command()
def ingest(
    corpus_dir: Path = typer.Argument(...),
    config_path: Path = typer.Option(..., "--config", "-c"),
    corpus: str | None = typer.Option(None, "--corpus"),
    ocr: bool = typer.Option(
        True,
        "--ocr/--no-ocr",
        help="Auto-escalate scanned PDFs to OCR (hi_res) when fast extraction is empty.",
    ),
) -> None:
    cfg = Config.from_yaml(config_path, env=os.environ)
    if not ocr:
        cfg = cfg.model_copy(update={"ocr_enabled": False})
    ...
```

After the ingest call, augment the existing summary print with the OCR counts when present (read from `cfg.data_dir / "ocr_events.jsonl"` or expose them via `IngestResult` — pick the lighter-touch option once you see the surrounding code).

- [ ] **Step 3: Run tests + lint**

Run: `uv run pytest tests/test_cli.py -q && uv run ruff check consistency_checker/cli/main.py && uv run mypy consistency_checker/cli/main.py`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add consistency_checker/cli/main.py tests/test_cli.py
git commit -m "feat: --no-ocr CLI flag + post-ingest OCR summary"
```

---

## Task 6: Docs — README, CORPORATE_SETUP, futureplans

**Files:**
- Modify: `README.md` (Known limitations + Supported formats)
- Modify: `docs/corporate-setup.md` (system dependency note)
- Modify: `futureplans.md` (move OCR item to Completed)

- [ ] **Step 1: Update README**

In `README.md`'s "Supported formats" table, add a note column or footnote:

> Scanned-image PDFs are auto-escalated to `unstructured`'s hi_res (OCR) strategy when fast extraction returns near-empty text. First OCR run downloads ~500 MB of layout + OCR models. Requires system Tesseract (`apt install tesseract-ocr` on Debian/Ubuntu, `brew install tesseract` on macOS).

In "Known limitations":

- Replace `PDF / DOCX loaders use 'strategy="fast"' by default — image-only PDFs need 'strategy="hi_res"' (model download, not in CI).` with `OCR fallback is automatic for image-only PDFs (`--no-ocr` to disable); first use downloads ~500 MB and requires system Tesseract.`

- [ ] **Step 2: Update CORPORATE_SETUP**

Add a "System dependencies" subsection noting that the OCR fallback requires Tesseract and the ~500 MB model download is a one-time cost. Flag the data-classification implication: OCR runs entirely locally — no document leaves the machine for OCR — so it doesn't change the data-handling story beyond what the LLM judge already does.

- [ ] **Step 3: Update futureplans.md**

Add to the "Completed" section:

```
- **OCR fallback for scanned PDFs (2026-05-31)**
  Auto-escalates fast-strategy PDF extraction to hi_res (Tesseract via
  unstructured) when the extracted text is near-empty on a multi-page
  non-trivial PDF. Disable with `--no-ocr` / `ocr_enabled: false`.
  Spec: inline in `docs/superpowers/plans/2026-05-31-ocr-fallback.md`.
  ADR-0014. Resolves the Atkins-corpus silent-drop failure mode recorded
  in `project_atkins_corpus_2026-05-30.md`.
```

- [ ] **Step 4: Commit**

```bash
git add README.md docs/corporate-setup.md futureplans.md
git commit -m "docs: OCR fallback in README + CORPORATE_SETUP + futureplans"
```

---

## Task 7: ADR-0014

**Files:**
- Create: `docs/decisions/0014-ocr-fallback.md`

- [ ] **Step 1: Write the ADR**

Use the same template as 0004-pdf-docx-loaders.md (Context / Decision / Consequences). Key points to record:

- **Context.** Atkins corpus run on 2026-05-30 silently dropped 3 of 5 PDFs to the empty-content sentinel. The fast strategy returns `""` for image-only PDFs with no error. For the legal/contract market this is 30–60% of typical corpora.
- **Alternatives considered.**
  - Always use `hi_res`: 10–100× slower; not acceptable for text PDFs.
  - OCRmyPDF as a preprocessing step: adds a separate process + intermediate file, but keeps ingest fast. Deferred — auto-escalation is a smaller surface change.
  - Cloud OCR (Textract, Google Vision, Azure): contradicts the local-first framing; documents would leave the machine. Out of scope.
- **Decision.** Auto-escalate `strategy="fast"` to `strategy="hi_res"` inside `UnstructuredLoader` when a pure detection predicate (`needs_ocr`) decides the fast result looks empty. One-shot retry — never a loop. `ocr_enabled` config flag + `--no-ocr` CLI flag for opt-out.
- **Consequences.**
  - Image PDFs now ingest content instead of collapsing into the empty sentinel.
  - First OCR run downloads ~500 MB; users see a one-line warning before it starts.
  - System Tesseract becomes a soft dependency (only loaded when needed). Documented in README and CORPORATE_SETUP.
  - Char-span invariant preserved because the loader rebuilds `full_text` from the new element list.
  - OCRmyPDF preprocessing remains an option for very large corpora where overnight batch processing makes more sense than per-ingest escalation; revisit if anyone asks.

- [ ] **Step 2: Commit**

```bash
git add docs/decisions/0014-ocr-fallback.md
git commit -m "docs: ADR-0014 OCR fallback for scanned PDFs"
```

---

## Task 8: Full-suite gate + manual verify on Atkins corpus

**Files:** none (verification only)

- [ ] **Step 1: Run the CI gate**

Run: `uv run pytest -m "not slow and not live" -q && uv run ruff check . && uv run ruff format --check . && uv run mypy consistency_checker`
Expected: all green

- [ ] **Step 2: Wipe Atkins corpus + re-ingest with OCR on**

```bash
uv run consistency-check corpus delete atkins --yes  # or wipe data/store/ if simpler
uv run consistency-check ingest /mnt/c/Users/teric/Downloads/Corpus_simplygood-atkins \
  --config config.yml --corpus atkins 2>&1 | tee /tmp/atkins_ingest_ocr.log
```

Expected:
- 5 doc rows in `documents` (no empty-sentinel row).
- Three `escalated` events in `data/store/ocr_events.jsonl` (one per scanned amendment PDF).
- Zero `ocr_failed` events.
- Assertion count substantially higher than the 8,645 from the pre-OCR run.

- [ ] **Step 3: Update the Atkins memory**

In `~/.claude/projects/-home-terickson-contradictionchecker/memory/project_atkins_corpus_2026-05-30.md`, append a section noting the post-OCR re-ingest assertion count and that the silent-drop failure mode is now closed by ADR-0014.

- [ ] **Step 4: Final commit if any doc/cleanup changes**

```bash
git add -A && git commit -m "chore: post-OCR Atkins ingest verification" || echo "nothing to commit"
```

---

## Self-review notes (for the implementer)

- The detection predicate is intentionally conservative — three guards must all fire. If a real text PDF gets escalated unnecessarily, loosen `_MIN_ALPHA_CHARS` first; don't relax the page-count or file-size guards (those protect against escalating tiny placeholder PDFs).
- The escalation is one-shot. If a future failure mode appears where hi_res needs *its own* retry (e.g. different OCR engine), do not add a loop — escalate to a separate "OCR provider" abstraction instead.
- The `_pdf_page_count` helper swallows all `pypdf` errors and returns 0 (= "skip OCR"). This is intentional: a broken pypdf parse should not block ingest. A doc that pypdf can't even count pages on is almost certainly not OCR-recoverable either.
- ADR-0014 is the place to document the OCRmyPDF / cloud-OCR alternatives. Do not litter the code with future-work TODOs.
- Tesseract install verification is **not** in the loader — checking `which tesseract` adds OS-shell complexity and gives weak signal. Let the first hi_res call fail with the unstructured/tesseract error; the warning we already log gives the user a starting point.
