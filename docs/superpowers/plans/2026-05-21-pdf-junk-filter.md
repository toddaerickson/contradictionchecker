# PDF Extraction Junk Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drop TOC dot-leaders, page numbers, and cross-reference "definitions" from PDF ingestion via a deterministic, audited two-stage filter.

**Architecture:** A pure-predicate module (`junk_filter.py`) exposes `is_junk_line` (text-stage) and `is_junk_assertion` (assertion-stage), plus a `JunkAudit` sink. The text filter is wired into `UnstructuredLoader`; the assertion filter is a `JunkFilteringExtractor` wrapper applied in `make_extractor`. A `junk_filter_enabled` config flag gates both.

**Tech Stack:** Python 3.11, stdlib `re`/`json`/`logging`, pytest, pydantic v2 (config), `unstructured` (loader, already a dep).

**Spec:** `docs/superpowers/specs/2026-05-21-pdf-junk-filter-design.md`

**Note on spec refinement:** the predicate signature is `is_junk_assertion(text: str) -> str | None` (the spec's `kind` param is dropped — the rules are kind-agnostic and the assertion's `kind` is recorded by the wrapper for the audit, not used in the decision).

---

## Task 1: Junk-filter module (pure predicates + audit sink)

**Files:**
- Create: `consistency_checker/corpus/junk_filter.py`
- Test: `tests/test_junk_filter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_junk_filter.py
"""Tests for the deterministic junk filter (pure predicates + audit sink)."""

from __future__ import annotations

import json
from pathlib import Path

from consistency_checker.corpus.junk_filter import (
    JunkAudit,
    is_junk_assertion,
    is_junk_line,
)


# --- is_junk_line: junk cases (each traces to a real observed example) ------
def test_is_junk_line_dot_leader() -> None:
    assert is_junk_line("Expenses ........ 15") == "dot_leader"
    assert is_junk_line("...............") == "dot_leader"


def test_is_junk_line_page_number() -> None:
    assert is_junk_line("15") == "page_number"
    assert is_junk_line("Page 3") == "page_number"
    assert is_junk_line("- 12 -") == "page_number"


def test_is_junk_line_mostly_non_alpha() -> None:
    assert is_junk_line("_________________") == "mostly_non_alpha"


# --- is_junk_line: clean cases that must NOT be dropped ---------------------
def test_is_junk_line_keeps_real_clause() -> None:
    assert is_junk_line("The Board shall consist of no fewer than three Directors.") is None


def test_is_junk_line_keeps_short_heading() -> None:
    assert is_junk_line("ARTICLE V") is None  # has alpha, no dots, not a bare number


def test_is_junk_line_keeps_sentence_with_ellipsis() -> None:
    # three-dot ellipsis must not trip the >=5-dot rule
    assert is_junk_line("The committee deliberated... and then voted.") is None


# --- is_junk_assertion: junk cases ------------------------------------------
def test_is_junk_assertion_cross_reference() -> None:
    assert is_junk_assertion("as defined in this Article 11") == "cross_reference"
    assert is_junk_assertion("See Section 4.2 below.") == "cross_reference"


def test_is_junk_assertion_near_empty() -> None:
    assert is_junk_assertion("1.") == "near_empty"
    assert is_junk_assertion("   ") == "near_empty"


def test_is_junk_assertion_dot_fragment() -> None:
    assert is_junk_assertion(".......... 15") == "dot_fragment"


# --- is_junk_assertion: clean cases that must NOT be dropped ----------------
def test_is_junk_assertion_keeps_real_definition() -> None:
    text = "Quorum means a majority of the Directors then in office."
    assert is_junk_assertion(text) is None


def test_is_junk_assertion_keeps_substantive_clause_referencing_section() -> None:
    # starts with a pointer phrase AND names a section, but carries real substance
    text = (
        "As set forth in Section 4, the Quorum for any meeting of the Board shall be "
        "a majority of the Directors then in office, present in person or by proxy."
    )
    assert is_junk_assertion(text) is None


# --- JunkAudit --------------------------------------------------------------
def test_junk_audit_counts_and_writes(tmp_path: Path) -> None:
    audit = JunkAudit(tmp_path / "junk_drops.jsonl")
    audit.record(stage="text", reason="dot_leader", doc_id="doc1", text="x" * 500)
    audit.record(stage="assertion", reason="cross_reference", doc_id="doc1", text="see X")
    assert audit.counts == {"dot_leader": 1, "cross_reference": 1}
    lines = (tmp_path / "junk_drops.jsonl").read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["stage"] == "text" and rec["reason"] == "dot_leader" and rec["doc_id"] == "doc1"
    assert len(rec["text_snippet"]) <= 200  # snippet is truncated


def test_junk_audit_none_path_is_memory_only(tmp_path: Path) -> None:
    audit = JunkAudit(None)
    audit.record(stage="text", reason="page_number", doc_id=None, text="15")
    assert audit.counts == {"page_number": 1}  # no file, counts still tracked


def test_junk_audit_write_failure_is_swallowed(tmp_path: Path) -> None:
    # point at a path whose parent is a file, so mkdir/open fails; must not raise
    bad = tmp_path / "afile"
    bad.write_text("x")
    audit = JunkAudit(bad / "nested" / "drops.jsonl")
    audit.record(stage="text", reason="dot_leader", doc_id=None, text="...")  # no exception
    assert audit.counts == {"dot_leader": 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_junk_filter.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'consistency_checker.corpus.junk_filter'`

- [ ] **Step 3: Write the module**

```python
# consistency_checker/corpus/junk_filter.py
"""Deterministic junk filter for PDF-extracted text and assertions.

Two pure predicates plus a small audit sink. No project imports — these are
text-in / reason-out functions so they can be lifted into a standalone package
later. Each predicate returns a short reason string when the input is junk, or
``None`` when it is clean. The predicates are total: on unexpected input they
return ``None`` (fail-open), so a filter bug can never delete real content.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import re

_log = logging.getLogger(__name__)

_DOT_LEADER_RE = re.compile(r"\.{5,}")
_PAGE_NUMBER_RE = re.compile(r"^(?:page\s+)?\d{1,4}$", re.IGNORECASE)
_DASH_PAGE_RE = re.compile(r"^[-–—]\s*\d{1,4}\s*[-–—]$")
_CROSS_REF_START_RE = re.compile(
    r"^(?:as\s+(?:defined|set\s+forth|described|provided|used|referenced)|see)\b",
    re.IGNORECASE,
)
_REF_TARGET_RE = re.compile(
    r"\b(?:article|section|paragraph|clause|bylaws?|agreement|exhibit|schedule|"
    r"herein|hereof|hereunder|above|below)\b",
    re.IGNORECASE,
)

_MIN_ALPHA_ASSERTION = 10  # below this many alpha chars → near_empty
_MAX_ALPHA_CROSS_REF = 60  # cross-ref pointer with fewer alpha chars carries no substance
_MIN_LEN_NON_ALPHA = 8  # mostly_non_alpha only applies at/above this length
_MAX_NON_ALPHA_RATIO = 0.15  # below this alpha fraction → mostly non-alphabetic


def _alpha_count(text: str) -> int:
    return sum(1 for c in text if c.isalpha())


def _alpha_ratio(text: str) -> float:
    return _alpha_count(text) / len(text) if text else 0.0


def is_junk_line(text: str) -> str | None:
    """Reason string if ``text`` (one extracted body element) is structural junk."""
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None  # the loader already skips empty elements
    if _PAGE_NUMBER_RE.match(stripped) or _DASH_PAGE_RE.match(stripped):
        return "page_number"
    if _DOT_LEADER_RE.search(stripped):
        return "dot_leader"
    if len(stripped) >= _MIN_LEN_NON_ALPHA and _alpha_ratio(stripped) < _MAX_NON_ALPHA_RATIO:
        return "mostly_non_alpha"
    return None


def is_junk_assertion(text: str) -> str | None:
    """Reason string if an extracted assertion/definition is junk."""
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if _alpha_count(stripped) < _MIN_ALPHA_ASSERTION:
        return "near_empty"
    if _DOT_LEADER_RE.search(stripped) or (
        len(stripped) >= _MIN_LEN_NON_ALPHA and _alpha_ratio(stripped) < _MAX_NON_ALPHA_RATIO
    ):
        return "dot_fragment"
    if (
        _CROSS_REF_START_RE.match(stripped)
        and _REF_TARGET_RE.search(stripped)
        and _alpha_count(stripped) < _MAX_ALPHA_CROSS_REF
    ):
        return "cross_reference"
    return None


class JunkAudit:
    """Records dropped items: in-memory counts always, JSONL when a path is set."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._counts: dict[str, int] = {}

    def record(self, *, stage: str, reason: str, doc_id: str | None, text: str) -> None:
        self._counts[reason] = self._counts.get(reason, 0) + 1
        if self._path is None:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "reason": reason,
            "doc_id": doc_id,
            "text_snippet": text[:200],
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:  # audit must never abort ingest
            _log.warning("junk audit write failed (%s): %s", self._path, exc)

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_junk_filter.py -q`
Expected: PASS (12 passed)

- [ ] **Step 5: Lint, format, type-check**

Run: `uv run ruff check consistency_checker/corpus/junk_filter.py tests/test_junk_filter.py && uv run ruff format consistency_checker/corpus/junk_filter.py tests/test_junk_filter.py && uv run mypy consistency_checker/corpus/junk_filter.py`
Expected: all pass / no issues

- [ ] **Step 6: Commit**

```bash
git add consistency_checker/corpus/junk_filter.py tests/test_junk_filter.py
git commit -m "feat: deterministic junk-filter predicates + audit sink"
```

---

## Task 2: Config flag `junk_filter_enabled`

**Files:**
- Modify: `consistency_checker/config.py` (add field after `judge_model`, ~line 67)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

```python
# append to tests/test_config.py
def test_junk_filter_enabled_defaults_true(tmp_path: Path) -> None:
    yml = write_yaml(tmp_path / "c.yml", {"corpus_dir": str(tmp_path / "corpus")})
    cfg = Config.from_yaml(yml, env={})
    assert cfg.junk_filter_enabled is True


def test_junk_filter_enabled_can_disable(tmp_path: Path) -> None:
    yml = write_yaml(
        tmp_path / "c.yml",
        {"corpus_dir": str(tmp_path / "corpus"), "junk_filter_enabled": False},
    )
    cfg = Config.from_yaml(yml, env={})
    assert cfg.junk_filter_enabled is False
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config.py -k junk_filter -q`
Expected: FAIL (`AttributeError` / `ValidationError: extra fields not permitted`)

- [ ] **Step 3: Add the field**

In `consistency_checker/config.py`, immediately after the `judge_model` field (line 67):

```python
    judge_model: str = Field(default="claude-sonnet-4-6")

    junk_filter_enabled: bool = Field(
        default=True,
        description="Drop structural junk (TOC dot-leaders, page numbers, "
        "cross-reference 'definitions') during ingest. See junk_filter.py.",
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_config.py -k junk_filter -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/config.py tests/test_config.py
git commit -m "feat: add junk_filter_enabled config flag (default on)"
```

---

## Task 3: Assertion seam — `JunkFilteringExtractor` + wire `make_extractor`

**Files:**
- Modify: `consistency_checker/extract/atomic_facts.py` (add class at end of file)
- Modify: `consistency_checker/pipeline.py` (`make_extractor`, ~line 120)
- Test: `tests/test_atomic_facts.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_atomic_facts.py
from consistency_checker.corpus.junk_filter import JunkAudit
from consistency_checker.extract.atomic_facts import JunkFilteringExtractor


def test_junk_filtering_extractor_drops_junk_keeps_clean() -> None:
    chunk = make_chunk("body text", doc_id="docZ")
    inner = FixtureExtractor(
        {chunk.chunk_id: ["Quorum means a majority of the Directors.", "as defined in Article 11"]}
    )
    audit = JunkAudit(None)
    ext = JunkFilteringExtractor(inner, audit=audit)
    out = ext.extract(chunk)
    assert [a.assertion_text for a in out] == ["Quorum means a majority of the Directors."]
    assert audit.counts == {"cross_reference": 1}


def test_junk_filtering_extractor_no_audit_ok() -> None:
    chunk = make_chunk("body", doc_id="docZ")
    inner = FixtureExtractor({chunk.chunk_id: ["1."]})  # near_empty → dropped
    ext = JunkFilteringExtractor(inner)
    assert ext.extract(chunk) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_atomic_facts.py -k JunkFiltering -q`
Expected: FAIL (`ImportError: cannot import name 'JunkFilteringExtractor'`)

- [ ] **Step 3: Add `JunkFilteringExtractor` to `atomic_facts.py`**

At the top of `consistency_checker/extract/atomic_facts.py`, add to imports:

```python
from consistency_checker.corpus.junk_filter import JunkAudit, is_junk_assertion
```

At the end of the file, add:

```python
class JunkFilteringExtractor:
    """Wraps an :class:`Extractor`, dropping assertions flagged by ``is_junk_assertion``.

    Applied in :func:`make_extractor` so every ingest path (CLI, web, headless)
    inherits assertion-stage filtering with no per-path wiring.
    """

    def __init__(self, inner: Extractor, *, audit: JunkAudit | None = None) -> None:
        self._inner = inner
        self._audit = audit

    def extract(self, chunk: Chunk) -> list[Assertion]:
        kept: list[Assertion] = []
        for assertion in self._inner.extract(chunk):
            reason = is_junk_assertion(assertion.assertion_text)
            if reason is None:
                kept.append(assertion)
            elif self._audit is not None:
                self._audit.record(
                    stage="assertion",
                    reason=reason,
                    doc_id=assertion.doc_id,
                    text=assertion.assertion_text,
                )
        return kept
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_atomic_facts.py -k JunkFiltering -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Wire `make_extractor` in `pipeline.py`**

Add `JunkFilteringExtractor` to the existing `atomic_facts` import block (~line 55):

```python
from consistency_checker.extract.atomic_facts import (
    AnthropicExtractor,
    Extractor,
    FixtureExtractor,
    JunkFilteringExtractor,
    MoonshotExtractor,
)
```

Add `JunkAudit` import near the other corpus imports:

```python
from consistency_checker.corpus.junk_filter import JunkAudit
```

Replace `make_extractor` (currently ~lines 120-127):

```python
def make_extractor(config: Config) -> Extractor:
    """Build an extractor from config; ``fixture`` provider returns an empty fixture."""
    if config.judge_provider == "fixture":
        return FixtureExtractor({})
    inner: Extractor
    if config.judge_provider == "moonshot":
        inner = MoonshotExtractor(model="kimi-k2.6")
    else:
        inner = AnthropicExtractor(model=config.judge_model)
    if config.junk_filter_enabled:
        audit = JunkAudit(config.data_dir / "junk_drops.jsonl")
        return JunkFilteringExtractor(inner, audit=audit)
    return inner
```

- [ ] **Step 6: Run the full extractor + pipeline-touching test suites**

Run: `uv run pytest tests/test_atomic_facts.py tests/test_pipeline.py -q`
Expected: PASS (existing tests still green; fixture path unaffected since it returns early)

- [ ] **Step 7: Lint, format, type-check**

Run: `uv run ruff check consistency_checker/extract/atomic_facts.py consistency_checker/pipeline.py && uv run ruff format consistency_checker/extract/atomic_facts.py consistency_checker/pipeline.py && uv run mypy consistency_checker/extract/atomic_facts.py consistency_checker/pipeline.py`
Expected: all pass / no issues

- [ ] **Step 8: Commit**

```bash
git add consistency_checker/extract/atomic_facts.py consistency_checker/pipeline.py tests/test_atomic_facts.py
git commit -m "feat: assertion-stage junk filter via make_extractor wrapper"
```

---

## Task 4: Text seam — `UnstructuredLoader` filtering + thread through `load_path`/`load_corpus`/ingest

**Files:**
- Modify: `consistency_checker/corpus/loader.py` (`UnstructuredLoader`, `load_path`, `load_corpus`)
- Modify: `consistency_checker/pipeline.py` (`ingest`)
- Modify: `consistency_checker/web/app.py` (`_ingest_uploaded_paths` call site, ~line 970)
- Test: `tests/test_corpus.py` (existing loader-test home; reuse its `_skip_unstructured_on_win` marker)

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_corpus.py
# (_skip_unstructured_on_win is already defined in this file; UnstructuredLoader,
# load_path, LoadedDocument are already imported at the top.)
from consistency_checker.corpus.junk_filter import JunkAudit


def _named(text: str, category: str):
    """Build a fake unstructured element whose type name drives BODY_TYPES."""
    element = type(category, (), {})()
    element.text = text
    return element


def _patch_partition(monkeypatch: pytest.MonkeyPatch, elements: list[object]) -> None:
    # UnstructuredLoader.__call__ does `from unstructured.partition.auto import partition`,
    # which resolves the attribute on this module at call time.
    import unstructured.partition.auto as auto

    monkeypatch.setattr(auto, "partition", lambda **kwargs: elements)


@_skip_unstructured_on_win
def test_loader_drops_junk_lines_and_keeps_real(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    elements = [
        _named("The Board shall consist of three Directors.", "NarrativeText"),
        _named("Table of Contents ........ 2", "Title"),  # dot_leader junk
        _named("15", "Text"),  # page_number junk
        _named("Members may vote by proxy.", "NarrativeText"),
    ]
    _patch_partition(monkeypatch, elements)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")
    audit = JunkAudit(None)
    loaded = UnstructuredLoader(drop_junk_lines=True, audit=audit)(pdf)
    assert "Board shall consist" in loaded.text
    assert "vote by proxy" in loaded.text
    assert "........" not in loaded.text
    assert audit.counts == {"dot_leader": 1, "page_number": 1}


@_skip_unstructured_on_win
def test_loader_char_span_invariant_holds_after_filtering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    elements = [
        _named("First real clause.", "NarrativeText"),
        _named("........ 9", "Title"),  # dropped
        _named("Second real clause.", "NarrativeText"),
    ]
    _patch_partition(monkeypatch, elements)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF stub")
    loaded = UnstructuredLoader(drop_junk_lines=True)(pdf)
    spans = json.loads(loaded.document.metadata_json)["element_spans"]
    for span in spans:
        assert loaded.text[span["char_start"] : span["char_end"]] == (
            "First real clause." if span["char_start"] == 0 else "Second real clause."
        )
    assert "........" not in loaded.text


@_skip_unstructured_on_win
def test_loader_disabled_keeps_everything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    elements = [_named("Real clause.", "NarrativeText"), _named("15", "Text")]
    _patch_partition(monkeypatch, elements)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF stub")
    loaded = UnstructuredLoader(drop_junk_lines=False)(pdf)
    assert "15" in loaded.text
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_corpus.py -k "loader_drops_junk or char_span_invariant_holds_after or disabled_keeps" -q`
Expected: FAIL (`TypeError: __init__() got an unexpected keyword argument 'drop_junk_lines'`)

- [ ] **Step 3: Add filtering to `UnstructuredLoader`**

In `consistency_checker/corpus/loader.py`, add the import near the top:

```python
from consistency_checker.corpus.junk_filter import JunkAudit, is_junk_line
```

Replace `UnstructuredLoader.__init__` (currently `def __init__(self, *, strategy: str = "fast") -> None: self._strategy = strategy`):

```python
    def __init__(
        self,
        *,
        strategy: str = "fast",
        drop_junk_lines: bool = True,
        audit: JunkAudit | None = None,
    ) -> None:
        self._strategy = strategy
        self._drop_junk_lines = drop_junk_lines
        self._audit = audit

    def with_options(
        self, *, drop_junk_lines: bool, audit: JunkAudit | None
    ) -> UnstructuredLoader:
        """Return a copy with junk-filter settings overridden (strategy preserved)."""
        return UnstructuredLoader(
            strategy=self._strategy, drop_junk_lines=drop_junk_lines, audit=audit
        )
```

In `UnstructuredLoader.__call__`, the loop currently is:

```python
            if element_type not in self.BODY_TYPES:
                continue
            element_text = (getattr(element, "text", "") or "").strip()
            if not element_text:
                continue
```

Add the junk check immediately after the empty check:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_loader.py -q`
Expected: PASS

- [ ] **Step 5: Thread junk settings through `load_path` and `load_corpus`**

Replace `load_path`:

```python
def load_path(
    path: Path | str,
    *,
    junk_filter_enabled: bool = True,
    junk_audit: JunkAudit | None = None,
) -> LoadedDocument:
    """Load a single document by path. Dispatches via :data:`LOADERS`.

    Raises ``NotImplementedError`` for stubbed extensions, ``ValueError`` for
    extensions with no registered loader, and ``FileNotFoundError`` for missing
    paths.
    """
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
        loader = loader.with_options(drop_junk_lines=junk_filter_enabled, audit=junk_audit)
    return loader(p)
```

Replace the tail of `load_corpus` (the `for path in sorted(...)` block) so it forwards to `load_path`, and add the two params to its signature:

```python
def load_corpus(
    corpus_dir: Path | str,
    *,
    junk_filter_enabled: bool = True,
    junk_audit: JunkAudit | None = None,
) -> Iterator[LoadedDocument]:
    """Walk ``corpus_dir`` recursively, yielding loaded documents.

    Files with unregistered extensions are skipped silently (DEBUG log). Stub
    extensions emit an explicit WARNING so users see them.
    """
    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus path is not a directory: {root}")

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
            path, junk_filter_enabled=junk_filter_enabled, junk_audit=junk_audit
        )
```

- [ ] **Step 6: Wire ingest entry points**

In `consistency_checker/pipeline.py`, `ingest()` — replace the `for loaded in load_corpus(config.corpus_dir):` line, and add a one-line summary after the loop (next to the existing `_log.info("Ingested ...")` call):

```python
    junk_audit = JunkAudit(config.data_dir / "junk_drops.jsonl") if config.junk_filter_enabled else None
    for loaded in load_corpus(
        config.corpus_dir,
        junk_filter_enabled=config.junk_filter_enabled,
        junk_audit=junk_audit,
    ):
```

After the document loop (before `return IngestResult(...)`), add the audit summary:

```python
    if junk_audit is not None and junk_audit.counts:
        _log.info("Junk filter dropped (text stage): %s", junk_audit.counts)
```

(Assertion-stage drops are written to the same `junk_drops.jsonl` by the extractor's audit and are filterable by `"stage": "assertion"`; the spec's per-ingest summary is satisfied by this log line plus the shared JSONL.)

In `consistency_checker/web/app.py`, `_ingest_uploaded_paths` — the loop body has `loaded = load_path(path)`. Add a `JunkAudit` built from `config` once before the loop and pass it:

```python
    junk_audit = (
        JunkAudit(config.data_dir / "junk_drops.jsonl") if config.junk_filter_enabled else None
    )
    for path in paths:
        loaded = load_path(
            path,
            junk_filter_enabled=config.junk_filter_enabled,
            junk_audit=junk_audit,
        )
```

Add the import to `web/app.py`: `from consistency_checker.corpus.junk_filter import JunkAudit`.

- [ ] **Step 7: Run the loader + ingest + web suites**

Run: `uv run pytest tests/test_corpus.py tests/test_pipeline.py tests/test_web*.py -q`
Expected: PASS (existing ingest/web tests still green)

- [ ] **Step 8: Lint, format, type-check**

Run: `uv run ruff check consistency_checker/corpus/loader.py consistency_checker/pipeline.py consistency_checker/web/app.py tests/test_corpus.py && uv run ruff format consistency_checker/corpus/loader.py consistency_checker/pipeline.py consistency_checker/web/app.py tests/test_corpus.py && uv run mypy consistency_checker/corpus/loader.py consistency_checker/pipeline.py consistency_checker/web/app.py`
Expected: all pass / no issues

- [ ] **Step 9: Commit**

```bash
git add consistency_checker/corpus/loader.py consistency_checker/pipeline.py consistency_checker/web/app.py tests/test_corpus.py
git commit -m "feat: text-stage junk filter in loader, threaded through ingest paths"
```

---

## Task 5: Full-suite gate + manual re-check on the real corpus

**Files:** none (verification only)

- [ ] **Step 1: Run the CI gate**

Run: `uv run pytest -m "not slow and not live" -q && uv run ruff check . && uv run ruff format --check . && uv run mypy consistency_checker`
Expected: all green

- [ ] **Step 2: Re-ingest the nonprofit corpus into a fresh store and confirm junk drops**

Run (the 3 PDFs already on disk from the earlier upload):
```bash
uv run python -c "
from pathlib import Path
from consistency_checker.config import Config, load_local_env
from consistency_checker.corpus.loader import load_path
from consistency_checker.corpus.junk_filter import JunkAudit
load_local_env()
audit = JunkAudit(Path('data/nonprofit2/junk_drops.jsonl'))
src = Path('data/store/uploads/2026-05-20T20-01-36_13872f15')
for f in sorted(src.glob('*.pdf')):
    ld = load_path(f, junk_filter_enabled=True, junk_audit=audit)
    print(f.name, '->', len(ld.text), 'chars kept')
print('junk dropped (text stage):', audit.counts)
"
```
Expected: non-empty `audit.counts` showing `dot_leader` / `page_number` / `mostly_non_alpha` drops on `Annotated-Bylaws.pdf`, and clause text still present.

- [ ] **Step 3: Final commit if any doc/cleanup changes**

```bash
git add -A && git commit -m "chore: junk-filter verification artifacts" || echo "nothing to commit"
```

---

## Self-review notes (for the implementer)

- Predicate thresholds are intentionally conservative; if Step 2 of Task 5 shows real clauses being dropped, loosen `_MAX_NON_ALPHA_RATIO` / `_MIN_ALPHA_ASSERTION` in `junk_filter.py` and add a failing-then-passing test capturing the real example before changing the constant.
- The `cross_reference` rule is the highest false-positive risk; the `_MAX_ALPHA_CROSS_REF = 60` guard is what protects substantive clauses that merely cite a Section.
