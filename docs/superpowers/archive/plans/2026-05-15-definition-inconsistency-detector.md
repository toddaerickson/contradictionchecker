# Definition-inconsistency detector — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a second member of the consistency-detector family that flags divergent definitions of the same term across the corpus, reusing the existing ingest/judge/audit infrastructure with two additive migrations.

**Architecture:** Definitions are extracted alongside atomic facts by extending the existing extractor's tool schema, stored in `assertions` with `kind='definition'` plus `term` / `definition_text` columns, and judged pairwise by a new `DefinitionChecker` whose findings land in the existing `findings` table under a new `detector_type='definition_inconsistency'` discriminator. The NLI gate is bypassed for this stage; the gate is replaced by canonical-term grouping.

**Tech Stack:** Python 3.11+, SQLite (canonical store), Pydantic v2 (structured-output validation), Anthropic + OpenAI SDKs (judge providers), pytest with the `slow` / `live` / `e2e_fixture` marks, ruff + mypy strict.

**Reference:** [`docs/superpowers/specs/2026-05-15-definition-inconsistency-detector-design.md`](../specs/2026-05-15-definition-inconsistency-detector-design.md). Re-read the spec's "Decisions deferred to the implementation plan" before starting.

---

## Phase 0 — Schema

### Task 1: Migration 0007 — `assertions.kind` / `term` / `definition_text`

**Files:**
- Create: `consistency_checker/index/migrations/0007_assertion_kind.sql`
- Test: `tests/index/test_migrations.py` (add new tests; create the file if absent)

- [ ] **Step 1: Write the failing test**

Append to `tests/index/test_migrations.py` (or create with the same imports the existing migration tests use — check `tests/index/test_assertion_store.py` for the pattern):

```python
from pathlib import Path

from consistency_checker.index.assertion_store import AssertionStore


def test_migration_0007_adds_kind_columns(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    cols = {
        row["name"]: row
        for row in store._conn.execute("PRAGMA table_info(assertions)").fetchall()
    }
    assert "kind" in cols
    assert cols["kind"]["dflt_value"] == "'claim'"
    assert cols["kind"]["notnull"] == 1
    assert "term" in cols
    assert "definition_text" in cols
    # Indexes created
    idx = {row["name"] for row in store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='assertions'"
    ).fetchall()}
    assert "idx_assertions_kind" in idx
    assert "idx_assertions_term" in idx
    store.close()


def test_migration_0007_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    AssertionStore(db).migrate()
    # Second open + migrate must be a no-op (no exception, no duplicate columns).
    store = AssertionStore(db)
    applied = store.migrate()
    assert applied == []
    store.close()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/index/test_migrations.py::test_migration_0007_adds_kind_columns -v
```

Expected: FAIL — `kind` column missing.

- [ ] **Step 3: Write the migration**

Create `consistency_checker/index/migrations/0007_assertion_kind.sql`:

```sql
-- Adds the `kind` discriminator to assertions and the two definition-only
-- columns. Backwards-compatible: existing rows default to `kind='claim'` so
-- the contradiction detector continues to see them unchanged.
ALTER TABLE assertions ADD COLUMN kind TEXT NOT NULL DEFAULT 'claim';
ALTER TABLE assertions ADD COLUMN term TEXT;
ALTER TABLE assertions ADD COLUMN definition_text TEXT;
CREATE INDEX IF NOT EXISTS idx_assertions_kind ON assertions(kind);
CREATE INDEX IF NOT EXISTS idx_assertions_term ON assertions(term) WHERE kind = 'definition';
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/index/test_migrations.py -v -k 0007
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/index/migrations/0007_assertion_kind.sql tests/index/test_migrations.py
git commit -m "feat(schema): migration 0007 — assertion kind / term / definition_text"
```

---

### Task 2: Migration 0008 — `findings.detector_type`

**Files:**
- Create: `consistency_checker/index/migrations/0008_finding_detector_type.sql`
- Test: `tests/index/test_migrations.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/index/test_migrations.py`:

```python
def test_migration_0008_adds_detector_type(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    cols = {
        row["name"]: row
        for row in store._conn.execute("PRAGMA table_info(findings)").fetchall()
    }
    assert "detector_type" in cols
    assert cols["detector_type"]["dflt_value"] == "'contradiction'"
    assert cols["detector_type"]["notnull"] == 1
    idx = {row["name"] for row in store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='findings'"
    ).fetchall()}
    assert "idx_findings_detector" in idx
    store.close()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/index/test_migrations.py::test_migration_0008_adds_detector_type -v
```

Expected: FAIL — `detector_type` column missing.

- [ ] **Step 3: Write the migration**

Create `consistency_checker/index/migrations/0008_finding_detector_type.sql`:

```sql
-- Adds the detector_type discriminator to findings so the same table can hold
-- pair-shaped findings produced by multiple detectors (contradiction, then
-- definition_inconsistency in this build, then others later).
ALTER TABLE findings ADD COLUMN detector_type TEXT NOT NULL DEFAULT 'contradiction';
CREATE INDEX IF NOT EXISTS idx_findings_detector ON findings(detector_type);
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/index/test_migrations.py -v -k 0008
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/index/migrations/0008_finding_detector_type.sql tests/index/test_migrations.py
git commit -m "feat(schema): migration 0008 — findings.detector_type discriminator"
```

---

### Task 3: Extend `Assertion` dataclass

**Files:**
- Modify: `consistency_checker/extract/schema.py`
- Test: `tests/extract/test_schema.py` (add to existing file)

- [ ] **Step 1: Write the failing test**

Add to `tests/extract/test_schema.py`:

```python
from consistency_checker.extract.schema import Assertion


def test_assertion_defaults_to_claim_kind() -> None:
    a = Assertion.build("doc1", "The Borrower is ABC Corp.")
    assert a.kind == "claim"
    assert a.term is None
    assert a.definition_text is None


def test_assertion_can_be_a_definition() -> None:
    a = Assertion(
        assertion_id="abcd1234abcd1234",
        doc_id="doc1",
        assertion_text='"Borrower" means ABC Corp and its Subsidiaries.',
        kind="definition",
        term="Borrower",
        definition_text="ABC Corp and its Subsidiaries",
    )
    assert a.kind == "definition"
    assert a.term == "Borrower"
    assert a.definition_text == "ABC Corp and its Subsidiaries"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/extract/test_schema.py::test_assertion_can_be_a_definition -v
```

Expected: FAIL — `kind`/`term`/`definition_text` unknown.

- [ ] **Step 3: Update the dataclass**

In `consistency_checker/extract/schema.py`, extend `Assertion` (keep `frozen=True, slots=True`):

```python
@dataclass(frozen=True, slots=True)
class Assertion:
    """An atomic, decontextualised claim or definition extracted from a document."""

    assertion_id: str
    doc_id: str
    assertion_text: str
    chunk_id: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    faiss_row: int | None = None
    embedded_at: datetime | None = None
    created_at: datetime | None = None
    kind: str = "claim"
    term: str | None = None
    definition_text: str | None = None

    @classmethod
    def build(
        cls,
        doc_id: str,
        assertion_text: str,
        *,
        chunk_id: str | None = None,
        char_start: int | None = None,
        char_end: int | None = None,
        kind: str = "claim",
        term: str | None = None,
        definition_text: str | None = None,
    ) -> Assertion:
        return cls(
            assertion_id=hash_id(doc_id, assertion_text),
            doc_id=doc_id,
            assertion_text=assertion_text,
            chunk_id=chunk_id,
            char_start=char_start,
            char_end=char_end,
            kind=kind,
            term=term,
            definition_text=definition_text,
        )
```

- [ ] **Step 4: Run the schema tests**

```bash
uv run pytest tests/extract/test_schema.py -v
```

Expected: PASS.

- [ ] **Step 5: Verify nothing else broke**

```bash
uv run pytest -m "not slow and not live" -x
```

Expected: PASS (every existing call site uses kwargs or positional-up-to-assertion_text, both still work).

- [ ] **Step 6: Commit**

```bash
git add consistency_checker/extract/schema.py tests/extract/test_schema.py
git commit -m "feat(schema): Assertion gains kind/term/definition_text with claim defaults"
```

---

### Task 4: AssertionStore — round-trip new columns

**Files:**
- Modify: `consistency_checker/index/assertion_store.py`
- Test: `tests/index/test_assertion_store.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to `tests/index/test_assertion_store.py`:

```python
from pathlib import Path

from consistency_checker.extract.schema import Assertion
from consistency_checker.index.assertion_store import AssertionStore


def test_definition_assertion_round_trips(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    # documents row required by FK
    from consistency_checker.extract.schema import Document
    store.add_document(Document(doc_id="docA", source_path="/A.txt"))
    a = Assertion.build(
        "docA",
        '"Borrower" means ABC Corp and its Subsidiaries.',
        kind="definition",
        term="Borrower",
        definition_text="ABC Corp and its Subsidiaries",
    )
    store.add_assertion(a)
    fetched = store.get_assertion(a.assertion_id)
    assert fetched is not None
    assert fetched.kind == "definition"
    assert fetched.term == "Borrower"
    assert fetched.definition_text == "ABC Corp and its Subsidiaries"
    store.close()


def test_claim_assertion_unaffected_by_kind_columns(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    from consistency_checker.extract.schema import Document
    store.add_document(Document(doc_id="docA", source_path="/A.txt"))
    a = Assertion.build("docA", "Revenue grew 12 percent.")
    store.add_assertion(a)
    fetched = store.get_assertion(a.assertion_id)
    assert fetched is not None
    assert fetched.kind == "claim"
    assert fetched.term is None
    assert fetched.definition_text is None
    store.close()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/index/test_assertion_store.py::test_definition_assertion_round_trips -v
```

Expected: FAIL — `term`/`definition_text` not persisted.

- [ ] **Step 3: Update the store**

In `consistency_checker/index/assertion_store.py`:

(a) Extend `_ALL_ASSERTION_COLUMNS`:

```python
_ALL_ASSERTION_COLUMNS: tuple[str, ...] = (
    "assertion_id",
    "doc_id",
    "assertion_text",
    "chunk_id",
    "char_start",
    "char_end",
    "faiss_row",
    "embedded_at",
    "created_at",
    "kind",
    "term",
    "definition_text",
)
```

(b) Update `_row_to_assertion`:

```python
def _row_to_assertion(row: sqlite3.Row) -> Assertion:
    return Assertion(
        assertion_id=row["assertion_id"],
        doc_id=row["doc_id"],
        assertion_text=row["assertion_text"],
        chunk_id=row["chunk_id"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        faiss_row=row["faiss_row"],
        embedded_at=_parse_timestamp(row["embedded_at"]),
        created_at=_parse_timestamp(row["created_at"]),
        kind=row["kind"],
        term=row["term"],
        definition_text=row["definition_text"],
    )
```

(c) Update `add_assertions` to write the new columns:

```python
def add_assertions(self, assertions: Iterable[Assertion]) -> None:
    rows = [
        (
            a.assertion_id,
            a.doc_id,
            a.assertion_text,
            a.chunk_id,
            a.char_start,
            a.char_end,
            a.kind,
            a.term,
            a.definition_text,
        )
        for a in assertions
    ]
    if not rows:
        return
    with self._conn:
        self._conn.executemany(
            "INSERT OR IGNORE INTO assertions"
            "(assertion_id, doc_id, assertion_text, chunk_id, char_start, char_end, "
            "kind, term, definition_text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
```

- [ ] **Step 4: Run the assertion-store tests**

```bash
uv run pytest tests/index/test_assertion_store.py -v
```

Expected: PASS.

- [ ] **Step 5: Run the full hermetic suite to catch any unexpected breakage**

```bash
uv run pytest -m "not slow and not live" -x
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add consistency_checker/index/assertion_store.py tests/index/test_assertion_store.py
git commit -m "feat(store): persist kind/term/definition_text on assertions"
```

---

## Phase 1 — Definition queries

### Task 5: Term canonicalization helper

**Files:**
- Create: `consistency_checker/check/definition_terms.py`
- Test: `tests/check/test_definition_terms.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/check/test_definition_terms.py`:

```python
import pytest

from consistency_checker.check.definition_terms import canonicalize_term


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Borrower", "borrower"),
        ("borrower", "borrower"),
        ("the Borrower", "borrower"),
        ("The Borrower", "borrower"),
        ("Borrowers", "borrower"),
        ('"Borrower"', "borrower"),
        ("“Borrower”", "borrower"),  # curly quotes
        ("  Borrower  ", "borrower"),
        ("Material Adverse Effect", "material adverse effect"),
        ("MAE", "mae"),
    ],
)
def test_canonicalize_term(raw: str, expected: str) -> None:
    assert canonicalize_term(raw) == expected


def test_canonicalize_empty() -> None:
    assert canonicalize_term("") == ""
    assert canonicalize_term("   ") == ""
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/check/test_definition_terms.py -v
```

Expected: FAIL — module not defined.

- [ ] **Step 3: Implement the helper**

Create `consistency_checker/check/definition_terms.py`:

```python
"""Term canonicalization for the definition-inconsistency detector.

Two definition assertions are grouped if their canonical terms match. The
canonical form folds case, strips surrounding whitespace and quote characters,
removes the leading "the ", and trims a trailing plural "s". Intentionally
conservative — we'd rather miss a group (false negative) than merge unrelated
terms (false positive).
"""

from __future__ import annotations

_QUOTE_CHARS = ('"', "'", "“", "”", "‘", "’", "`")


def canonicalize_term(raw: str) -> str:
    """Return the canonical, comparison-ready form of a defined-term string."""
    text = raw.strip()
    while text and text[0] in _QUOTE_CHARS:
        text = text[1:]
    while text and text[-1] in _QUOTE_CHARS:
        text = text[:-1]
    text = text.strip().lower()
    if text.startswith("the "):
        text = text[4:]
    # Conservative depluralisation: only strip a trailing "s" when the word is
    # longer than two characters and doesn't end in "ss".
    if len(text) > 2 and text.endswith("s") and not text.endswith("ss"):
        text = text[:-1]
    return text
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/check/test_definition_terms.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/check/definition_terms.py tests/check/test_definition_terms.py
git commit -m "feat(check): canonical-term helper for definition grouping"
```

---

### Task 6: AssertionStore — iterate definitions, group by canonical term

**Files:**
- Modify: `consistency_checker/index/assertion_store.py`
- Test: `tests/index/test_assertion_store.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to `tests/index/test_assertion_store.py`:

```python
def test_iter_definitions_filters_to_kind_definition(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    from consistency_checker.extract.schema import Document
    store.add_document(Document(doc_id="docA", source_path="/A.txt"))
    store.add_document(Document(doc_id="docB", source_path="/B.txt"))
    store.add_assertion(Assertion.build("docA", "Revenue grew 12%."))
    store.add_assertion(
        Assertion.build(
            "docA",
            '"Borrower" means ABC Corp.',
            kind="definition",
            term="Borrower",
            definition_text="ABC Corp",
        )
    )
    store.add_assertion(
        Assertion.build(
            "docB",
            "Borrower means ABC Corp and its Subsidiaries.",
            kind="definition",
            term="Borrower",
            definition_text="ABC Corp and its Subsidiaries",
        )
    )
    defs = list(store.iter_definitions())
    assert len(defs) == 2
    assert all(d.kind == "definition" for d in defs)
    store.close()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/index/test_assertion_store.py::test_iter_definitions_filters_to_kind_definition -v
```

Expected: FAIL — `iter_definitions` not defined.

- [ ] **Step 3: Add the iterator**

In `consistency_checker/index/assertion_store.py`, alongside `iter_assertions`:

```python
def iter_definitions(self) -> Iterator[Assertion]:
    """Iterate every assertion with ``kind='definition'`` ordered by created_at."""
    cursor = self._conn.execute(
        "SELECT * FROM assertions WHERE kind = 'definition' "
        "ORDER BY created_at, assertion_id"
    )
    for row in cursor:
        yield _row_to_assertion(row)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/index/test_assertion_store.py::test_iter_definitions_filters_to_kind_definition -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/index/assertion_store.py tests/index/test_assertion_store.py
git commit -m "feat(store): iter_definitions() for the definition detector"
```

---

## Phase 2 — Extraction

### Task 7: Extend extractor tool-schema + Pydantic guard

**Files:**
- Modify: `consistency_checker/extract/atomic_facts.py`
- Test: `tests/extract/test_atomic_facts.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to `tests/extract/test_atomic_facts.py`:

```python
from consistency_checker.extract.atomic_facts import (
    TOOL_SCHEMA,
    _ExtractionPayload,
)


def test_tool_schema_includes_definitions_array() -> None:
    props = TOOL_SCHEMA["input_schema"]["properties"]
    assert "assertions" in props
    assert "definitions" in props
    assert props["definitions"]["type"] == "array"
    item = props["definitions"]["items"]
    assert item["type"] == "object"
    assert "term" in item["properties"]
    assert "definition_text" in item["properties"]
    assert "containing_sentence" in item["properties"]
    assert set(item["required"]) == {"term", "definition_text", "containing_sentence"}


def test_extraction_payload_parses_combined_response() -> None:
    payload = _ExtractionPayload.model_validate(
        {
            "assertions": ["Revenue grew 12 percent in fiscal 2025."],
            "definitions": [
                {
                    "term": "Borrower",
                    "definition_text": "ABC Corp and its Subsidiaries",
                    "containing_sentence": (
                        '"Borrower" means ABC Corp and its Subsidiaries.'
                    ),
                }
            ],
        }
    )
    assert len(payload.assertions) == 1
    assert len(payload.definitions) == 1
    assert payload.definitions[0].term == "Borrower"


def test_extraction_payload_defaults_empty() -> None:
    payload = _ExtractionPayload.model_validate({"assertions": [], "definitions": []})
    assert payload.assertions == []
    assert payload.definitions == []
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/extract/test_atomic_facts.py -v -k "tool_schema_includes_definitions or extraction_payload"
```

Expected: FAIL — `_ExtractionPayload` not defined; tool schema only has `assertions`.

- [ ] **Step 3: Update the schema**

In `consistency_checker/extract/atomic_facts.py`, replace `TOOL_SCHEMA` and `_AssertionList` with:

```python
TOOL_NAME = "record_extraction"
TOOL_SCHEMA: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Record both atomic assertions and any definitions extracted from a chunk."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "assertions": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of atomic, decontextualised assertions extracted from the text. "
                    "May be empty if the chunk contains no verifiable claims."
                ),
            },
            "definitions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {
                            "type": "string",
                            "description": "The term being defined, as written.",
                        },
                        "definition_text": {
                            "type": "string",
                            "description": "What the term is said to mean (the right-hand side).",
                        },
                        "containing_sentence": {
                            "type": "string",
                            "description": (
                                "The full sentence (or clause) in which the definition "
                                "appears, copied verbatim from the source."
                            ),
                        },
                    },
                    "required": ["term", "definition_text", "containing_sentence"],
                },
                "description": (
                    "Definitions found in the text, whether formally ('X means …') or "
                    "informally ('by X we mean …', 'an eligible Y is …'). "
                    "May be empty."
                ),
            },
        },
        "required": ["assertions", "definitions"],
    },
}


class _DefinitionItem(BaseModel):
    """One definition extracted from a chunk."""

    term: str = Field(min_length=1)
    definition_text: str = Field(min_length=1)
    containing_sentence: str = Field(min_length=1)


class _ExtractionPayload(BaseModel):
    """Pydantic guard on the combined tool-use payload."""

    assertions: list[str] = Field(default_factory=list)
    definitions: list[_DefinitionItem] = Field(default_factory=list)
```

Note: the legacy `_AssertionList` is no longer referenced; remove it and its
import sites.

- [ ] **Step 4: Update the parser**

Replace `parse_tool_response` with:

```python
def parse_tool_response(response: Any) -> _ExtractionPayload:
    """Extract the combined assertion + definition payload from an Anthropic response."""
    blocks = getattr(response, "content", None) or []
    for block in blocks:
        block_type = getattr(block, "type", None)
        block_name = getattr(block, "name", None)
        if block_type == "tool_use" and block_name == TOOL_NAME:
            payload = getattr(block, "input", None) or {}
            return _ExtractionPayload.model_validate(payload)
    raise ValueError(f"No tool_use block named {TOOL_NAME!r} found in response")
```

- [ ] **Step 5: Update `_assertions_from_texts` consumers and the `Extractor` Protocol**

Replace `_assertions_from_texts` with a single helper that produces both claim and definition assertions:

```python
def _assertions_from_payload(
    chunk: Chunk, payload: _ExtractionPayload
) -> list[Assertion]:
    out: list[Assertion] = []
    for text in payload.assertions:
        if not text.strip():
            continue
        out.append(
            Assertion.build(
                chunk.doc_id,
                text,
                chunk_id=chunk.chunk_id,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
            )
        )
    for d in payload.definitions:
        if not d.containing_sentence.strip():
            continue
        out.append(
            Assertion.build(
                chunk.doc_id,
                d.containing_sentence,
                chunk_id=chunk.chunk_id,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                kind="definition",
                term=d.term,
                definition_text=d.definition_text,
            )
        )
    return out
```

Update `AnthropicExtractor.extract` to call `parse_tool_response` (which now returns `_ExtractionPayload`) and pass it to `_assertions_from_payload`.

- [ ] **Step 6: Run the test to verify it passes**

```bash
uv run pytest tests/extract/test_atomic_facts.py -v
```

Expected: PASS (existing tests in this file may need minor updates — see Step 7).

- [ ] **Step 7: Fix any failing existing tests in `test_atomic_facts.py`**

Existing tests probably mock `parse_tool_response` returning `list[str]`. Update those mocks to return `_ExtractionPayload(assertions=[...])`. Search:

```bash
uv run pytest tests/extract/test_atomic_facts.py -v
```

Update the failing tests inline. Common rewrites:
- `assert parse_tool_response(response) == ["..."]` → `assert parse_tool_response(response).assertions == ["..."]`
- Mocks that return `list[str]` → `_ExtractionPayload(assertions=[...])`

- [ ] **Step 8: Run the full hermetic suite**

```bash
uv run pytest -m "not slow and not live" -x
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add consistency_checker/extract/atomic_facts.py tests/extract/test_atomic_facts.py
git commit -m "feat(extract): combined extractor schema — atomic facts + definitions"
```

---

### Task 8: FixtureExtractor — definitions support

**Files:**
- Modify: `consistency_checker/extract/atomic_facts.py`
- Test: `tests/extract/test_atomic_facts.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to `tests/extract/test_atomic_facts.py`:

```python
from consistency_checker.corpus.chunker import Chunk
from consistency_checker.extract.atomic_facts import FixtureExtractor


def test_fixture_extractor_emits_definitions() -> None:
    chunk = Chunk(
        chunk_id="c1",
        doc_id="docA",
        text='"Borrower" means ABC Corp.',
        char_start=0,
        char_end=27,
    )
    extractor = FixtureExtractor(
        facts={},
        definitions={
            "c1": [
                {
                    "term": "Borrower",
                    "definition_text": "ABC Corp",
                    "containing_sentence": '"Borrower" means ABC Corp.',
                }
            ]
        },
    )
    out = extractor.extract(chunk)
    assert len(out) == 1
    assert out[0].kind == "definition"
    assert out[0].term == "Borrower"
    assert out[0].definition_text == "ABC Corp"
    assert out[0].assertion_text == '"Borrower" means ABC Corp.'


def test_fixture_extractor_back_compat_facts_only() -> None:
    """Old FixtureExtractor({chunk_id: [facts]}) form keeps working."""
    chunk = Chunk(chunk_id="c1", doc_id="docA", text="Revenue grew 12%.",
                  char_start=0, char_end=18)
    extractor = FixtureExtractor({"c1": ["Revenue grew 12 percent in fiscal 2025."]})
    out = extractor.extract(chunk)
    assert len(out) == 1
    assert out[0].kind == "claim"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/extract/test_atomic_facts.py -v -k fixture_extractor
```

Expected: FAIL — current `FixtureExtractor` takes positional `Mapping[str, list[str]]` only.

- [ ] **Step 3: Update FixtureExtractor**

Replace the `FixtureExtractor` class in `consistency_checker/extract/atomic_facts.py`:

```python
class FixtureExtractor:
    """Canned-response extractor for hermetic tests.

    Two call forms — the legacy positional ``FixtureExtractor({chunk_id: [facts]})``
    keeps working for older tests; the keyword form
    ``FixtureExtractor(facts=..., definitions=...)`` adds definition support.
    """

    def __init__(
        self,
        fixtures: Mapping[str, list[str]] | None = None,
        *,
        facts: Mapping[str, list[str]] | None = None,
        definitions: Mapping[str, list[Mapping[str, str]]] | None = None,
    ) -> None:
        if fixtures is not None and facts is not None:
            raise ValueError("pass either fixtures (legacy) or facts=, not both")
        self._facts: dict[str, list[str]] = dict(fixtures or facts or {})
        self._definitions: dict[str, list[Mapping[str, str]]] = dict(definitions or {})

    def extract(self, chunk: Chunk) -> list[Assertion]:
        payload = _ExtractionPayload(
            assertions=list(self._facts.get(chunk.chunk_id, [])),
            definitions=[
                _DefinitionItem(**d) for d in self._definitions.get(chunk.chunk_id, [])
            ],
        )
        return _assertions_from_payload(chunk, payload)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/extract/test_atomic_facts.py -v -k fixture_extractor
```

Expected: PASS.

- [ ] **Step 5: Run the full hermetic suite**

```bash
uv run pytest -m "not slow and not live" -x
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add consistency_checker/extract/atomic_facts.py tests/extract/test_atomic_facts.py
git commit -m "feat(extract): FixtureExtractor.definitions for hermetic tests"
```

---

### Task 9: Atomic-fact prompt update

**Files:**
- Modify: `consistency_checker/extract/prompts/atomic_facts.txt`
- Test: `tests/extract/test_atomic_facts.py` (light golden snapshot)

- [ ] **Step 1: Write the failing snapshot test**

Add to `tests/extract/test_atomic_facts.py`:

```python
from consistency_checker.extract.atomic_facts import render_prompt


def test_prompt_mentions_definitions() -> None:
    rendered = render_prompt("Some chunk text.")
    assert "definitions" in rendered.lower()
    assert "means" in rendered  # one of the canonical signaling verbs
    assert "for purposes of" in rendered.lower()  # informal phrasing example
    assert "record_extraction" in rendered
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/extract/test_atomic_facts.py::test_prompt_mentions_definitions -v
```

Expected: FAIL — current prompt mentions only assertions / `record_assertions`.

- [ ] **Step 3: Rewrite the prompt**

Replace the contents of `consistency_checker/extract/prompts/atomic_facts.txt`:

```text
You are extracting two kinds of items from a chunk of text: atomic assertions
and definitions.

## Atomic assertions

An atomic assertion is a short declarative sentence that:
- Encodes exactly one factual claim (no compound conjunctions).
- Can be verified independently of any surrounding context.
- Preserves numeric values, dates, entity names, and qualifying conditions verbatim from the source.
- Stands on its own — no pronouns or deictic references that point outside the assertion itself.

Examples of good atomic assertions:
- "Revenue from the Alpha product line grew 12% year-over-year in fiscal 2025."
- "The contract is governed by the laws of the State of Delaware."
- "The Borrower's debt-to-EBITDA ratio must not exceed 3.5x as of any quarter end."

Examples of what NOT to do:
- "It grew significantly." (pronoun "it"; no entity; vague qualifier)
- "Revenue grew 12% and operating margin improved." (compound — two claims)
- "Things looked positive." (no verifiable claim)

If the chunk contains no verifiable claims (e.g. headers, navigation, boilerplate),
return an empty assertions list.

## Definitions

A definition is a sentence where the author tells you what a term means.
Capture both formal definitions and informal ones:

- Formal: `"X" means …`, `X shall mean …`, `X refers to …`, `(the "X")`, `"X" is defined as …`.
- Informal: `by X we mean …`, `an eligible Y is …`, `for purposes of this policy, X is …`,
  `we consider X to be …`.
- Implicit / descriptive: a sentence whose function is to tell the reader what
  category of thing X is (e.g. "An eligible employee is someone who has
  completed 90 days of continuous service.").

For each definition, return:
- `term`: the term being defined, as written (do not normalise case).
- `definition_text`: the right-hand side — what the term means.
- `containing_sentence`: the full sentence (or clause) that contains the
  definition, copied verbatim from the source. This is what gets stored as
  the assertion text and is what the judge will see.

If the same sentence is also an atomic claim, return it in BOTH lists
(definitions for the definition detector, assertions for the contradiction
detector). They are independent.

If the chunk contains no definitions, return an empty definitions list.

## Output

Text chunk:
---
{chunk_text}
---

Call the ``record_extraction`` tool with both lists.
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/extract/test_atomic_facts.py::test_prompt_mentions_definitions -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/extract/prompts/atomic_facts.txt tests/extract/test_atomic_facts.py
git commit -m "feat(extract): extend extractor prompt to harvest definitions"
```

---

### Task 10: Pipeline `ingest()` — no code change needed, just verify

**Files:**
- Test: `tests/test_pipeline_ingest.py` (extend) — verify definitions flow end-to-end through ingest

- [ ] **Step 1: Write the failing test**

Add to (or create) `tests/test_pipeline_ingest.py`:

```python
from pathlib import Path

from consistency_checker.config import Config
from consistency_checker.extract.atomic_facts import FixtureExtractor
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.index.faiss_store import FaissStore
from consistency_checker.pipeline import ingest
from tests.conftest import HashEmbedder


def test_ingest_persists_definitions(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "term-sheet.txt").write_text(
        '"Borrower" means ABC Corp and its Subsidiaries.\n'
    )
    config = Config(
        corpus_dir=corpus,
        judge_provider="fixture",
        judge_model="test",
        data_dir=tmp_path / "store",
        log_dir=tmp_path / "logs",
        embedder_model="hash",
        nli_model="fixture",
    )
    store = AssertionStore(config.data_dir / "store.db")
    store.migrate()
    faiss = FaissStore.open_or_create(config.data_dir / "faiss.idx", dim=64)
    # The fixture extractor needs the chunk id — chunk it the way ingest will.
    # Easier: configure the fixture to match every chunk id by inspecting
    # corpus during the test (or use a wildcard fixture form). For this test
    # we rely on a single-chunk file and look up its chunk id after the fact.
    extractor = FixtureExtractor(
        facts={},
        # The chunker emits deterministic chunk ids based on doc + char span;
        # accept any chunk by using a "match-all" form below.
        definitions={},
    )
    # Replace with a thin adapter that returns one definition for every chunk:
    def extract_one_definition(chunk):
        from consistency_checker.extract.schema import Assertion
        return [
            Assertion.build(
                chunk.doc_id,
                '"Borrower" means ABC Corp and its Subsidiaries.',
                chunk_id=chunk.chunk_id,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                kind="definition",
                term="Borrower",
                definition_text="ABC Corp and its Subsidiaries",
            )
        ]
    extractor.extract = extract_one_definition  # type: ignore[assignment]
    embedder = HashEmbedder(dim=64)

    ingest(config, store=store, faiss_store=faiss, extractor=extractor, embedder=embedder)

    defs = list(store.iter_definitions())
    assert len(defs) >= 1
    assert defs[0].term == "Borrower"
    store.close()
```

- [ ] **Step 2: Run the test to verify it passes (or fails)**

```bash
uv run pytest tests/test_pipeline_ingest.py::test_ingest_persists_definitions -v
```

Expected: PASS — no pipeline change should be needed because `ingest()` already calls `store.add_assertions()` on whatever the extractor returns, and we wired definitions through the store in Task 4.

- [ ] **Step 3: If the test fails**, the diff between actual and expected tells you which pipeline call is dropping the new fields. Fix in `pipeline.py` (most likely a tighter copy is shadowing the new columns) and re-run.

- [ ] **Step 4: Commit**

```bash
git add tests/test_pipeline_ingest.py
git commit -m "test(pipeline): definitions survive end-to-end ingest"
```

---

## Phase 3 — Definition Checker

### Task 11: Definition judge payload + provider Protocol

**Files:**
- Create: `consistency_checker/check/providers/definition_base.py`
- Test: `tests/check/test_definition_judge_payload.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/check/test_definition_judge_payload.py`:

```python
import pytest
from pydantic import ValidationError

from consistency_checker.check.providers.definition_base import (
    DefinitionJudgePayload,
    DefinitionVerdictLabel,
)


def test_payload_accepts_three_verdicts() -> None:
    for v in ("definition_consistent", "definition_divergent", "uncertain"):
        p = DefinitionJudgePayload(
            verdict=v,  # type: ignore[arg-type]
            confidence=0.8,
            rationale="x",
            evidence_spans=[],
        )
        assert p.verdict == v


def test_payload_rejects_unknown_verdict() -> None:
    with pytest.raises(ValidationError):
        DefinitionJudgePayload(
            verdict="contradiction",  # type: ignore[arg-type]
            confidence=0.5,
            rationale="x",
            evidence_spans=[],
        )


def test_payload_rejects_extras() -> None:
    with pytest.raises(ValidationError):
        DefinitionJudgePayload.model_validate(
            {
                "verdict": "definition_consistent",
                "confidence": 0.5,
                "rationale": "x",
                "evidence_spans": [],
                "extra_field": "nope",
            }
        )


def test_payload_requires_nonempty_rationale() -> None:
    with pytest.raises(ValidationError):
        DefinitionJudgePayload(
            verdict="uncertain",
            confidence=0.0,
            rationale="",
            evidence_spans=[],
        )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/check/test_definition_judge_payload.py -v
```

Expected: FAIL — module not defined.

- [ ] **Step 3: Implement the payload**

Create `consistency_checker/check/providers/definition_base.py`:

```python
"""Provider-agnostic shapes for the definition-inconsistency judge.

Mirrors :mod:`consistency_checker.check.providers.base` — a strict Pydantic
schema plus a Protocol — but with the three-verdict vocabulary specific to the
definition detector. Keeping the surface narrow ensures an LLM in the
definition path can't accidentally claim a contradiction verdict and vice
versa.
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

DefinitionVerdictLabel = Literal[
    "definition_consistent",
    "definition_divergent",
    "uncertain",
]

#: Verdicts that count as a confirmed definition inconsistency in run totals
#: and reports. Single-element today; kept as a frozenset for parallel
#: structure with ``CONTRADICTION_VERDICTS``.
DEFINITION_INCONSISTENCY_VERDICTS: frozenset[str] = frozenset({"definition_divergent"})


class DefinitionJudgePayload(BaseModel):
    """Strict schema every definition-judge provider must satisfy."""

    model_config = ConfigDict(extra="forbid")

    verdict: DefinitionVerdictLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    evidence_spans: list[str] = Field(default_factory=list)


class DefinitionJudgeProvider(Protocol):
    """Anything that can produce a validated :class:`DefinitionJudgePayload`."""

    def request_payload(self, system: str, user: str) -> DefinitionJudgePayload: ...
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/check/test_definition_judge_payload.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/check/providers/definition_base.py tests/check/test_definition_judge_payload.py
git commit -m "feat(check): DefinitionJudgePayload + Protocol"
```

---

### Task 12: Definition judge prompts

**Files:**
- Create: `consistency_checker/check/prompts/definition_judge_system.txt`
- Create: `consistency_checker/check/prompts/definition_judge_user.txt`
- Test: `tests/check/test_definition_judge.py` (new file; rendering test)

- [ ] **Step 1: Write the failing test**

Create `tests/check/test_definition_judge.py`:

```python
from consistency_checker.check.definition_judge import (
    render_definition_system_prompt,
    render_definition_user_prompt,
)
from consistency_checker.extract.schema import Assertion


def test_definition_system_prompt_explains_verdicts() -> None:
    s = render_definition_system_prompt()
    assert "definition_consistent" in s
    assert "definition_divergent" in s
    assert "uncertain" in s


def test_definition_user_prompt_renders_both_definitions() -> None:
    a = Assertion.build(
        "docA",
        '"MAE" means a material adverse effect on the Borrower\'s business.',
        kind="definition",
        term="MAE",
        definition_text="a material adverse effect on the Borrower's business",
    )
    b = Assertion.build(
        "docB",
        '"MAE" means an effect that materially impairs the Borrower\'s ability to perform.',
        kind="definition",
        term="MAE",
        definition_text="an effect that materially impairs the Borrower's ability to perform",
    )
    user = render_definition_user_prompt(a, b)
    assert "MAE" in user
    assert "docA" in user
    assert "docB" in user
    assert "material adverse effect on the Borrower's business" in user
    assert "materially impairs" in user
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/check/test_definition_judge.py::test_definition_system_prompt_explains_verdicts -v
```

Expected: FAIL — module not defined.

- [ ] **Step 3: Write the system prompt**

Create `consistency_checker/check/prompts/definition_judge_system.txt`:

```text
You are comparing two definitions of the same term to decide whether they
express the same meaning or materially diverge.

Return one of three verdicts:

- `definition_consistent` — the two definitions describe the same concept.
  Minor wording differences, additional examples, or rephrasing that does
  not change the scope or applicability are *consistent*.
- `definition_divergent` — the two definitions describe different concepts,
  or the same concept with materially different scope, inclusions, exclusions,
  or thresholds. A reasonable reader following one definition would reach a
  different conclusion than a reader following the other.
- `uncertain` — you cannot decide given the text provided. Use sparingly.

Be conservative. Surface-level wording differences are NOT divergence. Material
scope shifts, different inclusions/exclusions, different thresholds, or
different objects of reference ARE divergence.

Return your verdict using the structured response schema. The rationale must
cite the specific phrases that justify the verdict; the `evidence_spans` list
should contain the exact substrings from each definition that drove your call.
```

- [ ] **Step 4: Write the user prompt**

Create `consistency_checker/check/prompts/definition_judge_user.txt`:

```text
Compare these two definitions of the same term and decide whether they
materially diverge.

## Term

{term}

## Definition A (from document {doc_a_id})

{assertion_a_text}

## Definition B (from document {doc_b_id})

{assertion_b_text}

## Decision

Is Definition A and Definition B consistent or materially divergent?
Respond using the structured tool.
```

- [ ] **Step 5: Stub `consistency_checker/check/definition_judge.py` so the import test passes**

Create `consistency_checker/check/definition_judge.py` with just the renderers (the rest of the module comes in Task 13):

```python
"""Definition-inconsistency judge.

Mirrors :mod:`consistency_checker.check.llm_judge` but for the definition
detector — same retry-with-degraded-fallback pattern, different prompts,
different verdict vocabulary, different payload type.
"""

from __future__ import annotations

from pathlib import Path

from consistency_checker.extract.schema import Assertion

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "definition_judge_system.txt"
USER_PROMPT_PATH = PROMPTS_DIR / "definition_judge_user.txt"


def render_definition_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def render_definition_user_prompt(a: Assertion, b: Assertion) -> str:
    """Render the user prompt with both definitions and the shared term.

    ``a.term`` and ``b.term`` should canonicalise to the same value (the
    DefinitionChecker only judges pairs within a term group). The displayed
    term is ``a.term`` for stability; the prompt does not need both.
    """
    if a.term is None or b.term is None:
        raise ValueError("definition judge requires both assertions to have a `term`")
    template = USER_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template.replace("{term}", a.term)
        .replace("{doc_a_id}", a.doc_id)
        .replace("{doc_b_id}", b.doc_id)
        .replace("{assertion_a_text}", a.assertion_text)
        .replace("{assertion_b_text}", b.assertion_text)
    )
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
uv run pytest tests/check/test_definition_judge.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add consistency_checker/check/prompts/definition_judge_*.txt consistency_checker/check/definition_judge.py tests/check/test_definition_judge.py
git commit -m "feat(check): definition judge prompts + prompt renderers"
```

---

### Task 13: DefinitionJudge — FixtureDefinitionJudge + LLMDefinitionJudge

**Files:**
- Modify: `consistency_checker/check/definition_judge.py`
- Test: `tests/check/test_definition_judge.py` (extend)

- [ ] **Step 1: Write the failing tests**

Add to `tests/check/test_definition_judge.py`:

```python
from consistency_checker.check.definition_judge import (
    DefinitionJudgeVerdict,
    FixtureDefinitionJudge,
    LLMDefinitionJudge,
    definition_uncertain_fallback,
)
from consistency_checker.check.providers.definition_base import DefinitionJudgePayload


def _mae_pair() -> tuple[Assertion, Assertion]:
    a = Assertion.build(
        "docA",
        '"MAE" means a material adverse effect on the Borrower\'s business.',
        kind="definition",
        term="MAE",
        definition_text="a material adverse effect on the Borrower's business",
    )
    b = Assertion.build(
        "docB",
        '"MAE" means an effect that materially impairs the Borrower\'s ability to perform.',
        kind="definition",
        term="MAE",
        definition_text="an effect that materially impairs the Borrower's ability to perform",
    )
    return a, b


def test_fixture_definition_judge_returns_canned_verdict() -> None:
    a, b = _mae_pair()
    verdict = DefinitionJudgeVerdict(
        assertion_a_id=a.assertion_id,
        assertion_b_id=b.assertion_id,
        verdict="definition_divergent",
        confidence=0.9,
        rationale="A scopes business; B scopes performance.",
        evidence_spans=["business", "ability to perform"],
    )
    judge = FixtureDefinitionJudge({(min(a.assertion_id, b.assertion_id),
                                     max(a.assertion_id, b.assertion_id)): verdict})
    out = judge.judge(a, b)
    assert out.verdict == "definition_divergent"


def test_fixture_definition_judge_falls_back_uncertain() -> None:
    a, b = _mae_pair()
    judge = FixtureDefinitionJudge({})
    out = judge.judge(a, b)
    assert out.verdict == "uncertain"
    assert out.confidence == 0.0


class _StubProvider:
    def __init__(self, payload: DefinitionJudgePayload) -> None:
        self.payload = payload
        self.calls = 0

    def request_payload(self, system: str, user: str) -> DefinitionJudgePayload:
        self.calls += 1
        return self.payload


def test_llm_definition_judge_round_trips_payload() -> None:
    a, b = _mae_pair()
    payload = DefinitionJudgePayload(
        verdict="definition_divergent",
        confidence=0.85,
        rationale="scope shift",
        evidence_spans=["business", "ability to perform"],
    )
    provider = _StubProvider(payload)
    judge = LLMDefinitionJudge(provider)
    out = judge.judge(a, b)
    assert out.verdict == "definition_divergent"
    assert out.confidence == 0.85
    assert provider.calls == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/check/test_definition_judge.py -v -k "fixture_definition_judge or llm_definition_judge"
```

Expected: FAIL — classes not defined.

- [ ] **Step 3: Extend `definition_judge.py`**

Append to `consistency_checker/check/definition_judge.py`:

```python
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import ValidationError

from consistency_checker.check.providers.definition_base import (
    DefinitionJudgePayload,
    DefinitionJudgeProvider,
    DefinitionVerdictLabel,
)
from consistency_checker.logging_setup import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DefinitionJudgeVerdict:
    """Verdict for one definition pair plus its provenance."""

    assertion_a_id: str
    assertion_b_id: str
    verdict: DefinitionVerdictLabel
    confidence: float
    rationale: str
    evidence_spans: list[str] = field(default_factory=list)

    @classmethod
    def from_payload(
        cls, a: Assertion, b: Assertion, payload: DefinitionJudgePayload
    ) -> "DefinitionJudgeVerdict":
        return cls(
            assertion_a_id=a.assertion_id,
            assertion_b_id=b.assertion_id,
            verdict=payload.verdict,
            confidence=payload.confidence,
            rationale=payload.rationale,
            evidence_spans=list(payload.evidence_spans),
        )


def definition_uncertain_fallback(
    a: Assertion, b: Assertion, reason: str
) -> DefinitionJudgeVerdict:
    return DefinitionJudgeVerdict(
        assertion_a_id=a.assertion_id,
        assertion_b_id=b.assertion_id,
        verdict="uncertain",
        confidence=0.0,
        rationale=f"Definition judge degraded to uncertain: {reason}",
        evidence_spans=[],
    )


class DefinitionJudge(Protocol):
    """Anything that produces a DefinitionJudgeVerdict for a pair of definitions."""

    def judge(self, a: Assertion, b: Assertion) -> DefinitionJudgeVerdict: ...


class FixtureDefinitionJudge:
    """Returns canned verdicts keyed by the canonical assertion-id pair."""

    def __init__(
        self, fixtures: Mapping[tuple[str, str], DefinitionJudgeVerdict]
    ) -> None:
        self._fixtures = dict(fixtures)

    def judge(self, a: Assertion, b: Assertion) -> DefinitionJudgeVerdict:
        key = (min(a.assertion_id, b.assertion_id), max(a.assertion_id, b.assertion_id))
        if key in self._fixtures:
            return self._fixtures[key]
        return definition_uncertain_fallback(a, b, reason="no fixture configured for pair")


class LLMDefinitionJudge:
    """Provider-backed judge with retry-on-malformed and degraded fallback."""

    def __init__(
        self, provider: DefinitionJudgeProvider, *, max_retries: int = 2
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self._provider = provider
        self._max_retries = max_retries

    def judge(self, a: Assertion, b: Assertion) -> DefinitionJudgeVerdict:
        system = render_definition_system_prompt()
        user = render_definition_user_prompt(a, b)
        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            try:
                payload = self._provider.request_payload(system, user)
                return DefinitionJudgeVerdict.from_payload(a, b, payload)
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)
                _log.warning(
                    "Definition judge attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    last_error,
                )
        return definition_uncertain_fallback(a, b, reason=last_error or "unknown error")
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/check/test_definition_judge.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/check/definition_judge.py tests/check/test_definition_judge.py
git commit -m "feat(check): DefinitionJudge — Fixture + LLM backends"
```

---

### Task 14: DefinitionChecker — group + pair + judge

**Files:**
- Create: `consistency_checker/check/definition_checker.py`
- Test: `tests/check/test_definition_checker.py` (new file)

- [ ] **Step 1: Write the failing tests**

Create `tests/check/test_definition_checker.py`:

```python
from consistency_checker.check.definition_checker import (
    DefinitionChecker,
    DefinitionPair,
)
from consistency_checker.check.definition_judge import (
    DefinitionJudgeVerdict,
    FixtureDefinitionJudge,
)
from consistency_checker.extract.schema import Assertion


def _def(doc: str, term: str, text: str) -> Assertion:
    return Assertion.build(
        doc,
        f'"{term}" means {text}.',
        kind="definition",
        term=term,
        definition_text=text,
    )


def test_singleton_term_emits_zero_pairs() -> None:
    defs = [_def("docA", "Borrower", "ABC Corp")]
    checker = DefinitionChecker(judge=FixtureDefinitionJudge({}))
    findings = list(checker.find_inconsistencies(defs))
    assert findings == []


def test_pair_within_term_group_is_judged() -> None:
    a = _def("docA", "Borrower", "ABC Corp")
    b = _def("docB", "Borrower", "ABC Corp and its Subsidiaries")
    verdict = DefinitionJudgeVerdict(
        assertion_a_id=min(a.assertion_id, b.assertion_id),
        assertion_b_id=max(a.assertion_id, b.assertion_id),
        verdict="definition_divergent",
        confidence=0.9,
        rationale="scope shift",
        evidence_spans=[],
    )
    judge = FixtureDefinitionJudge({
        (min(a.assertion_id, b.assertion_id), max(a.assertion_id, b.assertion_id)): verdict,
    })
    checker = DefinitionChecker(judge=judge)
    findings = list(checker.find_inconsistencies([a, b]))
    assert len(findings) == 1
    assert findings[0].verdict.verdict == "definition_divergent"


def test_different_canonical_terms_do_not_pair() -> None:
    a = _def("docA", "Borrower", "ABC Corp")
    b = _def("docB", "Lender", "First Bank")
    checker = DefinitionChecker(judge=FixtureDefinitionJudge({}))
    findings = list(checker.find_inconsistencies([a, b]))
    assert findings == []


def test_plurals_and_articles_group_together() -> None:
    a = _def("docA", "Borrower", "ABC Corp")
    b = _def("docB", "the Borrowers", "ABC Corp and its Subsidiaries")
    judge = FixtureDefinitionJudge({})  # uncertain fallback, but still judged
    checker = DefinitionChecker(judge=judge)
    findings = list(checker.find_inconsistencies([a, b]))
    assert len(findings) == 1
    assert findings[0].verdict.verdict == "uncertain"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/check/test_definition_checker.py -v
```

Expected: FAIL — module not defined.

- [ ] **Step 3: Implement the checker**

Create `consistency_checker/check/definition_checker.py`:

```python
"""Definition-inconsistency detector.

Groups definition assertions by canonical term, enumerates unordered pairs
within each group, and asks the definition judge whether each pair is
consistent or divergent. Findings flow into the existing ``findings`` table
via the audit logger under ``detector_type='definition_inconsistency'``.

Unlike the contradiction pipeline, this stage skips the NLI gate. The DeBERTa
gate is contradiction-tuned and unhelpful when comparing two definitions of
the same concept — the term-group gate above is both cheaper and more precise
for this question.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations

from consistency_checker.check.definition_judge import (
    DefinitionJudge,
    DefinitionJudgeVerdict,
)
from consistency_checker.check.definition_terms import canonicalize_term
from consistency_checker.extract.schema import Assertion


@dataclass(frozen=True, slots=True)
class DefinitionPair:
    """A pair of definition assertions whose canonical terms match."""

    a: Assertion
    b: Assertion
    canonical_term: str


@dataclass(frozen=True, slots=True)
class DefinitionFinding:
    """One definition-pair verdict — what the checker emits, before audit."""

    pair: DefinitionPair
    verdict: DefinitionJudgeVerdict


def _group_by_canonical_term(
    definitions: Iterable[Assertion],
) -> dict[str, list[Assertion]]:
    groups: dict[str, list[Assertion]] = {}
    for d in definitions:
        if d.kind != "definition" or d.term is None:
            continue
        canonical = canonicalize_term(d.term)
        if not canonical:
            continue
        groups.setdefault(canonical, []).append(d)
    return groups


def _enumerate_pairs(
    groups: dict[str, list[Assertion]],
) -> Iterator[DefinitionPair]:
    for canonical, assertions in groups.items():
        if len(assertions) < 2:
            continue
        # Stable ordering: by assertion_id, so combinations are deterministic.
        ordered = sorted(assertions, key=lambda a: a.assertion_id)
        for a, b in combinations(ordered, 2):
            yield DefinitionPair(a=a, b=b, canonical_term=canonical)


class DefinitionChecker:
    """Orchestrates term-grouping → pair-enumeration → judge for definitions."""

    def __init__(self, *, judge: DefinitionJudge) -> None:
        self._judge = judge

    def find_inconsistencies(
        self, definitions: Sequence[Assertion]
    ) -> Iterator[DefinitionFinding]:
        groups = _group_by_canonical_term(definitions)
        for pair in _enumerate_pairs(groups):
            verdict = self._judge.judge(pair.a, pair.b)
            yield DefinitionFinding(pair=pair, verdict=verdict)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/check/test_definition_checker.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/check/definition_checker.py tests/check/test_definition_checker.py
git commit -m "feat(check): DefinitionChecker — group, pair, judge"
```

---

## Phase 4 — Audit + Pipeline

### Task 15: AuditLogger.record_definition_finding

**Files:**
- Modify: `consistency_checker/audit/logger.py`
- Test: `tests/audit/test_logger.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to `tests/audit/test_logger.py`:

```python
def test_record_definition_finding_writes_detector_type(tmp_path: Path) -> None:
    from consistency_checker.audit.logger import AuditLogger
    from consistency_checker.check.definition_checker import (
        DefinitionFinding,
        DefinitionPair,
    )
    from consistency_checker.check.definition_judge import DefinitionJudgeVerdict
    from consistency_checker.extract.schema import Assertion, Document
    from consistency_checker.index.assertion_store import AssertionStore

    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    store.add_document(Document(doc_id="docA", source_path="/A.txt"))
    store.add_document(Document(doc_id="docB", source_path="/B.txt"))
    a = Assertion.build("docA", '"X" means foo.', kind="definition", term="X", definition_text="foo")
    b = Assertion.build("docB", '"X" means bar.', kind="definition", term="X", definition_text="bar")
    store.add_assertions([a, b])
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    finding = DefinitionFinding(
        pair=DefinitionPair(a=a, b=b, canonical_term="x"),
        verdict=DefinitionJudgeVerdict(
            assertion_a_id=min(a.assertion_id, b.assertion_id),
            assertion_b_id=max(a.assertion_id, b.assertion_id),
            verdict="definition_divergent",
            confidence=0.9,
            rationale="scope shift",
            evidence_spans=["foo", "bar"],
        ),
    )
    logger.record_definition_finding(run_id, finding=finding)
    rows = store._conn.execute(
        "SELECT detector_type, judge_verdict FROM findings WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["detector_type"] == "definition_inconsistency"
    assert rows[0]["judge_verdict"] == "definition_divergent"
    store.close()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/audit/test_logger.py::test_record_definition_finding_writes_detector_type -v
```

Expected: FAIL — method not defined.

- [ ] **Step 3: Implement the method**

In `consistency_checker/audit/logger.py`, add this method to `AuditLogger`:

```python
def record_definition_finding(
    self,
    run_id: str,
    *,
    finding: "DefinitionFinding",
) -> str:
    """Persist a definition-inconsistency finding into the shared `findings` table."""
    a_id = finding.pair.a.assertion_id
    b_id = finding.pair.b.assertion_id
    finding_id = hash_id(run_id, "definition", a_id, b_id)
    spans_json = json.dumps(finding.verdict.evidence_spans)
    with self._conn:
        self._conn.execute(
            "INSERT OR REPLACE INTO findings ("
            "finding_id, run_id, assertion_a_id, assertion_b_id, "
            "gate_score, nli_label, nli_p_contradiction, nli_p_entailment, nli_p_neutral, "
            "judge_verdict, judge_confidence, judge_rationale, evidence_spans_json, "
            "detector_type"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                finding_id,
                run_id,
                a_id,
                b_id,
                None,  # no gate_score for definition pairs
                None, None, None, None,  # NLI columns intentionally null
                finding.verdict.verdict,
                finding.verdict.confidence,
                finding.verdict.rationale,
                spans_json,
                "definition_inconsistency",
            ),
        )
    return finding_id
```

Use a lazy `TYPE_CHECKING` import for `DefinitionFinding` to avoid the circular import:

```python
if TYPE_CHECKING:
    from consistency_checker.check.definition_checker import DefinitionFinding
```

(Add `from typing import TYPE_CHECKING` to the existing imports if it's not already there.)

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/audit/test_logger.py::test_record_definition_finding_writes_detector_type -v
```

Expected: PASS.

- [ ] **Step 5: Run the hermetic suite to catch regressions**

```bash
uv run pytest -m "not slow and not live" -x
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add consistency_checker/audit/logger.py tests/audit/test_logger.py
git commit -m "feat(audit): record definition-inconsistency findings"
```

---

### Task 16: Pipeline `check()` — run the definition stage

**Files:**
- Modify: `consistency_checker/pipeline.py`
- Test: `tests/test_pipeline_check.py` (extend or create)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline_check.py` (or create alongside the existing pipeline tests):

```python
from pathlib import Path

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.check.definition_checker import DefinitionChecker
from consistency_checker.check.definition_judge import (
    DefinitionJudgeVerdict,
    FixtureDefinitionJudge,
)
from consistency_checker.check.gate import AnnGate
from consistency_checker.check.llm_judge import FixtureJudge
from consistency_checker.check.nli_checker import FixtureNliChecker
from consistency_checker.config import Config
from consistency_checker.extract.schema import Assertion, Document
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.index.faiss_store import FaissStore
from consistency_checker.pipeline import check


def test_check_runs_definition_stage_and_logs_findings(tmp_path: Path) -> None:
    config = Config(
        corpus_dir=tmp_path / "corpus",
        judge_provider="fixture",
        judge_model="test",
        data_dir=tmp_path / "store",
        log_dir=tmp_path / "logs",
        embedder_model="hash",
        nli_model="fixture",
        gate_similarity_threshold=-1.0,
    )
    store = AssertionStore(config.data_dir / "store.db")
    store.migrate()
    store.add_document(Document(doc_id="docA", source_path="/A.txt"))
    store.add_document(Document(doc_id="docB", source_path="/B.txt"))
    a = Assertion.build(
        "docA", '"MAE" means A.', kind="definition", term="MAE", definition_text="A"
    )
    b = Assertion.build(
        "docB", '"MAE" means B.', kind="definition", term="MAE", definition_text="B"
    )
    store.add_assertions([a, b])
    faiss = FaissStore.open_or_create(config.data_dir / "faiss.idx", dim=64)
    logger = AuditLogger(store)
    run_id = logger.begin_run()

    definition_checker = DefinitionChecker(
        judge=FixtureDefinitionJudge({
            (min(a.assertion_id, b.assertion_id), max(a.assertion_id, b.assertion_id)):
                DefinitionJudgeVerdict(
                    assertion_a_id=min(a.assertion_id, b.assertion_id),
                    assertion_b_id=max(a.assertion_id, b.assertion_id),
                    verdict="definition_divergent",
                    confidence=0.9,
                    rationale="scope shift",
                    evidence_spans=["A", "B"],
                ),
        })
    )

    result = check(
        config,
        store=store,
        faiss_store=faiss,
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
        audit_logger=logger,
        run_id=run_id,
        definition_checker=definition_checker,
    )
    assert result.n_definition_findings == 1

    rows = store._conn.execute(
        "SELECT detector_type, judge_verdict FROM findings WHERE run_id = ? "
        "AND detector_type = 'definition_inconsistency'",
        (run_id,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["judge_verdict"] == "definition_divergent"
    store.close()


def test_check_without_definition_checker_is_unchanged(tmp_path: Path) -> None:
    """Passing definition_checker=None preserves prior behavior."""
    # Reuse the existing baseline test pattern — assert n_definition_findings == 0.
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/test_pipeline_check.py::test_check_runs_definition_stage_and_logs_findings -v
```

Expected: FAIL — `definition_checker` kwarg unknown.

- [ ] **Step 3: Wire the stage into `pipeline.check`**

In `consistency_checker/pipeline.py`:

(a) Extend `CheckResult`:

```python
@dataclass(frozen=True, slots=True)
class CheckResult:
    run_id: str
    n_assertions: int
    n_pairs_gated: int
    n_pairs_judged: int
    n_findings: int
    n_triangles_judged: int = 0
    n_multi_party_findings: int = 0
    n_definition_pairs_judged: int = 0
    n_definition_findings: int = 0
```

(b) Add an import:

```python
from consistency_checker.check.definition_checker import (
    DefinitionChecker,
    DefinitionFinding,
)
from consistency_checker.check.providers.definition_base import (
    DEFINITION_INCONSISTENCY_VERDICTS,
)
```

(c) Add a `definition_checker` kwarg to `check()` and a stage that runs after the multi-party pass:

```python
def check(
    config: Config,
    *,
    store: AssertionStore,
    faiss_store: FaissStore,
    nli_checker: NliChecker,
    judge: Judge,
    audit_logger: AuditLogger,
    gate: CandidateGate | None = None,
    multi_party_judge: MultiPartyJudge | None = None,
    definition_checker: DefinitionChecker | None = None,
    run_id: str,
) -> CheckResult:
    ...
    # existing body, up through end_run call

    n_definition_pairs_judged = 0
    n_definition_findings = 0
    if definition_checker is not None:
        n_definition_pairs_judged, n_definition_findings = _run_definition_pass(
            store=store,
            checker=definition_checker,
            audit_logger=audit_logger,
            run_id=run_id,
        )

    audit_logger.end_run(
        run_id,
        n_assertions=n_assertions,
        n_pairs_gated=n_pairs_gated,
        n_pairs_judged=n_pairs_judged,
        n_findings=n_findings + n_definition_findings,  # roll up
    )
    ...
    return CheckResult(
        ...,
        n_definition_pairs_judged=n_definition_pairs_judged,
        n_definition_findings=n_definition_findings,
    )
```

(d) Add the helper:

```python
def _run_definition_pass(
    *,
    store: AssertionStore,
    checker: DefinitionChecker,
    audit_logger: AuditLogger,
    run_id: str,
) -> tuple[int, int]:
    """Run the definition checker over all stored definitions and log findings."""
    definitions = list(store.iter_definitions())
    n_judged = 0
    n_findings = 0
    for finding in checker.find_inconsistencies(definitions):
        audit_logger.record_definition_finding(run_id, finding=finding)
        n_judged += 1
        if finding.verdict.verdict in DEFINITION_INCONSISTENCY_VERDICTS:
            n_findings += 1
    return n_judged, n_findings
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_pipeline_check.py -v
```

Expected: PASS.

- [ ] **Step 5: Run the full hermetic suite**

```bash
uv run pytest -m "not slow and not live" -x
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add consistency_checker/pipeline.py tests/test_pipeline_check.py
git commit -m "feat(pipeline): run definition stage after contradiction stage"
```

---

## Phase 5 — Reporting, UI, CLI

### Task 17: Markdown report — Definition Inconsistencies section

**Files:**
- Modify: `consistency_checker/audit/report.py`
- Test: `tests/audit/test_report.py` (extend)

- [ ] **Step 1: Read the existing report module**

```bash
sed -n '1,80p' consistency_checker/audit/report.py
```

Understand how the existing contradictions section is rendered before deciding where to insert the new section.

- [ ] **Step 2: Write the failing test**

Add to `tests/audit/test_report.py`:

```python
def test_report_renders_definition_section(tmp_path: Path) -> None:
    from consistency_checker.audit.logger import AuditLogger
    from consistency_checker.audit.report import render_report
    from consistency_checker.extract.schema import Assertion, Document
    from consistency_checker.index.assertion_store import AssertionStore

    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    store.add_document(Document(doc_id="docA", source_path="A.txt", title="Term Sheet"))
    store.add_document(Document(doc_id="docB", source_path="B.txt", title="Credit Agreement"))
    a = Assertion.build("docA", '"MAE" means A.', kind="definition", term="MAE", definition_text="A")
    b = Assertion.build("docB", '"MAE" means B.', kind="definition", term="MAE", definition_text="B")
    store.add_assertions([a, b])
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    # Insert a definition finding directly so we don't have to run the full pipeline.
    store._conn.execute(
        "INSERT INTO findings (finding_id, run_id, assertion_a_id, assertion_b_id, "
        "judge_verdict, judge_confidence, judge_rationale, evidence_spans_json, detector_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("fid1", run_id, a.assertion_id, b.assertion_id,
         "definition_divergent", 0.9, "scope shift", "[]", "definition_inconsistency"),
    )
    store._conn.commit()
    logger.end_run(run_id)

    text = render_report(store=store, audit_logger=logger, run_id=run_id)
    assert "Definition inconsistencies" in text
    assert "MAE" in text
    assert "definition_divergent" in text
    store.close()
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
uv run pytest tests/audit/test_report.py::test_report_renders_definition_section -v
```

Expected: FAIL — section not in output.

- [ ] **Step 4: Add the section**

In `consistency_checker/audit/report.py`, add (insertion point: after the contradictions / multi-party sections, before the run-stats footer; mirror the formatting helpers already used there):

```python
def _render_definition_section(
    *,
    store: AssertionStore,
    audit_logger: AuditLogger,
    run_id: str,
) -> str:
    rows = list(audit_logger._store._conn.execute(
        "SELECT f.assertion_a_id, f.assertion_b_id, f.judge_verdict, "
        "f.judge_rationale, a.term, da.title AS doc_a_title, db.title AS doc_b_title, "
        "a.assertion_text AS def_a_text, b.assertion_text AS def_b_text "
        "FROM findings f "
        "JOIN assertions a ON a.assertion_id = f.assertion_a_id "
        "JOIN assertions b ON b.assertion_id = f.assertion_b_id "
        "JOIN documents da ON da.doc_id = a.doc_id "
        "JOIN documents db ON db.doc_id = b.doc_id "
        "WHERE f.run_id = ? AND f.detector_type = 'definition_inconsistency' "
        "AND f.judge_verdict = 'definition_divergent' "
        "ORDER BY a.term, f.finding_id",
        (run_id,),
    ))
    if not rows:
        return ""
    lines = ["## Definition inconsistencies", ""]
    for row in rows:
        lines.append(f'### "{row["term"]}"')
        lines.append("")
        lines.append(f"- **{row['doc_a_title'] or row['assertion_a_id']}**: {row['def_a_text']}")
        lines.append(f"- **{row['doc_b_title'] or row['assertion_b_id']}**: {row['def_b_text']}")
        lines.append("")
        lines.append(f"**Verdict:** {row['judge_verdict']} — {row['judge_rationale']}")
        lines.append("")
    return "\n".join(lines)
```

Splice the call into the existing `render_report` function alongside the
contradiction/multi-party sections.

- [ ] **Step 5: Run the test to verify it passes**

```bash
uv run pytest tests/audit/test_report.py -v -k definition
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add consistency_checker/audit/report.py tests/audit/test_report.py
git commit -m "feat(report): definition-inconsistencies section in markdown report"
```

---

### Task 18: CLI — `--no-definitions` flag + factory

**Files:**
- Modify: `consistency_checker/cli/main.py`
- Modify: `consistency_checker/pipeline.py` (add `make_definition_checker` factory)
- Test: `tests/cli/test_main.py` (extend)

- [ ] **Step 1: Add the factory in pipeline.py**

```python
def make_definition_judge(config: Config) -> DefinitionJudge:
    from consistency_checker.check.definition_judge import LLMDefinitionJudge
    from consistency_checker.check.providers.anthropic import AnthropicDefinitionProvider
    from consistency_checker.check.providers.openai import OpenAIDefinitionProvider

    if config.judge_provider == "anthropic":
        return LLMDefinitionJudge(AnthropicDefinitionProvider(model=config.judge_model))
    if config.judge_provider == "openai":
        return LLMDefinitionJudge(OpenAIDefinitionProvider(model=config.judge_model))
    raise ValueError(
        f"make_definition_judge(): provider {config.judge_provider!r} has no factory; "
        "construct a FixtureDefinitionJudge directly in tests."
    )


def make_definition_checker(config: Config) -> DefinitionChecker:
    return DefinitionChecker(judge=make_definition_judge(config))
```

- [ ] **Step 2: Implement the new Anthropic/OpenAI definition providers**

This is two small classes that mirror `AnthropicProvider` / `OpenAIProvider` from
`consistency_checker/check/providers/anthropic.py` and `openai.py` — same SDK
calls, same tool-use / response_format mechanism, but with the definition
verdict vocabulary baked in. Approximate signature for each:

```python
class AnthropicDefinitionProvider:
    def __init__(self, *, model: str, client: anthropic.Anthropic | None = None) -> None: ...
    def request_payload(self, system: str, user: str) -> DefinitionJudgePayload: ...
```

Write them next to the existing providers, with `# type: ignore[call-overload]`
on the `messages.create` call (same pattern). Define a tool-use schema with
the three definition verdicts. Add tests modeled on the existing provider
tests in `tests/check/providers/`.

- [ ] **Step 3: Add the CLI flag**

In `consistency_checker/cli/main.py`, find the `check` command definition and add a typer option:

```python
no_definitions: bool = typer.Option(
    False,
    "--no-definitions",
    help="Skip the definition-inconsistency stage (runs by default).",
),
```

When constructing the pipeline call, conditionally instantiate the checker:

```python
definition_checker = None if no_definitions else make_definition_checker(config)

check_result = check(
    config,
    ...,
    definition_checker=definition_checker,
)
```

- [ ] **Step 4: Write the CLI test**

Add to `tests/cli/test_main.py`:

```python
def test_check_command_runs_definition_stage_by_default(...): ...
def test_check_command_respects_no_definitions_flag(...): ...
```

Modeled on the existing `--deep` flag tests (`tests/cli/test_main.py` — find them by `grep -n "deep" tests/cli/test_main.py`).

- [ ] **Step 5: Run the CLI tests**

```bash
uv run pytest tests/cli/test_main.py -v -k definitions
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add consistency_checker/cli/main.py consistency_checker/pipeline.py \
        consistency_checker/check/providers/anthropic.py \
        consistency_checker/check/providers/openai.py \
        tests/cli/test_main.py tests/check/providers/
git commit -m "feat(cli): --no-definitions flag and provider factories"
```

---

### Task 19: Web UI — Definitions tab and counters

**Files:**
- Create: `consistency_checker/web/templates/cc_definitions.html` (or whatever the project's tab naming convention is — check existing `cc_*.html` and follow it)
- Create: `consistency_checker/web/templates/cc__definition_row.html` (per-finding partial)
- Modify: `consistency_checker/web/app.py` — route + nav
- Modify: stats partials to surface `n_definition_findings`
- Test: `tests/web/test_routes.py` (extend)

This task mirrors the existing Contradictions tab very closely. Read
`consistency_checker/web/app.py` and `consistency_checker/web/templates/cc_contradictions.html`
first, then:

- [ ] **Step 1: Write the failing route test**

```python
def test_definitions_tab_renders(web_client) -> None:
    # Seed at least one definition_inconsistency finding into the test DB, then:
    resp = web_client.get("/tabs/definitions")
    assert resp.status_code == 200
    assert "Definition inconsistencies" in resp.text
```

- [ ] **Step 2: Add the route**

In `consistency_checker/web/app.py`, alongside `tab_contradictions`:

```python
@app.get("/tabs/definitions", response_class=HTMLResponse)
def tab_definitions(request: Request) -> HTMLResponse:
    # Reuse the existing run-resolution / pagination helpers in this file.
    findings = list(audit_logger.iter_findings(
        run_id=current_run_id,
        # No detector filter on iter_findings; do it inline via the store SQL.
    ))
    definition_findings = [
        f for f in findings if _detector_type(store, f) == "definition_inconsistency"
    ]
    return templates.TemplateResponse(
        "cc_definitions.html",
        {"request": request, "findings": definition_findings, "store": store},
    )
```

If `iter_findings` doesn't currently expose `detector_type`, add a thin helper
on `AuditLogger`:

```python
def iter_findings(
    self,
    *,
    run_id: str | None = None,
    verdict: str | None = None,
    detector_type: str | None = None,  # NEW
) -> Iterator[Finding]:
    clauses: list[str] = []
    params: list[Any] = []
    if run_id is not None:
        clauses.append("run_id = ?"); params.append(run_id)
    if verdict is not None:
        clauses.append("judge_verdict = ?"); params.append(verdict)
    if detector_type is not None:
        clauses.append("detector_type = ?"); params.append(detector_type)
    ...
```

(Add the corresponding tests alongside the existing `iter_findings` tests.)

- [ ] **Step 3: Add nav link**

In `cc_layout.html` (or whichever template renders the tab strip), add a new
`<a hx-get="/tabs/definitions" …>` entry mirroring the Contradictions link.

- [ ] **Step 4: Create the templates**

Mirror `cc_contradictions.html` for `cc_definitions.html`. Rows should show:
- The canonical term (title).
- Both definitions (left/right or stacked).
- The judge rationale.

- [ ] **Step 5: Add the stats counter**

Add `n_definition_findings` to `_live_counters()` (or whatever helper feeds
`cc__stats_live.html` / `cc__stats_final.html`). Query:

```sql
SELECT COUNT(*) FROM findings
WHERE run_id = ? AND detector_type = 'definition_inconsistency'
  AND judge_verdict = 'definition_divergent'
```

Render alongside the existing counter tiles.

- [ ] **Step 6: Run the web tests**

```bash
uv run pytest tests/web/ -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add consistency_checker/web/ tests/web/
git commit -m "feat(web): definitions tab + stats counter"
```

---

## Phase 6 — Validation + ADR

### Task 20: Slow A/B regression test on the combined extractor prompt

**Files:**
- Test: `tests/extract/test_atomic_facts_regression.py` (new file, marked `slow`)

This test is the gate on the "should we split into two LLM calls?" decision.
It runs the combined extractor against a small representative corpus and
asserts atomic-fact extraction quality has not regressed by more than 5%
versus a baseline manifest of expected atomic facts.

- [ ] **Step 1: Curate the fixture corpus**

Place 3-5 short text files under `tests/extract/fixtures/regression_corpus/`
representing the user's stated use cases:
- One underwriting-memo excerpt (~10 atomic facts).
- One HR-policy excerpt (~10 atomic facts).
- One organic-policy-memo excerpt (~5 atomic facts).

Alongside each `.txt`, place a `.expected.json` with the manifest of facts
the prompt should still extract.

- [ ] **Step 2: Write the slow test**

```python
import json
from pathlib import Path

import pytest

from consistency_checker.corpus.chunker import chunk_document
from consistency_checker.corpus.loader import load_corpus
from consistency_checker.extract.atomic_facts import AnthropicExtractor

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "regression_corpus"


@pytest.mark.slow
@pytest.mark.live  # also requires ANTHROPIC_API_KEY
def test_combined_extractor_recall_within_5pct_of_baseline() -> None:
    extractor = AnthropicExtractor(model="claude-sonnet-4-6")
    total_expected = 0
    total_recalled = 0
    for txt in FIXTURES.glob("*.txt"):
        text = txt.read_text(encoding="utf-8")
        expected = set(json.loads(txt.with_suffix(".expected.json").read_text()))
        # naive single-chunk run; production would chunk
        chunks = chunk_document_single(text, doc_id=txt.stem)
        recalled: set[str] = set()
        for ch in chunks:
            for a in extractor.extract(ch):
                if a.kind == "claim":
                    recalled.add(a.assertion_text.strip())
        intersect = sum(1 for e in expected if any(e in r for r in recalled))
        total_expected += len(expected)
        total_recalled += intersect
    assert total_expected > 0
    recall = total_recalled / total_expected
    assert recall >= 0.95, f"extractor recall regressed: {recall:.2%} (< 95%)"
```

- [ ] **Step 3: Run the slow test**

```bash
ANTHROPIC_API_KEY=... uv run pytest tests/extract/test_atomic_facts_regression.py -m slow -v
```

Expected: PASS at ≥95% recall. If it fails, the implementation plan branches:
either tune the combined prompt further, OR split the extractor into two
prompts (separate `record_assertions` + `record_definitions` calls). The
split-path code change is small — only `AnthropicExtractor.extract` and the
tool schema need to change; storage and pipeline stay identical.

- [ ] **Step 4: Commit**

```bash
git add tests/extract/test_atomic_facts_regression.py tests/extract/fixtures/regression_corpus/
git commit -m "test(extract): slow A/B test for combined-prompt recall regression"
```

---

### Task 21: Live end-to-end fixture test

**Files:**
- Test: `tests/test_definition_e2e.py` (new file, marked `live` + `e2e_fixture`)

- [ ] **Step 1: Write the fixture**

Place two short text files under `tests/fixtures/definition_e2e/` containing
a known divergence (e.g. an MAE definition that's broader in one and narrower
in the other).

- [ ] **Step 2: Write the test**

```python
import pytest

@pytest.mark.live
@pytest.mark.e2e_fixture
def test_mae_divergence_detected_by_real_judge(tmp_path) -> None:
    # ingest both fixture docs with AnthropicExtractor + real embedder,
    # call pipeline.check with DefinitionChecker wired to a real Anthropic
    # definition provider, assert a finding with verdict
    # 'definition_divergent' exists for term 'MAE' (canonical 'mae').
    ...
```

- [ ] **Step 3: Run the test**

```bash
ANTHROPIC_API_KEY=... uv run pytest tests/test_definition_e2e.py -m live -v
```

Expected: PASS — at least one definition-divergent finding for term `mae`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_definition_e2e.py tests/fixtures/definition_e2e/
git commit -m "test(e2e): live MAE-divergence fixture exercises the full path"
```

---

### Task 22: ADR-0009 + futureplans.md update

**Files:**
- Create: `docs/decisions/0009-definition-inconsistency-detector.md`
- Modify: `futureplans.md`

- [ ] **Step 1: Write ADR-0009**

Create `docs/decisions/0009-definition-inconsistency-detector.md`. Use ADR-0008
as the template. Required sections:

- **Status:** Accepted.
- **Context:** why the contradiction judge can't catch definition drift;
  why this detector is the natural next member of the family.
- **Decision:** the four design decisions confirmed during brainstorm
  (narrow-scope reuse of `findings`; flavor A only; LLM-only extraction
  folded into the existing extractor; term-grouping replaces NLI as the
  gate for this stage).
- **Consequences:** what `assertions.kind` and `findings.detector_type`
  unlock for future detectors; what flavor B and the gap detector will
  inherit.

- [ ] **Step 2: Update futureplans.md**

Move item #20 to the "Completed" section with one line noting the v0.4
release. Add a follow-up item under v0.4 for **flavor B (definition ↔
usage drift)** so the parked work has a tracked home.

- [ ] **Step 3: Run the full test suite + lint + type-check**

```bash
uv run pytest -m "not slow and not live" -x
uv run ruff check .
uv run ruff format --check .
uv run mypy consistency_checker
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add docs/decisions/0009-definition-inconsistency-detector.md futureplans.md
git commit -m "docs: ADR-0009 + futureplans update for definition detector"
```

- [ ] **Step 5: Open a single PR for the whole feature**

```bash
git push -u origin <feature-branch>
gh pr create --title "Definition-inconsistency detector (item #20, flavor A)" \
  --body "$(cat <<'EOF'
## Summary

Ships the second consistency detector — flagging divergent definitions of the
same term across the corpus. Reuses the existing `findings` table with a new
`detector_type` discriminator, adds `assertions.kind`/`term`/`definition_text`
columns, folds definition extraction into the existing atomic-fact extractor,
and adds a `DefinitionChecker` that bypasses the NLI gate in favour of
canonical-term grouping.

Spec: docs/superpowers/specs/2026-05-15-definition-inconsistency-detector-design.md
ADR: docs/decisions/0009-definition-inconsistency-detector.md

## Test plan

- [ ] Hermetic suite green (`uv run pytest -m "not slow and not live"`)
- [ ] Ruff + mypy clean
- [ ] Slow A/B regression test passes (`-m slow`) on the curated extractor corpus
- [ ] Live MAE divergence fixture detected (`-m live`)
- [ ] Manual: `consistency-check check --no-definitions` skips the new stage
- [ ] Manual: web UI shows new "Definitions" tab with counter on Stats

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review

**Spec coverage:**
- "Reuse `findings` + `detector_type` discriminator" → Task 2, 15.
- `assertions.kind` / `term` / `definition_text` migration → Task 1, 3, 4.
- LLM-only extraction folded into atomic-fact extractor → Task 7, 8, 9.
- Term canonicalization (case-fold, "the X", plural, quotes) → Task 5.
- `DefinitionChecker` Protocol + term-grouping pipeline → Task 14.
- NLI gate bypassed for the definition stage → Task 14 (no NLI invocation); Task 15 (NLI columns null).
- Definition judge prompts + provider → Task 11, 12, 13, 18.
- Persist with `detector_type='definition_inconsistency'` → Task 15.
- Pipeline `check()` integration after contradiction stage → Task 16.
- CLI `--no-definitions` flag → Task 18.
- Markdown report section → Task 17.
- Web UI tab + stats counters → Task 19.
- Hermetic + slow A/B + live fixture tests → Task 20, 21.
- ADR-0009 + futureplans update → Task 22.
- Decisions deferred to plan:
  - Combined-vs-split prompt decision → Task 20 (the slow A/B test gates this).
  - Exact judge prompt wording → Task 12 (drafted, will iterate during impl).
  - Same-document definition pairs → covered by the term-grouping logic in Task 14; no doc-id filter, so within-doc pairs are emitted (spec endorses this).
  - Persona filter wiring → out of scope, untouched.
  - Final CLI flag name → Task 18 uses `--no-definitions`.

**Placeholder scan:** the spec deferrals are all resolved or explicitly assigned to a task. No "TBD"/"TODO" left.

**Type consistency:**
- `DefinitionVerdictLabel` used consistently in payload, verdict dataclass, prompts, audit, and pipeline aggregation.
- `DefinitionJudgeVerdict.assertion_a_id` / `assertion_b_id` are sorted (min/max) to match the contradiction judge's pair-canonical convention and the audit `hash_id` pair convention.
- `FixtureDefinitionJudge.judge()` and `LLMDefinitionJudge.judge()` have the same signature; consumers don't switch on type.
- `record_definition_finding` writes to the same `findings` table as `record_finding`; the schema is consistent because Task 2 added `detector_type` as a column on that table.

---

## Execution Handoff

**Plan complete and saved to** `docs/superpowers/plans/2026-05-15-definition-inconsistency-detector.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks, fast iteration. Best for a 22-task plan where catching architectural drift early matters.

**2. Inline Execution** — execute tasks in the current session with checkpoint pauses for review. Lower context overhead per task but higher risk of running out of context partway through a large plan.

Which approach?
