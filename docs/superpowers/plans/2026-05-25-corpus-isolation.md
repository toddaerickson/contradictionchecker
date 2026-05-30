# Corpus Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make corpus a real isolation boundary at the CLI / pipeline / web layer via a `corpus_id` FK on `documents` + `pipeline_runs`, with `--corpus` mandatory on every mutating/judging command and a FAISS gate post-filter that prevents cross-corpus leakage.

**Architecture:** Logical isolation (single DB, single FAISS index, FK + filter clauses) per Q3-A. Pre-isolation data backfills to a `legacy` corpus. The check pipeline materializes the corpus's assertion-id set up front and drops any FAISS gate pair whose endpoints aren't both in it. Interactive corpus picker on TTY; hard error in scripts.

**Tech Stack:** Python 3.11+, SQLite (raw sqlite3), Pydantic v2 (Config), typer (CLI + prompts), Jinja2 (web templates), pytest.

**Spec:** [`docs/superpowers/specs/2026-05-25-corpus-isolation-design.md`](../specs/2026-05-25-corpus-isolation-design.md)

**Operating conventions (from `CLAUDE.md`):**
- One task ≈ one PR. Squash-merge. Rebase the working branch on fresh `origin/main` before each push.
- Hermetic tests by default. Live LLM tests get `@pytest.mark.live`.
- `Config` is frozen Pydantic — derive tweaks via `cfg.model_copy(update={...})`, never mutate.
- Use `uv run pytest -m "not slow and not live"` plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy consistency_checker` as the local CI gate. **All four must stay green.**
- Migrations are filename-ordered; never edit existing ones.
- Commit messages explain *why*, not *what*. End each with the existing co-author trailer.

**Commit-message style:** `feat(corpora): …`, `feat(isolation): …`, `chore(migrations): …`.

---

## Spec corrections discovered during the schema audit

The spec was written before re-reading the actual SQLite schema. Two corrections apply to every task below; treat these as authoritative over the spec text where they conflict:

1. **`corpora.corpus_id` is `TEXT` (UUID strings), not `INTEGER`.** Every `corpus_id` column and Python type annotation must be `TEXT` / `str`.
2. **`runs` (web table, migration 0011/0012) already has `corpus_id TEXT NOT NULL REFERENCES corpora(corpus_id) ON DELETE CASCADE`.** Migration 0014 only needs to add `corpus_id` to `documents` and `pipeline_runs`.

Other relevant cascade facts the plan relies on:
- `assertions.doc_id → documents(doc_id) ON DELETE CASCADE` exists.
- `findings.run_id → pipeline_runs(run_id) ON DELETE CASCADE` exists.
- `reviewer_verdicts` has **no FK** to assertions/documents (pair_key is a derived string). Corpus delete leaves orphan verdicts; v1 accepts orphans (no observed query impact). Cleanup as a future-spec item.

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `consistency_checker/index/migrations/0014_corpus_isolation.sql` | CREATE | Add `documents.corpus_id` and `pipeline_runs.corpus_id` (TEXT, FK with CASCADE); auto-create `legacy` corpus; backfill NULL rows. |
| `consistency_checker/extract/schema.py` | MODIFY | New `Corpus` frozen dataclass. |
| `consistency_checker/index/assertion_store.py` | MODIFY | Corpus CRUD (`get_or_create_corpus`, `list_corpora`, `delete_corpus`); `corpus_id` filter on every read (`iter_documents`, `iter_definitions`, `iter_assertions`, `stats`); new `iter_assertion_ids(corpus_id)`; NOT-NULL enforcement on `add_document`. |
| `consistency_checker/pipeline.py` | MODIFY | Thread `corpus_id` through `ingest`/`check`/`estimate_cost`; FAISS gate post-filter; `CheckResult.n_cross_corpus_gate_drops`. |
| `consistency_checker/cli/corpus_prompt.py` | CREATE | TTY-aware corpus picker — interactive prompt on TTY, hard error in scripts. |
| `consistency_checker/cli/main.py` | MODIFY | `--corpus` required on `ingest`/`check`/`estimate-cost`/`export`/`store reidentify-orgs`; new `corpus list/delete/reassign` subcommand group. |
| `consistency_checker/web/app.py` | MODIFY | `_ingest_uploaded_paths` accepts + persists `corpus_id`; every read route filters by selected corpus; `_corpus_banner_context` scoped. |
| `docs/decisions/0013-corpus-isolation.md` | CREATE | ADR. |
| `futureplans.md` | MODIFY | Move "retention gap" to Completed; note deferred archive spec. |
| Tests (new): `tests/test_corpus_crud.py`, `tests/test_cli_corpus_prompt.py`, `tests/test_cli_corpus_subcommands.py`, `tests/test_faiss_gate_corpus_filter.py`, `tests/test_pipeline_corpus_isolation.py` | CREATE | One module per new surface. |
| Tests (modified): `tests/test_migrations.py`, `tests/test_assertion_store.py`, `tests/test_pipeline.py`, `tests/test_cli.py`, `tests/test_estimate_cost.py`, `tests/test_web_corpus_banner.py` | MODIFY | Extend existing suites with corpus-scoped cases. |

**Ordering rationale:** Migration first (additive + backfill; no behavior change). Then store CRUD + writes. Then store reads (corpus filter). Then pipeline. Then CLI helpers + commands. Then web. Then bookkeeping. Each is one PR.

---

## Task 1: Migration 0014 + `Corpus` dataclass + AssertionStore corpus CRUD

**Files:**
- Create: `consistency_checker/index/migrations/0014_corpus_isolation.sql`
- Modify: `consistency_checker/extract/schema.py`
- Modify: `consistency_checker/index/assertion_store.py`
- Test: `tests/test_migrations.py` (extend), `tests/test_corpus_crud.py` (new)

- [ ] **Step 1: Write failing migration test**

In `tests/test_migrations.py`, append:

```python
def test_migration_0014_adds_corpus_id_and_backfills_legacy(tmp_path):
    from consistency_checker.index.assertion_store import AssertionStore
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    docs_cols = {r[1] for r in store._conn.execute("PRAGMA table_info(documents)")}
    pr_cols = {r[1] for r in store._conn.execute("PRAGMA table_info(pipeline_runs)")}
    assert "corpus_id" in docs_cols
    assert "corpus_id" in pr_cols
    # legacy auto-creation: only when there are NULLs to fix; fresh DB has none
    assert store._conn.execute(
        "SELECT COUNT(*) FROM corpora WHERE corpus_name='legacy'"
    ).fetchone()[0] == 0
    store.close()


def test_migration_0014_creates_legacy_when_orphan_docs_exist(tmp_path):
    from consistency_checker.index.assertion_store import AssertionStore
    # Build a DB at the pre-0014 schema by running migrations 0001..0013 only.
    import sqlite3
    db = tmp_path / "pre.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE corpora (corpus_id TEXT PRIMARY KEY, corpus_name TEXT UNIQUE,
            corpus_path TEXT, judge_provider TEXT, created_at TEXT, updated_at TEXT);
        CREATE TABLE documents (doc_id TEXT PRIMARY KEY, source_path TEXT);
        CREATE TABLE pipeline_runs (run_id TEXT PRIMARY KEY);
        CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT);
        INSERT INTO documents (doc_id, source_path) VALUES ('d1', '/x.txt');
        INSERT INTO pipeline_runs (run_id) VALUES ('r1');
    """)
    for v in range(1, 14):
        conn.execute("INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, datetime('now'))", (v,))
    conn.commit(); conn.close()

    store = AssertionStore(db)
    store.migrate()
    legacy_id = store._conn.execute(
        "SELECT corpus_id FROM corpora WHERE corpus_name='legacy'"
    ).fetchone()[0]
    assert legacy_id is not None
    assert store._conn.execute(
        "SELECT corpus_id FROM documents WHERE doc_id='d1'"
    ).fetchone()[0] == legacy_id
    assert store._conn.execute(
        "SELECT corpus_id FROM pipeline_runs WHERE run_id='r1'"
    ).fetchone()[0] == legacy_id
    store.close()
```

- [ ] **Step 2: Run — confirm FAIL**

```sh
uv run pytest tests/test_migrations.py -v -k 0014
```

Expected: FAIL (column not present, legacy not created).

- [ ] **Step 3: Add migration file**

Create `consistency_checker/index/migrations/0014_corpus_isolation.sql`:

```sql
-- Logical corpus isolation: every document and run belongs to one corpus.
-- corpora.corpus_id is TEXT (UUID), so the new FK columns mirror that type.
-- runs (web table) already has corpus_id from migration 0011/0012; no change needed there.

ALTER TABLE documents     ADD COLUMN corpus_id TEXT REFERENCES corpora(corpus_id) ON DELETE CASCADE;
ALTER TABLE pipeline_runs ADD COLUMN corpus_id TEXT REFERENCES corpora(corpus_id) ON DELETE CASCADE;

-- Auto-create "legacy" corpus when pre-isolation rows exist.
INSERT INTO corpora (corpus_id, corpus_name, corpus_path, judge_provider, created_at, updated_at)
SELECT lower(hex(randomblob(8))), 'legacy', '(pre-isolation)', 'moonshot',
       datetime('now'), datetime('now')
WHERE EXISTS (
    SELECT 1 FROM documents     WHERE corpus_id IS NULL
    UNION ALL
    SELECT 1 FROM pipeline_runs WHERE corpus_id IS NULL
)
AND NOT EXISTS (SELECT 1 FROM corpora WHERE corpus_name='legacy');

UPDATE documents
SET corpus_id = (SELECT corpus_id FROM corpora WHERE corpus_name='legacy')
WHERE corpus_id IS NULL;

UPDATE pipeline_runs
SET corpus_id = (SELECT corpus_id FROM corpora WHERE corpus_name='legacy')
WHERE corpus_id IS NULL;
```

- [ ] **Step 4: Run migration tests — confirm PASS**

```sh
uv run pytest tests/test_migrations.py -v -k 0014
```

- [ ] **Step 5: Write failing Corpus-dataclass + CRUD tests**

Create `tests/test_corpus_crud.py`:

```python
def test_corpus_dataclass_carries_fields():
    from consistency_checker.extract.schema import Corpus
    c = Corpus(corpus_id="abc", corpus_name="atkins", corpus_path="/data/atkins",
               judge_provider="moonshot")
    assert c.corpus_name == "atkins"


def test_get_or_create_corpus_creates_then_returns_same_id(tmp_path):
    from consistency_checker.index.assertion_store import AssertionStore
    store = AssertionStore(tmp_path / "t.db"); store.migrate()
    cid1 = store.get_or_create_corpus("atkins", "/data/atkins", "moonshot")
    cid2 = store.get_or_create_corpus("atkins", "/different/path", "anthropic")
    assert cid1 == cid2  # name is the key; subsequent calls return existing id
    store.close()


def test_list_corpora_returns_all_in_creation_order(tmp_path):
    from consistency_checker.index.assertion_store import AssertionStore
    store = AssertionStore(tmp_path / "t.db"); store.migrate()
    store.get_or_create_corpus("alpha", "/a", "moonshot")
    store.get_or_create_corpus("beta",  "/b", "moonshot")
    names = [c.corpus_name for c in store.list_corpora()]
    assert names == ["alpha", "beta"]
    store.close()


def test_delete_corpus_cascades_to_documents_and_assertions(tmp_path):
    from consistency_checker.index.assertion_store import AssertionStore
    from consistency_checker.extract.schema import Document, Assertion
    store = AssertionStore(tmp_path / "t.db"); store.migrate()
    cid = store.get_or_create_corpus("atkins", "/data/atkins", "moonshot")
    store.add_document(Document(doc_id="d1", source_path="/a"), corpus_id=cid)
    store.add_assertion(Assertion.build("d1", "a claim", kind="claim"))
    assert store._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    assert store._conn.execute("SELECT COUNT(*) FROM assertions").fetchone()[0] == 1
    store.delete_corpus(cid)
    assert store._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    assert store._conn.execute("SELECT COUNT(*) FROM assertions").fetchone()[0] == 0
    store.close()
```

- [ ] **Step 6: Run — confirm FAIL (ImportError / AttributeError)**

- [ ] **Step 7: Implement `Corpus` dataclass**

In `consistency_checker/extract/schema.py`, append after `Document`:

```python
@dataclass(frozen=True, slots=True)
class Corpus:
    corpus_id: str
    corpus_name: str
    corpus_path: str
    judge_provider: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
```

- [ ] **Step 8: Implement CRUD on `AssertionStore`**

In `consistency_checker/index/assertion_store.py`, add:

```python
import uuid

# ...

def get_or_create_corpus(
    self, name: str, path: str, judge_provider: str
) -> str:
    """Return the corpus_id for ``name``, creating the row if needed.

    Why: ingest must resolve a stable id from the user-facing name; if the name
    already exists we return its id unchanged (subsequent ingests append to it).
    Path / judge_provider mismatches are logged elsewhere as warnings, not errors.
    """
    row = self._conn.execute(
        "SELECT corpus_id FROM corpora WHERE corpus_name = ?", (name,)
    ).fetchone()
    if row:
        return row["corpus_id"]
    new_id = uuid.uuid4().hex
    with self._conn:
        self._conn.execute(
            "INSERT INTO corpora (corpus_id, corpus_name, corpus_path, judge_provider) "
            "VALUES (?, ?, ?, ?)",
            (new_id, name, path, judge_provider),
        )
    return new_id


def list_corpora(self) -> list[Corpus]:
    rows = self._conn.execute(
        "SELECT * FROM corpora ORDER BY created_at, corpus_name"
    ).fetchall()
    return [
        Corpus(
            corpus_id=r["corpus_id"], corpus_name=r["corpus_name"],
            corpus_path=r["corpus_path"], judge_provider=r["judge_provider"],
            created_at=r["created_at"], updated_at=r["updated_at"],
        )
        for r in rows
    ]


def delete_corpus(self, corpus_id: str) -> None:
    """Delete a corpus and CASCADE to documents/pipeline_runs/runs (and their
    descendants). reviewer_verdicts may be left as orphans — v1 accepts this."""
    # Enable FK enforcement for this connection if not already on; SQLite
    # default is OFF and ON DELETE CASCADE only fires when on.
    self._conn.execute("PRAGMA foreign_keys = ON")
    with self._conn:
        self._conn.execute("DELETE FROM corpora WHERE corpus_id = ?", (corpus_id,))
```

Add `from consistency_checker.extract.schema import Corpus` to the imports.

- [ ] **Step 9: Run — confirm PASS**

```sh
uv run pytest tests/test_corpus_crud.py tests/test_migrations.py -v
```

- [ ] **Step 10: Full gate + commit**

```sh
uv run ruff check . && uv run ruff format --check . && uv run mypy consistency_checker && uv run pytest -m "not slow and not live"
git add consistency_checker/index/migrations/0014_corpus_isolation.sql \
        consistency_checker/extract/schema.py \
        consistency_checker/index/assertion_store.py \
        tests/test_migrations.py tests/test_corpus_crud.py
git commit -m "$(cat <<'EOF'
chore(migrations): 0014 corpus isolation — documents.corpus_id + pipeline_runs.corpus_id + legacy backfill

Adds the FK columns and auto-creates a "legacy" corpus when pre-isolation
rows exist. Corpus is now a real isolation boundary; subsequent tasks
make --corpus mandatory at the CLI and scope reads accordingly.

corpus_id is TEXT (UUID) to match the existing corpora.corpus_id type.
runs (web table) already had corpus_id from migration 0011/0012; only
documents and pipeline_runs needed the column added.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `AssertionStore.add_document` requires `corpus_id`

**Files:**
- Modify: `consistency_checker/index/assertion_store.py`
- Test: extend `tests/test_assertion_store.py`

- [ ] **Step 1: Write failing test**

```python
def test_add_document_requires_corpus_id(tmp_path):
    from consistency_checker.index.assertion_store import AssertionStore
    from consistency_checker.extract.schema import Document
    store = AssertionStore(tmp_path / "t.db"); store.migrate()
    with pytest.raises(ValueError, match="corpus_id is required"):
        store.add_document(Document(doc_id="d1", source_path="/x.txt"), corpus_id=None)
    store.close()


def test_add_document_persists_corpus_id(tmp_path):
    from consistency_checker.index.assertion_store import AssertionStore
    from consistency_checker.extract.schema import Document
    store = AssertionStore(tmp_path / "t.db"); store.migrate()
    cid = store.get_or_create_corpus("atkins", "/data/atkins", "moonshot")
    store.add_document(Document(doc_id="d1", source_path="/x.txt"), corpus_id=cid)
    row = store._conn.execute(
        "SELECT corpus_id FROM documents WHERE doc_id='d1'"
    ).fetchone()
    assert row[0] == cid
    store.close()
```

- [ ] **Step 2: Run — confirm FAIL**

- [ ] **Step 3: Modify `add_document` signature + INSERT**

Replace the current `add_document` with:

```python
def add_document(self, doc: Document, *, corpus_id: str | None) -> None:
    if corpus_id is None:
        raise ValueError(
            "corpus_id is required; pass --corpus <name> at the CLI"
        )
    with self._conn:
        self._conn.execute(
            "INSERT OR IGNORE INTO documents"
            "(doc_id, source_path, title, doc_date, doc_type, metadata_json, "
            "org_label, org_reason, corpus_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc.doc_id, doc.source_path, doc.title, doc.doc_date,
                doc.doc_type, doc.metadata_json, doc.org_label, doc.org_reason,
                corpus_id,
            ),
        )
```

Make `corpus_id` keyword-only via the `*`. Note: this is a **breaking API change** — every existing caller must update. Find them with `git grep -n "add_document(" consistency_checker tests`.

- [ ] **Step 4: Update existing callers**

For each existing call site, look at the test fixture pattern and either:
- Create a corpus first via `get_or_create_corpus` and pass its id.
- For pipeline.ingest call sites: those land in Task 4. For now, in any test that previously called `store.add_document(doc)`, change to `store.add_document(doc, corpus_id=cid)` where `cid` is obtained from `get_or_create_corpus`.

Run `uv run pytest -m "not slow and not live"` and fix every TypeError/ValueError that surfaces.

- [ ] **Step 5: Run — confirm PASS**

```sh
uv run pytest tests/test_assertion_store.py -v -k corpus_id
uv run pytest -m "not slow and not live"
```

- [ ] **Step 6: Full gate + commit**

```sh
git add consistency_checker/index/assertion_store.py tests/test_assertion_store.py [+ other tests touched]
git commit -m "$(cat <<'EOF'
feat(isolation): require corpus_id on AssertionStore.add_document

Application-layer NOT NULL enforcement (SQLite ALTER cannot add NOT NULL
without a rebuild). Pre-existing test fixtures updated to obtain a
corpus_id via get_or_create_corpus before adding documents.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Read filters — `corpus_id` kwarg on `iter_documents`, `iter_definitions`, `iter_assertions`, `iter_assertion_ids`, `stats`

**Files:**
- Modify: `consistency_checker/index/assertion_store.py`
- Test: extend `tests/test_assertion_store.py`

- [ ] **Step 1: Write failing tests**

```python
def test_iter_documents_filters_by_corpus_id(tmp_path):
    from consistency_checker.index.assertion_store import AssertionStore
    from consistency_checker.extract.schema import Document
    store = AssertionStore(tmp_path / "t.db"); store.migrate()
    a = store.get_or_create_corpus("atkins", "/atkins", "moonshot")
    b = store.get_or_create_corpus("beta",   "/beta",   "moonshot")
    store.add_document(Document(doc_id="d1", source_path="/x"), corpus_id=a)
    store.add_document(Document(doc_id="d2", source_path="/y"), corpus_id=b)
    assert {d.doc_id for d in store.iter_documents()} == {"d1", "d2"}
    assert {d.doc_id for d in store.iter_documents(corpus_id=a)} == {"d1"}
    assert {d.doc_id for d in store.iter_documents(corpus_id=b)} == {"d2"}
    store.close()


def test_iter_definitions_filters_by_corpus_id(tmp_path):
    from consistency_checker.index.assertion_store import AssertionStore
    from consistency_checker.extract.schema import Document, Assertion
    store = AssertionStore(tmp_path / "t.db"); store.migrate()
    a = store.get_or_create_corpus("atkins", "/atkins", "moonshot")
    b = store.get_or_create_corpus("beta",   "/beta",   "moonshot")
    store.add_document(Document(doc_id="d1", source_path="/x"), corpus_id=a)
    store.add_document(Document(doc_id="d2", source_path="/y"), corpus_id=b)
    for d in ("d1", "d2"):
        store.add_assertion(Assertion.build(
            d, '"Term" means foo.', kind="definition", term="Term", definition_text="foo"
        ))
    rows_all = list(store.iter_definitions())
    rows_a   = list(store.iter_definitions(corpus_id=a))
    assert len(rows_all) == 2
    assert len(rows_a)   == 1
    assert rows_a[0][0].doc_id == "d1"
    store.close()


def test_iter_assertion_ids_filters_by_corpus_id(tmp_path):
    from consistency_checker.index.assertion_store import AssertionStore
    from consistency_checker.extract.schema import Document, Assertion
    store = AssertionStore(tmp_path / "t.db"); store.migrate()
    a = store.get_or_create_corpus("a", "/a", "moonshot")
    b = store.get_or_create_corpus("b", "/b", "moonshot")
    store.add_document(Document(doc_id="d1", source_path="/x"), corpus_id=a)
    store.add_document(Document(doc_id="d2", source_path="/y"), corpus_id=b)
    store.add_assertion(Assertion.build("d1", "claim a", kind="claim"))
    store.add_assertion(Assertion.build("d2", "claim b", kind="claim"))
    assert set(store.iter_assertion_ids(corpus_id=a)) == {Assertion.build("d1", "claim a", kind="claim").assertion_id}
    store.close()
```

- [ ] **Step 2: Run — confirm FAIL**

- [ ] **Step 3: Add `corpus_id` filter to every read method**

In `consistency_checker/index/assertion_store.py`:

```python
def iter_documents(
    self, *, limit: int | None = None, offset: int = 0,
    corpus_id: str | None = None,
) -> Iterator[Document]:
    sql = "SELECT * FROM documents"
    params: list = []
    if corpus_id is not None:
        sql += " WHERE corpus_id = ?"
        params.append(corpus_id)
    sql += " ORDER BY ingested_at DESC, doc_id DESC"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    for row in self._conn.execute(sql, params):
        yield _row_to_document(row)


def iter_definitions(
    self, *, corpus_id: str | None = None,
) -> Iterator[tuple[Assertion, str]]:
    """Yield (definition_assertion, org_key) ordered by created_at.
    org_key is normalize_org(org_label) or '' (the unknown bucket)."""
    sql = (
        "SELECT a.*, d.org_label "
        "FROM assertions a "
        "LEFT JOIN documents d ON d.doc_id = a.doc_id "
        "WHERE a.kind = 'definition'"
    )
    params: list = []
    if corpus_id is not None:
        sql += " AND d.corpus_id = ?"
        params.append(corpus_id)
    sql += " ORDER BY a.created_at, a.assertion_id"
    for row in self._conn.execute(sql, params):
        org_key = normalize_org(row["org_label"]) if row["org_label"] else ""
        yield _row_to_assertion(row), org_key


def iter_assertions(
    self, doc_id: str | None = None, *,
    kind: str | None = None,
    corpus_id: str | None = None,
) -> Iterator[Assertion]:
    sql = "SELECT a.* FROM assertions a"
    where: list[str] = []
    params: list = []
    if corpus_id is not None:
        sql += " JOIN documents d ON d.doc_id = a.doc_id"
        where.append("d.corpus_id = ?")
        params.append(corpus_id)
    if doc_id:
        where.append("a.doc_id = ?")
        params.append(doc_id)
    if kind:
        where.append("a.kind = ?")
        params.append(kind)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY a.created_at, a.assertion_id"
    for row in self._conn.execute(sql, params):
        yield _row_to_assertion(row)


def iter_assertion_ids(
    self, *, corpus_id: str | None = None,
) -> Iterator[str]:
    """Stream assertion ids; used by the FAISS gate post-filter (Task 5)."""
    sql = "SELECT a.assertion_id FROM assertions a"
    params: list = []
    if corpus_id is not None:
        sql += " JOIN documents d ON d.doc_id = a.doc_id WHERE d.corpus_id = ?"
        params.append(corpus_id)
    for row in self._conn.execute(sql, params):
        yield row["assertion_id"]


def stats(self, *, corpus_id: str | None = None) -> dict[str, int]:
    where = ""
    params: list = []
    if corpus_id is not None:
        where = " WHERE corpus_id = ?"
        params = [corpus_id]
    n_docs = self._conn.execute(f"SELECT COUNT(*) FROM documents{where}", params).fetchone()[0]
    n_asserts = self._conn.execute(
        f"SELECT COUNT(*) FROM assertions"
        + (" WHERE doc_id IN (SELECT doc_id FROM documents WHERE corpus_id = ?)" if corpus_id else ""),
        params,
    ).fetchone()[0]
    return {"documents": n_docs, "assertions": n_asserts}
```

Match the existing patterns in the file (some signatures may be slightly different — preserve them). Key invariant: when `corpus_id=None`, every method returns the same rows it did before (backward compat).

- [ ] **Step 4: Run — confirm PASS**

```sh
uv run pytest tests/test_assertion_store.py -v
uv run pytest -m "not slow and not live"   # confirm existing tests still pass
```

- [ ] **Step 5: Full gate + commit**

```sh
git add consistency_checker/index/assertion_store.py tests/test_assertion_store.py
git commit -m "$(cat <<'EOF'
feat(isolation): AssertionStore reads gain corpus_id filter

iter_documents, iter_definitions, iter_assertions, stats all accept an
optional corpus_id kwarg. Default None preserves pre-feature behavior.
New iter_assertion_ids supports the FAISS gate post-filter (Task 5).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `pipeline.ingest` accepts + persists `corpus_id`

**Files:**
- Modify: `consistency_checker/pipeline.py`
- Test: extend `tests/test_pipeline_definition_stage.py`

- [ ] **Step 1: Write failing test**

```python
def test_ingest_persists_corpus_id(tmp_path):
    from consistency_checker.config import Config
    from consistency_checker.extract.atomic_facts import FixtureExtractor
    from consistency_checker.index.assertion_store import AssertionStore
    from consistency_checker.index.faiss_store import FaissStore
    from consistency_checker.pipeline import ingest
    from tests.conftest import HashEmbedder

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("Bylaws of Acme", encoding="utf-8")
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(f"corpus_dir: {tmp_path}\ndata_dir: {tmp_path}\n", encoding="utf-8")

    store = AssertionStore(tmp_path / "store.db"); store.migrate()
    cid = store.get_or_create_corpus("atkins", str(tmp_path), "moonshot")
    faiss = FaissStore(tmp_path / "faiss")
    ingest(
        Config.from_yaml(cfg_path),
        store=store, faiss_store=faiss,
        extractor=FixtureExtractor({}),
        embedder=HashEmbedder(dim=32),
        corpus_id=cid,
    )
    rows = list(store.iter_documents())
    assert all(d.corpus_id == cid for d in rows) or all(
        store._conn.execute("SELECT corpus_id FROM documents WHERE doc_id=?", (d.doc_id,)).fetchone()[0] == cid
        for d in rows
    )
    store.close()
```

(The `or` branch covers whether `Document` carries `corpus_id` as a field or only as a row attribute — pick whichever lands in Task 1's `Document` extension. If `Document` doesn't carry it, the second branch is the canonical check.)

- [ ] **Step 2: Run — confirm FAIL**

- [ ] **Step 3: Thread `corpus_id` through `pipeline.ingest`**

In `consistency_checker/pipeline.py`, modify `ingest`'s signature:

```python
def ingest(
    config: Config, *,
    store: AssertionStore, faiss_store: FaissStore,
    extractor: Extractor, embedder: Embedder,
    corpus_id: str,
) -> IngestResult:
    ...
```

(Add `corpus_id` as a required keyword arg. No default — fail fast if caller forgets.)

In the loop where `store.add_document(doc)` is called today, change to:

```python
store.add_document(doc, corpus_id=corpus_id)
```

- [ ] **Step 4: Run — confirm PASS**

- [ ] **Step 5: Full gate + commit**

```sh
git add consistency_checker/pipeline.py tests/test_pipeline_definition_stage.py
git commit -m "$(cat <<'EOF'
feat(isolation): pipeline.ingest requires + persists corpus_id

Threads the corpus through to add_document. Task 7 (CLI ingest) supplies
it; tests use get_or_create_corpus + a fixture extractor.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `pipeline.check` scopes by `corpus_id` + FAISS gate post-filter; `estimate_cost` same

**Files:**
- Modify: `consistency_checker/pipeline.py`
- Test: `tests/test_faiss_gate_corpus_filter.py` (new), `tests/test_pipeline_corpus_isolation.py` (new), extend `tests/test_estimate_cost.py`

- [ ] **Step 1: Write failing tests**

`tests/test_pipeline_corpus_isolation.py` (new):

```python
def test_check_does_not_judge_cross_corpus_pairs(tmp_path):
    """Build two corpora with similar text; check on corpus A.
    Assert no finding references any doc from corpus B."""
    from consistency_checker.config import Config
    from consistency_checker.extract.atomic_facts import FixtureExtractor
    from consistency_checker.extract.schema import Document, Assertion
    from consistency_checker.index.assertion_store import AssertionStore
    from consistency_checker.index.faiss_store import FaissStore
    from consistency_checker.pipeline import check
    from tests.conftest import HashEmbedder
    # Set up two corpora, each with one assertion that would be a FAISS hit.
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(f"corpus_dir: {tmp_path}\ndata_dir: {tmp_path}\n", encoding="utf-8")
    store = AssertionStore(tmp_path / "store.db"); store.migrate()
    a = store.get_or_create_corpus("a", "/a", "moonshot")
    b = store.get_or_create_corpus("b", "/b", "moonshot")
    store.add_document(Document(doc_id="dA", source_path="/a"), corpus_id=a)
    store.add_document(Document(doc_id="dB", source_path="/b"), corpus_id=b)
    aa = Assertion.build("dA", "Revenue grew 12% in 2023.")
    ab = Assertion.build("dB", "Revenue grew 12% in 2023.")  # identical, would be a FAISS hit
    store.add_assertion(aa); store.add_assertion(ab)
    # Embed both into the same FAISS index.
    emb = HashEmbedder(dim=32)
    vecs = emb.embed_texts([aa.assertion_text, ab.assertion_text])
    faiss = FaissStore(tmp_path / "faiss", dim=32)
    faiss.add(aa.assertion_id, vecs[0]); faiss.add(ab.assertion_id, vecs[1])

    result = check(
        Config.from_yaml(cfg_path),
        store=store, faiss_store=faiss,
        corpus_id=a,
    )
    # Every finding must reference only corpus-a docs.
    for f in result.findings:
        for aid in (f.assertion_a_id, f.assertion_b_id):
            doc_id = store.get_assertion(aid).doc_id
            assert store._conn.execute(
                "SELECT corpus_id FROM documents WHERE doc_id=?", (doc_id,)
            ).fetchone()[0] == a, f"finding leaks corpus-b doc {doc_id}"
    assert result.n_cross_corpus_gate_drops >= 1  # the cross-corpus pair WAS dropped
    store.close()
```

(Adapt fixture shapes — `check`'s real signature may need `extractor`/`embedder`/`judge` kwargs. Look at existing tests for the pattern.)

`tests/test_faiss_gate_corpus_filter.py` (new):

```python
def test_faiss_pairs_outside_corpus_are_dropped_pre_judge(tmp_path):
    """Unit-level: with corpus_assertion_ids = {a1, a2}, a candidate pair
    (a1, b1) where b1 is not in the set must be dropped before reaching the judge."""
    from consistency_checker.pipeline import _filter_pairs_by_corpus  # NEW helper

    corpus_ids = {"a1", "a2"}
    pairs = [("a1", "a2"), ("a1", "b1"), ("b1", "b2"), ("a2", "a1")]
    kept, dropped = _filter_pairs_by_corpus(pairs, corpus_ids)
    assert kept == [("a1", "a2"), ("a2", "a1")]
    assert dropped == 2
```

- [ ] **Step 2: Run — confirm FAIL**

- [ ] **Step 3: Implement helper + thread `corpus_id` through `check`**

In `consistency_checker/pipeline.py`:

```python
def _filter_pairs_by_corpus(
    pairs: Iterable[tuple[str, str]], corpus_assertion_ids: set[str]
) -> tuple[list[tuple[str, str]], int]:
    """Drop pairs where either endpoint is outside the corpus. Returns (kept, n_dropped)."""
    kept: list[tuple[str, str]] = []
    dropped = 0
    for a, b in pairs:
        if a in corpus_assertion_ids and b in corpus_assertion_ids:
            kept.append((a, b))
        else:
            dropped += 1
    return kept, dropped
```

Modify `check`'s signature to take `corpus_id: str`. At the top of `check`:

```python
corpus_assertion_ids: set[str] = set(store.iter_assertion_ids(corpus_id=corpus_id))
```

Locate the FAISS gate's candidate-pair output and filter:

```python
candidate_pairs, n_cross_corpus_drops = _filter_pairs_by_corpus(
    candidate_pairs, corpus_assertion_ids
)
```

Pass `corpus_id=corpus_id` to `store.iter_definitions(...)` in the definition pass.

In `CheckResult`, add `n_cross_corpus_gate_drops: int = 0` and populate it.

`pipeline_runs` row write must include `corpus_id`. Find the existing INSERT and add the column.

- [ ] **Step 4: Same change to `estimate_cost`**

Apply the same `_filter_pairs_by_corpus` step and the `iter_definitions(corpus_id=...)` filter so the preview matches the actual run.

- [ ] **Step 5: Run — confirm PASS**

```sh
uv run pytest tests/test_pipeline_corpus_isolation.py tests/test_faiss_gate_corpus_filter.py tests/test_estimate_cost.py -v
uv run pytest -m "not slow and not live"
```

- [ ] **Step 6: Full gate + commit**

```sh
git add consistency_checker/pipeline.py tests/test_pipeline_corpus_isolation.py tests/test_faiss_gate_corpus_filter.py tests/test_estimate_cost.py
git commit -m "$(cat <<'EOF'
feat(isolation): pipeline.check + estimate_cost are corpus-scoped + FAISS gate post-filter

FAISS is a shared index across all corpora; a corpus-scoped check
materializes the corpus's assertion-id set up front and drops any
candidate pair whose endpoints aren't both inside it. Same filter in
estimate_cost so the preview matches the run. New
CheckResult.n_cross_corpus_gate_drops counter for telemetry.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: CLI corpus prompt helper (TTY-aware)

**Files:**
- Create: `consistency_checker/cli/corpus_prompt.py`
- Test: `tests/test_cli_corpus_prompt.py` (new)

- [ ] **Step 1: Write failing tests**

```python
def test_resolve_corpus_returns_passed_name_if_given(tmp_path):
    from consistency_checker.cli.corpus_prompt import resolve_corpus
    from consistency_checker.index.assertion_store import AssertionStore
    store = AssertionStore(tmp_path / "t.db"); store.migrate()
    store.get_or_create_corpus("atkins", "/atkins", "moonshot")
    cid = resolve_corpus(store, "atkins", "/atkins", "moonshot", isatty=False)
    assert cid is not None


def test_resolve_corpus_creates_new_corpus_if_name_unknown(tmp_path):
    from consistency_checker.cli.corpus_prompt import resolve_corpus
    from consistency_checker.index.assertion_store import AssertionStore
    store = AssertionStore(tmp_path / "t.db"); store.migrate()
    cid = resolve_corpus(store, "newcorpus", "/newpath", "moonshot", isatty=False)
    assert cid is not None
    assert any(c.corpus_name == "newcorpus" for c in store.list_corpora())


def test_resolve_corpus_raises_when_missing_and_non_tty(tmp_path):
    from consistency_checker.cli.corpus_prompt import resolve_corpus, CorpusRequiredError
    from consistency_checker.index.assertion_store import AssertionStore
    store = AssertionStore(tmp_path / "t.db"); store.migrate()
    store.get_or_create_corpus("alpha", "/a", "moonshot")
    store.get_or_create_corpus("beta",  "/b", "moonshot")
    with pytest.raises(CorpusRequiredError, match="alpha, beta"):
        resolve_corpus(store, None, None, "moonshot", isatty=False)


def test_resolve_corpus_interactive_picks_from_list(tmp_path, monkeypatch):
    from consistency_checker.cli.corpus_prompt import resolve_corpus
    from consistency_checker.index.assertion_store import AssertionStore
    store = AssertionStore(tmp_path / "t.db"); store.migrate()
    store.get_or_create_corpus("alpha", "/a", "moonshot")
    store.get_or_create_corpus("beta",  "/b", "moonshot")
    # Simulate user typing "2" (beta is second).
    monkeypatch.setattr("typer.prompt", lambda *a, **k: "2")
    cid = resolve_corpus(store, None, None, "moonshot", isatty=True)
    beta_id = next(c.corpus_id for c in store.list_corpora() if c.corpus_name == "beta")
    assert cid == beta_id
```

- [ ] **Step 2: Run — confirm FAIL**

- [ ] **Step 3: Implement helper**

Create `consistency_checker/cli/corpus_prompt.py`:

```python
"""TTY-aware corpus resolution.

Why: --corpus is required on every mutating CLI command (per spec §3). When
the operator forgets to pass it on a TTY we interactively prompt; in a
scripted/piped environment we fail fast with the available list. Lives in
its own module so the policy is unit-testable without typer machinery.
"""

from __future__ import annotations

import sys

import typer

from consistency_checker.index.assertion_store import AssertionStore


class CorpusRequiredError(typer.BadParameter):
    pass


def resolve_corpus(
    store: AssertionStore,
    name: str | None,
    path: str | None,
    judge_provider: str,
    *,
    isatty: bool | None = None,
) -> str:
    """Return a corpus_id for ``name``. Creates the corpus if name is new.

    If name is None and stdin is a TTY → interactive picker.
    If name is None and not a TTY    → CorpusRequiredError listing available corpora.
    """
    if isatty is None:
        isatty = sys.stdin.isatty()

    if name is not None:
        return store.get_or_create_corpus(name, path or "(unset)", judge_provider)

    existing = store.list_corpora()
    if not isatty:
        names = ", ".join(c.corpus_name for c in existing) or "<none>"
        raise CorpusRequiredError(f"--corpus is required (available: {names})")

    if not existing:
        new_name = typer.prompt("No corpora yet. Name a new one")
        return store.get_or_create_corpus(new_name, path or "(unset)", judge_provider)

    typer.echo("Available corpora:")
    for i, c in enumerate(existing, 1):
        typer.echo(f"  {i}. {c.corpus_name:30s}  ({c.corpus_path})")
    typer.echo("  [new]  create a new corpus")
    choice = typer.prompt("Pick one").strip()
    if choice == "new":
        new_name = typer.prompt("Name")
        return store.get_or_create_corpus(new_name, path or "(unset)", judge_provider)
    if choice.isdigit() and 1 <= int(choice) <= len(existing):
        return existing[int(choice) - 1].corpus_id
    # One re-prompt then give up.
    choice = typer.prompt("Invalid; pick a number from the list, or [new]").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(existing):
        return existing[int(choice) - 1].corpus_id
    if choice == "new":
        new_name = typer.prompt("Name")
        return store.get_or_create_corpus(new_name, path or "(unset)", judge_provider)
    raise CorpusRequiredError("No valid corpus selection.")
```

- [ ] **Step 4: Run — confirm PASS**

- [ ] **Step 5: Full gate + commit**

```sh
git add consistency_checker/cli/corpus_prompt.py tests/test_cli_corpus_prompt.py
git commit -m "$(cat <<'EOF'
feat(cli): TTY-aware corpus_prompt helper

Interactive picker on a TTY; CorpusRequiredError with the available
list when stdin is piped (scripts must always specify --corpus).
Selecting [new] creates a corpus inline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: CLI `ingest --corpus` required + wired

**Files:**
- Modify: `consistency_checker/cli/main.py`
- Test: extend `tests/test_cli.py`

- [ ] **Step 1: Write failing test**

```python
def test_ingest_without_corpus_errors_in_non_tty(monkeypatch, tmp_path):
    from typer.testing import CliRunner
    from consistency_checker.cli.main import app
    cfg = tmp_path / "config.yml"
    cfg.write_text(f"corpus_dir: {tmp_path}\ndata_dir: {tmp_path}\n", encoding="utf-8")
    runner = CliRunner()
    res = runner.invoke(app, ["ingest", str(tmp_path), "--config", str(cfg)])
    assert res.exit_code != 0
    assert "--corpus is required" in (res.output + str(res.exception or ""))


def test_ingest_with_corpus_creates_and_persists(monkeypatch, tmp_path):
    from typer.testing import CliRunner
    from consistency_checker.cli.main import app
    from consistency_checker.extract.atomic_facts import FixtureExtractor, OrgIdentification
    from consistency_checker.index.assertion_store import AssertionStore
    import consistency_checker.pipeline as pipeline_mod

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("Atkins bylaws text", encoding="utf-8")
    cfg = tmp_path / "config.yml"
    cfg.write_text(
        f"corpus_dir: {tmp_path}\ndata_dir: {tmp_path}\njudge_provider: moonshot\n",
        encoding="utf-8",
    )
    fx = FixtureExtractor(
        {}, org_fixtures={("doc", "Atkins"): OrgIdentification("Atkins", "org_found")}
    )
    monkeypatch.setattr(pipeline_mod, "make_extractor", lambda c: fx)

    runner = CliRunner()
    res = runner.invoke(app, ["ingest", str(tmp_path), "--config", str(cfg),
                              "--corpus", "atkins"])
    assert res.exit_code == 0, res.output

    store = AssertionStore(tmp_path / "store.db")
    names = [c.corpus_name for c in store.list_corpora()]
    assert "atkins" in names
    atkins_id = next(c.corpus_id for c in store.list_corpora() if c.corpus_name == "atkins")
    rows = store._conn.execute(
        "SELECT corpus_id FROM documents"
    ).fetchall()
    assert rows and all(r[0] == atkins_id for r in rows)
    store.close()
```

(Fill in the second test body using existing test fixtures in `tests/test_cli.py`.)

- [ ] **Step 2: Run — confirm FAIL**

- [ ] **Step 3: Modify `ingest` typer command**

In `consistency_checker/cli/main.py`, find `def ingest(...)` and add:

```python
corpus: str | None = typer.Option(
    None, "--corpus", help="Corpus name. Required (interactive picker on TTY).",
),
```

Inside the function, before calling `pipeline.ingest(...)`:

```python
from consistency_checker.cli.corpus_prompt import resolve_corpus
corpus_id = resolve_corpus(
    store, corpus, str(config.corpus_dir), config.judge_provider,
)
# ...
pipeline.ingest(config, store=store, faiss_store=faiss, extractor=extractor,
                embedder=embedder, corpus_id=corpus_id)
```

- [ ] **Step 4: Run — confirm PASS**

- [ ] **Step 5: Full gate + commit**

```sh
git add consistency_checker/cli/main.py tests/test_cli.py
git commit -m "$(cat <<'EOF'
feat(cli): ingest --corpus required + interactive picker

resolve_corpus does the TTY-aware thing; scripts must pass --corpus.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: CLI `check`, `estimate-cost`, `export`, `store reidentify-orgs` — `--corpus` required + wired

**Files:**
- Modify: `consistency_checker/cli/main.py`
- Test: extend `tests/test_cli.py` (+ touch `tests/test_cli_reidentify.py`, `tests/test_estimate_cost.py` as needed)

- [ ] **Step 1: Write failing tests**

Four tests — one per command — using the same monkeypatch pattern as Task 7. Each follows the shape:

```python
def test_check_without_corpus_errors_in_non_tty(monkeypatch, tmp_path):
    from typer.testing import CliRunner
    from consistency_checker.cli.main import app
    cfg = tmp_path / "config.yml"
    cfg.write_text(f"corpus_dir: {tmp_path}\ndata_dir: {tmp_path}\n", encoding="utf-8")
    runner = CliRunner()
    res = runner.invoke(app, ["check", "--config", str(cfg)])
    assert res.exit_code != 0
    assert "--corpus is required" in (res.output + str(res.exception or ""))


def test_check_with_corpus_passes_corpus_id_to_pipeline(monkeypatch, tmp_path):
    """Spy on pipeline.check to capture the corpus_id it was called with."""
    import consistency_checker.cli.main as cli_main
    captured = {}
    def fake_check(config, *, store, faiss_store, corpus_id, **kw):
        captured["corpus_id"] = corpus_id
        from consistency_checker.pipeline import CheckResult
        return CheckResult()
    monkeypatch.setattr(cli_main, "pipeline_check", fake_check, raising=False)
    # ... configure store with an "atkins" corpus, invoke runner, assert
    # captured["corpus_id"] == the atkins corpus_id.
```

Repeat for `estimate-cost`, `export`, and `store reidentify-orgs`, each with the same `--corpus is required` and `pass-through` shape.

- [ ] **Step 2: Run — confirm FAIL**

- [ ] **Step 3: Add the `--corpus` option to each command and the same `resolve_corpus(...)` resolution step before invoking the pipeline / store path**

Pattern, applied 4 times:

```python
corpus: str | None = typer.Option(None, "--corpus"),
# ...
corpus_id = resolve_corpus(store, corpus, str(config.corpus_dir), config.judge_provider)
# pass corpus_id to pipeline.check / pipeline.estimate_cost / store.iter_documents (export) / store reidentify-orgs main loop
```

For `report`: don't add `--corpus required`; instead, infer from `pipeline_runs.corpus_id` (since the run encodes its corpus). Mismatch with an explicit `--corpus` flag is an error.

- [ ] **Step 4: Run — confirm PASS**

- [ ] **Step 5: Full gate + commit**

```sh
git add consistency_checker/cli/main.py tests/test_cli.py tests/test_cli_reidentify.py tests/test_estimate_cost.py
git commit -m "$(cat <<'EOF'
feat(cli): check / estimate-cost / export / store reidentify-orgs require --corpus

All four mutating-or-judging commands go through resolve_corpus so the
TTY-vs-script policy is uniform. report infers from pipeline_runs.corpus_id.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: CLI `corpus list / delete / reassign` subcommand group

**Files:**
- Modify: `consistency_checker/cli/main.py`
- Test: `tests/test_cli_corpus_subcommands.py` (new)

- [ ] **Step 1: Write failing tests**

```python
def test_corpus_list_shows_each_corpus_with_doc_count(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from consistency_checker.cli.main import app
    from consistency_checker.extract.schema import Document
    from consistency_checker.index.assertion_store import AssertionStore
    db = tmp_path / "t.db"; store = AssertionStore(db); store.migrate()
    a = store.get_or_create_corpus("alpha", "/a", "moonshot")
    store.add_document(Document(doc_id="d1", source_path="/x"), corpus_id=a)
    store.close()

    runner = CliRunner()
    res = runner.invoke(app, ["corpus", "list", "--db", str(db)])
    assert res.exit_code == 0
    assert "alpha" in res.output
    assert "1" in res.output  # doc count


def test_corpus_delete_requires_yes_i_mean_it(tmp_path):
    from typer.testing import CliRunner
    from consistency_checker.cli.main import app
    from consistency_checker.index.assertion_store import AssertionStore
    db = tmp_path / "t.db"; store = AssertionStore(db); store.migrate()
    store.get_or_create_corpus("alpha", "/a", "moonshot")
    store.close()

    runner = CliRunner()
    res = runner.invoke(app, ["corpus", "delete", "alpha", "--db", str(db)])
    assert res.exit_code != 0
    assert "--yes-i-mean-it" in res.output

    res2 = runner.invoke(app, ["corpus", "delete", "alpha", "--yes-i-mean-it", "--db", str(db)])
    assert res2.exit_code == 0
    store = AssertionStore(db)
    assert store.list_corpora() == []
    store.close()


def test_corpus_reassign_moves_matching_rows(tmp_path):
    from typer.testing import CliRunner
    from consistency_checker.cli.main import app
    from consistency_checker.extract.schema import Document
    from consistency_checker.index.assertion_store import AssertionStore
    db = tmp_path / "t.db"; store = AssertionStore(db); store.migrate()
    legacy = store.get_or_create_corpus("legacy", "/legacy", "moonshot")
    store.add_document(
        Document(doc_id="d1", source_path="/x", org_label="ATKINS NUTRITIONALS", org_reason="org_found"),
        corpus_id=legacy,
    )
    store.add_document(
        Document(doc_id="d2", source_path="/y", org_label="Lockhart Springs", org_reason="org_found"),
        corpus_id=legacy,
    )
    store.close()

    runner = CliRunner()
    res = runner.invoke(
        app,
        ["corpus", "reassign", "--db", str(db),
         "--from", "legacy", "--to", "atkins",
         "--where", "org_label LIKE 'ATKINS%'"],
    )
    assert res.exit_code == 0, res.output
    assert "Moved 1 document" in res.output

    store = AssertionStore(db)
    atkins_id = next(c.corpus_id for c in store.list_corpora() if c.corpus_name == "atkins")
    d1_cid = store._conn.execute("SELECT corpus_id FROM documents WHERE doc_id='d1'").fetchone()[0]
    d2_cid = store._conn.execute("SELECT corpus_id FROM documents WHERE doc_id='d2'").fetchone()[0]
    assert d1_cid == atkins_id
    assert d2_cid == legacy
    store.close()


def test_corpus_reassign_rejects_unsafe_where_clause(tmp_path):
    from typer.testing import CliRunner
    from consistency_checker.cli.main import app
    from consistency_checker.index.assertion_store import AssertionStore
    db = tmp_path / "t.db"; store = AssertionStore(db); store.migrate()
    store.get_or_create_corpus("legacy", "/l", "moonshot")
    store.close()
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["corpus", "reassign", "--db", str(db), "--from", "legacy", "--to", "x",
         "--where", "1=1; DROP TABLE documents"],
    )
    assert res.exit_code != 0
    assert "--where" in (res.output + str(res.exception or ""))
```

- [ ] **Step 2: Run — confirm FAIL**

- [ ] **Step 3: Implement the subcommand group**

In `consistency_checker/cli/main.py`:

```python
corpus_app = typer.Typer(help="Inspect or maintain corpora.")
app.add_typer(corpus_app, name="corpus")


@corpus_app.command("list")
def corpus_list(db: Path = typer.Option(...,"--db")) -> None:
    store = AssertionStore(db); store.migrate()
    try:
        for c in store.list_corpora():
            stats = store.stats(corpus_id=c.corpus_id)
            typer.echo(f"{c.corpus_name:20s}  {stats['documents']:>4d} docs  ({c.corpus_path})")
    finally:
        store.close()


@corpus_app.command("delete")
def corpus_delete(
    name: str,
    db: Path = typer.Option(...,"--db"),
    yes: bool = typer.Option(False, "--yes-i-mean-it"),
) -> None:
    if not yes:
        raise typer.BadParameter("Refusing to delete without --yes-i-mean-it")
    store = AssertionStore(db); store.migrate()
    try:
        match = next((c for c in store.list_corpora() if c.corpus_name == name), None)
        if not match:
            available = ", ".join(c.corpus_name for c in store.list_corpora()) or "<none>"
            raise typer.BadParameter(f"No corpus named {name!r} (available: {available})")
        store.delete_corpus(match.corpus_id)
        typer.echo(f"Deleted corpus {name!r}.")
    finally:
        store.close()


@corpus_app.command("reassign")
def corpus_reassign(
    db: Path = typer.Option(...,"--db"),
    src: str = typer.Option(..., "--from"),
    dst: str = typer.Option(..., "--to"),
    where: str = typer.Option("", "--where", help="Safe-listed WHERE clause for documents."),
) -> None:
    """Move documents from --from corpus into --to. --where is a safe-listed
    suffix; only [A-Za-z0-9_=' %]+ allowed (plus LIKE, AND, OR)."""
    import re
    if where and not re.fullmatch(r"[A-Za-z0-9_=' %]+(?:\s+(?:LIKE|AND|OR)\s+[A-Za-z0-9_=' %]+)*", where):
        raise typer.BadParameter(
            "--where allows only column=literal / column LIKE 'pattern' joined by AND/OR"
        )
    store = AssertionStore(db); store.migrate()
    try:
        src_id = next((c.corpus_id for c in store.list_corpora() if c.corpus_name == src), None)
        dst_id = store.get_or_create_corpus(dst, "(reassigned)", "moonshot")
        if src_id is None:
            raise typer.BadParameter(f"--from corpus {src!r} not found")
        sql = "UPDATE documents SET corpus_id = ? WHERE corpus_id = ?"
        params: list = [dst_id, src_id]
        if where:
            sql += f" AND ({where})"
        with store._conn:
            cur = store._conn.execute(sql, params)
            typer.echo(f"Moved {cur.rowcount} document(s) from {src} to {dst}.")
    finally:
        store.close()
```

- [ ] **Step 4: Run — confirm PASS**

- [ ] **Step 5: Full gate + commit**

```sh
git add consistency_checker/cli/main.py tests/test_cli_corpus_subcommands.py
git commit -m "$(cat <<'EOF'
feat(cli): corpus list / delete / reassign auxiliary commands

list shows doc counts per corpus. delete cascades via the FK chain
(reviewer_verdicts orphans accepted in v1). reassign moves docs between
corpora using a safe-listed --where clause (column=literal / LIKE).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Web UI — `_ingest_uploaded_paths` accepts corpus_id; every read route filters

**Files:**
- Modify: `consistency_checker/web/app.py`
- Test: extend `tests/test_web_corpus_banner.py`; add a route-level corpus-filter test

- [ ] **Step 1: Write failing tests**

```python
def test_web_upload_requires_corpus_id(client):
    r = client.post("/uploads", files=[("file", ("doc.txt", b"hello", "text/plain"))])
    # without corpus_id: 400
    assert r.status_code == 400
    assert "corpus" in r.text.lower()


def test_web_upload_with_corpus_id_persists_link(client_factory, tmp_path):
    """client_factory mirrors existing tests/test_web_*.py fixtures — it
    returns (TestClient, AssertionStore) backed by a pre-migrated DB."""
    client, store = client_factory(tmp_path)
    cid = store.get_or_create_corpus("atkins", "/atkins", "moonshot")

    r = client.post(
        "/uploads",
        data={"corpus_id": cid},
        files=[("file", ("doc.txt", b"Atkins bylaws", "text/plain"))],
    )
    assert r.status_code in (200, 303)
    rows = store._conn.execute(
        "SELECT corpus_id FROM documents"
    ).fetchall()
    assert rows and all(r[0] == cid for r in rows)


def test_stats_tab_filters_to_selected_corpus(client_with_two_corpora):
    r = client_with_two_corpora.get("/tabs/stats?corpus=atkins")
    assert "atkins" in r.text
    assert "lockhart" not in r.text
```

- [ ] **Step 2: Run — confirm FAIL**

- [ ] **Step 3: Modify `_ingest_uploaded_paths`**

Accept a `corpus_id` parameter; the upload form posts a `corpus_id` field; the handler returns 400 if absent. Inside the loop, call `store.add_document(doc, corpus_id=corpus_id)`.

- [ ] **Step 4: Filter every read route by selected corpus**

For each route that calls `store.iter_documents()`, `store.iter_definitions()`, or queries `findings` / `assertions` directly, add a `corpus_id = <selected>` filter. The selector value is already in the request — read from `request.query_params.get("corpus")` or the session/cookie pattern existing routes use.

Update `_corpus_banner_context` to scope to the selected corpus rather than all documents (otherwise warnings would surface multi-org noise from other corpora).

- [ ] **Step 5: Run — confirm PASS**

- [ ] **Step 6: Full gate + commit**

```sh
git add consistency_checker/web/app.py tests/test_web_corpus_banner.py
git commit -m "$(cat <<'EOF'
feat(web): upload requires corpus_id; read routes filter by selected corpus

_ingest_uploaded_paths persists corpus_id alongside the doc (mirrors the
CLI path PR #64 wired). _corpus_banner_context and every aggregate
read scope to the selected corpus.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: ADR-0013 + futureplans bookkeeping + final gate

**Files:**
- Create: `docs/decisions/0013-corpus-isolation.md`
- Modify: `futureplans.md`

- [ ] **Step 1: Write ADR-0013**

```markdown
# ADR-0013 — Corpus isolation (logical, single DB)

**Date:** 2026-05-25
**Status:** Accepted
**Spec:** [`docs/superpowers/specs/2026-05-25-corpus-isolation-design.md`]
**Plan:** [`docs/superpowers/plans/2026-05-25-corpus-isolation.md`]

## Context

A single-DB pool with no CLI-level scoping forced a choice between
polluting prior runs or deleting them. Surfaced mid-Atkins shake-down
test 2026-05-25.

## Decision

- Logical isolation: every document and pipeline_runs row carries a
  TEXT corpus_id FK to corpora. FAISS index stays shared.
- --corpus is required on every mutating/judging CLI command;
  interactive prompt on TTY, hard error in scripts.
- Pre-isolation rows backfill to a "legacy" corpus by migration 0014.
- FAISS gate post-filter drops any candidate pair whose endpoints
  aren't both in the corpus's assertion-id set — mandatory, silent.

## Rejected

- Per-corpus DB files / per-corpus directories (Q3-B/C): cleaner but
  every CLI command would need refactoring; corpora table would move
  to a top-level registry. Bigger change for marginal isolation gain
  given FAISS still shared.
- Default-on suppression / optional --corpus: would re-introduce the
  same accidental-pollution risk the spec exists to eliminate.

## Consequences

- One extra mandatory CLI flag on five commands; interactive picker
  reduces friction on TTY.
- reviewer_verdicts has no FK to assertions → corpus delete leaves
  orphans. Accepted in v1; cleanup is a follow-up.
- Companion archive spec (deferred) builds on this: a corpus can be
  exported to a portable artifact once it's a real isolation unit.
```

- [ ] **Step 2: Update `futureplans.md`**

Add to Completed:

```markdown
- **Corpus isolation (item: retention gap, 2026-05-25)**
  Spec: `docs/superpowers/specs/2026-05-25-corpus-isolation-design.md`.
  Plan: `docs/superpowers/plans/2026-05-25-corpus-isolation.md`.
  ADR-0013. Migration 0014. --corpus required on ingest/check/
  estimate-cost/export/store reidentify-orgs. FAISS gate post-filter.
  Companion archive spec deferred.
```

Add a new open-item line for the deferred archive spec:

```markdown
- **Corpus archive (deferred)** — bundle a corpus + its runs + verdicts
  into a portable artifact for review and off-machine retention.
  Companion to ADR-0013; will be drafted post-isolation-merge.
```

- [ ] **Step 3: Final full gate**

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy consistency_checker
uv run pytest -m "not slow and not live"
uv build
```

All five must succeed.

- [ ] **Step 4: Commit**

```sh
git add docs/decisions/0013-corpus-isolation.md futureplans.md
git commit -m "$(cat <<'EOF'
docs(adr): record corpus-isolation decision; futureplans bookkeeping

ADR-0013 documents the logical-isolation choice (Q3-A), rejected
alternatives (per-corpus DB / per-corpus directory), --corpus-required
policy, and the reviewer_verdicts orphan acceptance.

Companion archive spec moves to deferred.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review checklist (run by the engineer)

After all 11 tasks land, before final merge to main:

1. **Operator workflow:** Run `consistency-check ingest <dir> --corpus atkins`, then `check --corpus atkins`. Confirm no warnings about cross-corpus leaks; check the run summary line for `n_cross_corpus_gate_drops`.
2. **Backfill verification:** On the operator's existing DB, confirm migration 0014 created a `legacy` corpus and assigned all pre-existing docs to it. Run `consistency-check corpus list` to see counts.
3. **Reassignment for in-flight Atkins:** Run `consistency-check corpus reassign --from legacy --to atkins --where "org_label LIKE 'ATKINS%'"` and verify the row count matches expectations.
4. **/review before each push** (per memory `feedback_review_before_push.md`). Address all critical/moderate findings before pushing.
5. **§9 measurement still applies:** the corpus-org-warning §9 measurement (definition-divergent rate delta on the Atkins corpus) can now actually be run cleanly, since the Atkins corpus is isolated from legacy noise.
