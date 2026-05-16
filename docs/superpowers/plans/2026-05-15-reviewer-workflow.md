# Reviewer-workflow implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the read-only web UI into a workflow tool by letting the user mark each finding with a three-value verdict (Real issue / Not an issue / Skip), persistent across re-runs of the same corpus.

**Architecture:** New `reviewer_verdicts` table content-keyed by `(pair_key, detector_type)` so verdicts survive re-runs. `AuditLogger` gains four CRUD methods. Two new POST routes (`/verdicts`, `/verdicts/undo`) plus three new partial templates handle the UI. Each finding tab's section gets a "X of N reviewed" progress count, a "Show reviewed" toggle (default off — reviewed rows hidden), inline icon-only verdict buttons (with keyboard shortcuts C/F/D), and a persistent undo toast. Markdown report filters `false_positive` and tags surviving findings.

**Tech Stack:** Python 3.11+, SQLite (canonical store, with CHECK constraints), Pydantic v2 (not used here — pure dataclasses), FastAPI + HTMX (no JS framework), Jinja2 templates, pytest (hermetic only — no slow/live marks), ruff + mypy strict.

**Reference:** [`docs/superpowers/specs/2026-05-15-reviewer-workflow-design.md`](../specs/2026-05-15-reviewer-workflow-design.md). The spec has the full SQL, the route shapes, the template skeletons, and the rationale. This plan turns it into bite-sized TDD tasks.

---

## Phase 0 — Schema + helpers

### Task 1: Migration 0009 — `reviewer_verdicts` table

**Files:**
- Create: `consistency_checker/index/migrations/0009_reviewer_verdicts.sql`
- Test: `tests/test_migrations.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_migrations.py`:

```python
def test_migration_0009_adds_reviewer_verdicts_table(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    cols = {
        row["name"]: row
        for row in store._conn.execute("PRAGMA table_info(reviewer_verdicts)").fetchall()
    }
    assert set(cols.keys()) == {"pair_key", "detector_type", "verdict", "set_at", "note"}
    # Composite PK: both pair_key AND detector_type carry pk > 0
    assert cols["pair_key"]["pk"] >= 1
    assert cols["detector_type"]["pk"] >= 1
    # NOT NULL on the four required columns
    for c in ("pair_key", "detector_type", "verdict", "set_at"):
        assert cols[c]["notnull"] == 1
    # note is nullable
    assert cols["note"]["notnull"] == 0
    store.close()


def test_migration_0009_check_constraints_reject_bogus_values(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    # Bad verdict
    with pytest.raises(sqlite3.IntegrityError, match=r"CHECK"):
        store._conn.execute(
            "INSERT INTO reviewer_verdicts (pair_key, detector_type, verdict) "
            "VALUES ('a:b', 'contradiction', 'banana')"
        )
    # Bad detector_type
    with pytest.raises(sqlite3.IntegrityError, match=r"CHECK"):
        store._conn.execute(
            "INSERT INTO reviewer_verdicts (pair_key, detector_type, verdict) "
            "VALUES ('a:b', 'made_up_detector', 'confirmed')"
        )
    store.close()


def test_migration_0009_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    AssertionStore(db).migrate()
    store = AssertionStore(db)
    assert store.migrate() == []
    store.close()
```

Add the missing import at the top of the file if not present:
```python
import sqlite3
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python3 -m pytest tests/test_migrations.py -v -k 0009
```

Expected: FAIL (table doesn't exist).

- [ ] **Step 3: Write the migration**

Create `consistency_checker/index/migrations/0009_reviewer_verdicts.sql`:

```sql
-- 0009_reviewer_verdicts.sql
--
-- Verdicts are content-addressed (not run-scoped) so re-checking the same
-- corpus does not wipe review work. The pair_key construction MUST match the
-- Python builder in `consistency_checker.audit.reviewer.build_pair_key`:
--   pair findings:     min(a_id, b_id) || ':' || max(a_id, b_id)
--   triangle findings: ':'.join(sorted(assertion_ids))
-- If this formula diverges, render-time joins go silently empty.

CREATE TABLE reviewer_verdicts (
  pair_key TEXT NOT NULL,
  detector_type TEXT NOT NULL
    CHECK (detector_type IN ('contradiction', 'definition_inconsistency', 'multi_party')),
  verdict TEXT NOT NULL
    CHECK (verdict IN ('confirmed', 'false_positive', 'dismissed')),
  set_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  note TEXT,
  PRIMARY KEY (pair_key, detector_type)
);
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python3 -m pytest tests/test_migrations.py -v -k 0009
```

Expected: PASS (3 tests).

- [ ] **Step 5: Run lint + the wider migration suite to catch regressions**

```bash
python3 -m ruff check consistency_checker/index/migrations/ tests/test_migrations.py
python3 -m pytest tests/test_migrations.py -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add consistency_checker/index/migrations/0009_reviewer_verdicts.sql tests/test_migrations.py
git commit -m "feat(schema): migration 0009 — reviewer_verdicts table with CHECK constraints"
```

---

### Task 2: `consistency_checker/audit/reviewer.py` module

**Files:**
- Create: `consistency_checker/audit/reviewer.py`
- Test: `tests/test_reviewer.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reviewer.py`:

```python
"""Tests for the reviewer-verdict types and pair_key helper."""

from __future__ import annotations

import pytest

from consistency_checker.audit.reviewer import (
    ReviewerVerdict,
    build_pair_key,
)


def test_build_pair_key_two_assertions_sorted() -> None:
    assert build_pair_key("b", "a") == "a:b"
    assert build_pair_key("a", "b") == "a:b"


def test_build_pair_key_three_assertions_sorted() -> None:
    assert build_pair_key("c", "a", "b") == "a:b:c"


def test_build_pair_key_rejects_singleton() -> None:
    with pytest.raises(ValueError, match=r"at least 2 assertion ids"):
        build_pair_key("a")
    with pytest.raises(ValueError, match=r"at least 2 assertion ids"):
        build_pair_key()


def test_reviewer_verdict_dataclass_defaults() -> None:
    v = ReviewerVerdict(
        pair_key="a:b",
        detector_type="contradiction",
        verdict="confirmed",
    )
    assert v.set_at is None
    assert v.note is None
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python3 -m pytest tests/test_reviewer.py -v
```

Expected: FAIL (module not defined).

- [ ] **Step 3: Implement the module**

Create `consistency_checker/audit/reviewer.py`:

```python
"""Reviewer-verdict types and the canonical pair_key builder.

The pair_key formula here MUST match the SQL CASE WHEN expression used by
render-time joins (see migration 0009 header for the formula).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

ReviewerVerdictLabel = Literal["confirmed", "false_positive", "dismissed"]
DetectorType = Literal["contradiction", "definition_inconsistency", "multi_party"]


def build_pair_key(*assertion_ids: str) -> str:
    """Canonical pair_key for a finding (works for pair or triangle)."""
    if len(assertion_ids) < 2:
        raise ValueError("build_pair_key needs at least 2 assertion ids")
    return ":".join(sorted(assertion_ids))


@dataclass(frozen=True, slots=True)
class ReviewerVerdict:
    """One reviewer-set verdict for a finding."""

    pair_key: str
    detector_type: DetectorType
    verdict: ReviewerVerdictLabel
    set_at: datetime | None = None
    note: str | None = None
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python3 -m pytest tests/test_reviewer.py -v
python3 -m ruff check consistency_checker/audit/reviewer.py tests/test_reviewer.py
```

Expected: 4 passed, lint clean.

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/audit/reviewer.py tests/test_reviewer.py
git commit -m "feat(audit): reviewer verdict types + pair_key builder"
```

---

## Phase 1 — AuditLogger surface

### Task 3: Write methods — `set_reviewer_verdict` + `delete_reviewer_verdict`

**Files:**
- Modify: `consistency_checker/audit/logger.py`
- Test: `tests/test_audit_logger.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_audit_logger.py`:

```python
def test_set_reviewer_verdict_inserts_row(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    logger = AuditLogger(store)
    logger.set_reviewer_verdict(
        pair_key="a:b",
        detector_type="contradiction",
        verdict="confirmed",
    )
    rows = store._conn.execute(
        "SELECT pair_key, detector_type, verdict, note FROM reviewer_verdicts"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["pair_key"] == "a:b"
    assert rows[0]["detector_type"] == "contradiction"
    assert rows[0]["verdict"] == "confirmed"
    assert rows[0]["note"] is None
    store.close()


def test_set_reviewer_verdict_upserts_on_conflict(tmp_path: Path) -> None:
    """Second set with different verdict overwrites; set_at refreshes."""
    import time
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    logger = AuditLogger(store)
    logger.set_reviewer_verdict(
        pair_key="a:b", detector_type="contradiction", verdict="confirmed", note="initial"
    )
    first = store._conn.execute(
        "SELECT verdict, note, set_at FROM reviewer_verdicts"
    ).fetchone()
    time.sleep(1.1)  # SQLite CURRENT_TIMESTAMP has 1-second resolution
    logger.set_reviewer_verdict(
        pair_key="a:b", detector_type="contradiction", verdict="false_positive", note="changed"
    )
    second = store._conn.execute(
        "SELECT verdict, note, set_at FROM reviewer_verdicts"
    ).fetchone()
    assert second["verdict"] == "false_positive"
    assert second["note"] == "changed"
    assert second["set_at"] > first["set_at"]
    store.close()


def test_set_reviewer_verdict_same_pair_different_detector_coexist(tmp_path: Path) -> None:
    """Composite PK lets a single content pair carry independent verdicts per detector."""
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    logger = AuditLogger(store)
    logger.set_reviewer_verdict(
        pair_key="a:b", detector_type="contradiction", verdict="confirmed"
    )
    logger.set_reviewer_verdict(
        pair_key="a:b", detector_type="definition_inconsistency", verdict="false_positive"
    )
    rows = store._conn.execute(
        "SELECT detector_type, verdict FROM reviewer_verdicts ORDER BY detector_type"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["detector_type"] == "contradiction"
    assert rows[0]["verdict"] == "confirmed"
    assert rows[1]["detector_type"] == "definition_inconsistency"
    assert rows[1]["verdict"] == "false_positive"
    store.close()


def test_delete_reviewer_verdict_removes_row(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    logger = AuditLogger(store)
    logger.set_reviewer_verdict(
        pair_key="a:b", detector_type="contradiction", verdict="confirmed"
    )
    logger.delete_reviewer_verdict(pair_key="a:b", detector_type="contradiction")
    rows = store._conn.execute("SELECT COUNT(*) FROM reviewer_verdicts").fetchone()
    assert rows[0] == 0
    store.close()


def test_delete_reviewer_verdict_targets_only_matching_detector(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    logger = AuditLogger(store)
    logger.set_reviewer_verdict(
        pair_key="a:b", detector_type="contradiction", verdict="confirmed"
    )
    logger.set_reviewer_verdict(
        pair_key="a:b", detector_type="definition_inconsistency", verdict="false_positive"
    )
    logger.delete_reviewer_verdict(pair_key="a:b", detector_type="contradiction")
    rows = store._conn.execute(
        "SELECT detector_type FROM reviewer_verdicts"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["detector_type"] == "definition_inconsistency"
    store.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python3 -m pytest tests/test_audit_logger.py -v -k reviewer
```

Expected: FAIL (methods undefined).

- [ ] **Step 3: Add the import + methods to AuditLogger**

In `consistency_checker/audit/logger.py`, add to the imports (alongside the existing `TYPE_CHECKING` block):

```python
from consistency_checker.audit.reviewer import DetectorType, ReviewerVerdictLabel
```

Then add to the `AuditLogger` class (alongside the existing `record_*` methods):

```python
def set_reviewer_verdict(
    self,
    *,
    pair_key: str,
    detector_type: DetectorType,
    verdict: ReviewerVerdictLabel,
    note: str | None = None,
) -> None:
    """UPSERT a reviewer verdict. Refreshes set_at on every call."""
    with self._conn:
        self._conn.execute(
            "INSERT INTO reviewer_verdicts (pair_key, detector_type, verdict, note) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT (pair_key, detector_type) "
            "DO UPDATE SET verdict = excluded.verdict, "
            "              note = excluded.note, "
            "              set_at = CURRENT_TIMESTAMP",
            (pair_key, detector_type, verdict, note),
        )

def delete_reviewer_verdict(
    self, *, pair_key: str, detector_type: DetectorType
) -> None:
    """Remove a verdict (backs the undo toast for first-click cases)."""
    with self._conn:
        self._conn.execute(
            "DELETE FROM reviewer_verdicts "
            "WHERE pair_key = ? AND detector_type = ?",
            (pair_key, detector_type),
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python3 -m pytest tests/test_audit_logger.py -v -k reviewer
python3 -m ruff check consistency_checker/audit/logger.py tests/test_audit_logger.py
```

Expected: 5 passed, lint clean.

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/audit/logger.py tests/test_audit_logger.py
git commit -m "feat(audit): set/delete_reviewer_verdict on AuditLogger"
```

---

### Task 4: Read methods — `get_reviewer_verdicts_bulk` + `count_reviewer_verdicts`

**Files:**
- Modify: `consistency_checker/audit/logger.py`
- Test: `tests/test_audit_logger.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_audit_logger.py`:

```python
def test_get_reviewer_verdicts_bulk_returns_present_only(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    logger = AuditLogger(store)
    logger.set_reviewer_verdict(
        pair_key="a:b", detector_type="contradiction", verdict="confirmed"
    )
    logger.set_reviewer_verdict(
        pair_key="c:d", detector_type="definition_inconsistency", verdict="dismissed"
    )
    out = logger.get_reviewer_verdicts_bulk(
        [
            ("a:b", "contradiction"),
            ("c:d", "definition_inconsistency"),
            ("ghost:ghost", "contradiction"),  # not present
        ]
    )
    assert set(out.keys()) == {("a:b", "contradiction"), ("c:d", "definition_inconsistency")}
    assert out[("a:b", "contradiction")].verdict == "confirmed"
    store.close()


def test_get_reviewer_verdicts_bulk_empty_input_no_query(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    logger = AuditLogger(store)
    assert logger.get_reviewer_verdicts_bulk([]) == {}
    store.close()


def test_count_reviewer_verdicts_groups_by_verdict(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    logger = AuditLogger(store)
    logger.set_reviewer_verdict(pair_key="a:b", detector_type="contradiction", verdict="confirmed")
    logger.set_reviewer_verdict(pair_key="c:d", detector_type="contradiction", verdict="confirmed")
    logger.set_reviewer_verdict(pair_key="e:f", detector_type="contradiction", verdict="dismissed")
    counts = logger.count_reviewer_verdicts()
    assert counts == {"confirmed": 2, "dismissed": 1}
    store.close()


def test_count_reviewer_verdicts_filtered_by_detector(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "test.db")
    store.migrate()
    logger = AuditLogger(store)
    logger.set_reviewer_verdict(pair_key="a:b", detector_type="contradiction", verdict="confirmed")
    logger.set_reviewer_verdict(
        pair_key="c:d", detector_type="definition_inconsistency", verdict="confirmed"
    )
    counts = logger.count_reviewer_verdicts(detector_type="contradiction")
    assert counts == {"confirmed": 1}
    store.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python3 -m pytest tests/test_audit_logger.py -v -k "get_reviewer or count_reviewer"
```

Expected: FAIL.

- [ ] **Step 3: Implement the read methods**

In `consistency_checker/audit/logger.py`, add to the imports:

```python
from collections.abc import Sequence
from consistency_checker.audit.reviewer import ReviewerVerdict
```

Add a tiny row converter near `_row_to_finding`:

```python
def _row_to_reviewer_verdict(row: sqlite3.Row) -> ReviewerVerdict:
    return ReviewerVerdict(
        pair_key=row["pair_key"],
        detector_type=row["detector_type"],
        verdict=row["verdict"],
        set_at=_parse_ts(row["set_at"]),
        note=row["note"],
    )
```

Add the two read methods to `AuditLogger`:

```python
def get_reviewer_verdicts_bulk(
    self,
    keys: Sequence[tuple[str, DetectorType]],
) -> dict[tuple[str, str], ReviewerVerdict]:
    """Fetch verdicts for the given (pair_key, detector_type) tuples.

    Uses SQLite IN (VALUES (?, ?), ...) for batch lookup in one query.
    Missing keys are absent from the returned dict.
    """
    if not keys:
        return {}
    values_placeholders = ", ".join("(?, ?)" for _ in keys)
    flat: list[str] = []
    for pk, dt in keys:
        flat.extend([pk, dt])
    rows = self._conn.execute(
        f"SELECT pair_key, detector_type, verdict, set_at, note "
        f"FROM reviewer_verdicts "
        f"WHERE (pair_key, detector_type) IN (VALUES {values_placeholders})",
        flat,
    ).fetchall()
    return {(r["pair_key"], r["detector_type"]): _row_to_reviewer_verdict(r) for r in rows}

def count_reviewer_verdicts(
    self, *, detector_type: DetectorType | None = None
) -> dict[str, int]:
    """Counts per verdict label. Used for 'X of N reviewed' progress."""
    if detector_type is None:
        sql = "SELECT verdict, COUNT(*) AS c FROM reviewer_verdicts GROUP BY verdict"
        params: tuple = ()
    else:
        sql = (
            "SELECT verdict, COUNT(*) AS c FROM reviewer_verdicts "
            "WHERE detector_type = ? GROUP BY verdict"
        )
        params = (detector_type,)
    return {r["verdict"]: int(r["c"]) for r in self._conn.execute(sql, params).fetchall()}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python3 -m pytest tests/test_audit_logger.py -v
python3 -m ruff check consistency_checker/audit/logger.py
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/audit/logger.py tests/test_audit_logger.py
git commit -m "feat(audit): get_reviewer_verdicts_bulk + count_reviewer_verdicts"
```

---

## Phase 2 — Web POST endpoints

### Task 5: `POST /verdicts` endpoint

**Files:**
- Modify: `consistency_checker/web/app.py`
- Create: `consistency_checker/web/templates/cc__verdict_toast.html`
- Create: `consistency_checker/web/templates/cc__progress_count.html`
- Test: `tests/test_web_verdicts.py` (new)

- [ ] **Step 1: Create the OOB-swap partials**

Create `consistency_checker/web/templates/cc__verdict_toast.html`:

```html
<div id="cc-toast" class="cc-toast" role="status" hx-swap-oob="outerHTML">
  Marked as <strong>{{ verdict_label }}</strong>.
  <button class="cc-toast-undo"
          hx-post="/verdicts/undo"
          hx-vals='{"pair_key": "{{ pair_key }}",
                    "detector_type": "{{ detector_type }}",
                    "prior_verdict": "{{ prior_verdict|default('') }}"}'
          hx-target="#cc-tab-content"
          hx-swap="innerHTML">Undo</button>
  <button class="cc-toast-close"
          type="button"
          aria-label="Dismiss notification"
          onclick="this.parentElement.remove()">×</button>
</div>
```

Create `consistency_checker/web/templates/cc__progress_count.html`:

```html
<span id="cc-progress-count-{{ detector_type }}" class="cc-progress-count" hx-swap-oob="outerHTML">
  {{ reviewed_count }} of {{ total_count }} reviewed
</span>
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_web_verdicts.py`:

```python
"""Tests for the verdict-setting + undo POST endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.check.llm_judge import FixtureJudge
from consistency_checker.check.nli_checker import FixtureNliChecker
from consistency_checker.config import Config
from consistency_checker.extract.atomic_facts import FixtureExtractor
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.web.app import create_app
from tests.conftest import HashEmbedder


@pytest.fixture
def app_client(tmp_path: Path) -> tuple[TestClient, Config]:
    cfg = Config(
        corpus_dir=tmp_path / "corpus",
        judge_provider="fixture",
        judge_model="test",
        data_dir=tmp_path / "store",
        log_dir=tmp_path / "logs",
        embedder_model="hash",
        nli_model="fixture",
    )
    AssertionStore(cfg.db_path).migrate()
    app = create_app(
        cfg,
        extractor=FixtureExtractor({}),
        embedder=HashEmbedder(dim=64),
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
    )
    return TestClient(app), cfg


def test_post_verdicts_inserts_row(
    app_client: tuple[TestClient, Config],
) -> None:
    client, cfg = app_client
    resp = client.post(
        "/verdicts",
        data={
            "pair_key": "a:b",
            "detector_type": "contradiction",
            "verdict": "confirmed",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    # Row exists
    store = AssertionStore(cfg.db_path)
    rows = store._conn.execute(
        "SELECT verdict FROM reviewer_verdicts WHERE pair_key = ?", ("a:b",)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["verdict"] == "confirmed"
    store.close()


def test_post_verdicts_response_contains_oob_swaps(
    app_client: tuple[TestClient, Config],
) -> None:
    client, _cfg = app_client
    resp = client.post(
        "/verdicts",
        data={
            "pair_key": "a:b",
            "detector_type": "contradiction",
            "verdict": "confirmed",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    # Toast OOB swap present
    assert 'id="cc-toast"' in resp.text
    assert 'hx-swap-oob="outerHTML"' in resp.text
    # Progress count OOB swap present
    assert 'id="cc-progress-count-contradiction"' in resp.text
    # Plain-English verdict label
    assert "Real issue" in resp.text
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
python3 -m pytest tests/test_web_verdicts.py -v -k post_verdicts
```

Expected: FAIL (route undefined).

- [ ] **Step 4: Add the POST /verdicts route**

In `consistency_checker/web/app.py`, alongside the other `@app.post` routes, add the verdict label map at module-top:

```python
VERDICT_LABELS: dict[str, str] = {
    "confirmed": "Real issue",
    "false_positive": "Not an issue",
    "dismissed": "Dismissed",
}
```

Then inside `create_app(...)`, add the route:

```python
@app.post("/verdicts", response_class=HTMLResponse)
def post_verdict(
    request: Request,
    pair_key: str = Form(...),
    detector_type: str = Form(...),
    verdict: str = Form(...),
    prior_verdict: str = Form(""),
) -> HTMLResponse:
    if detector_type not in {"contradiction", "definition_inconsistency", "multi_party"}:
        raise HTTPException(status_code=400, detail=f"unknown detector_type {detector_type!r}")
    if verdict not in {"confirmed", "false_positive", "dismissed"}:
        raise HTTPException(status_code=400, detail=f"unknown verdict {verdict!r}")

    store, audit = _open_audit()
    try:
        audit.set_reviewer_verdict(
            pair_key=pair_key,
            detector_type=detector_type,  # type: ignore[arg-type]
            verdict=verdict,  # type: ignore[arg-type]
        )
        counts = audit.count_reviewer_verdicts(detector_type=detector_type)  # type: ignore[arg-type]
        reviewed_count = sum(counts.values())
        total_count = _count_total_findings(store, detector_type)
    finally:
        store.close()

    # Response is empty body for the row swap + two OOB swap fragments.
    toast = templates.get_template("cc__verdict_toast.html").render(
        verdict_label=VERDICT_LABELS[verdict],
        pair_key=pair_key,
        detector_type=detector_type,
        prior_verdict=prior_verdict,
    )
    progress = templates.get_template("cc__progress_count.html").render(
        detector_type=detector_type,
        reviewed_count=reviewed_count,
        total_count=total_count,
    )
    # Row removal: empty body returned for hx-swap="outerHTML" on the row's hx-target.
    return HTMLResponse(content=toast + progress)


def _count_total_findings(store: AssertionStore, detector_type: str) -> int:
    """Total findings of the given detector type in the most-recent run."""
    if detector_type == "multi_party":
        row = store._conn.execute(
            "SELECT COUNT(*) FROM multi_party_findings WHERE judge_verdict = 'multi_party_contradiction'"
        ).fetchone()
        return int(row[0])
    row = store._conn.execute(
        "SELECT COUNT(*) FROM findings WHERE detector_type = ?",
        (detector_type,),
    ).fetchone()
    return int(row[0])
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python3 -m pytest tests/test_web_verdicts.py -v
python3 -m ruff check consistency_checker/web/app.py
```

Expected: 2 passed, lint clean.

- [ ] **Step 6: Commit**

```bash
git add consistency_checker/web/app.py consistency_checker/web/templates/cc__verdict_toast.html consistency_checker/web/templates/cc__progress_count.html tests/test_web_verdicts.py
git commit -m "feat(web): POST /verdicts endpoint + toast/progress OOB partials"
```

---

### Task 6: `POST /verdicts/undo` endpoint

**Files:**
- Modify: `consistency_checker/web/app.py`
- Test: `tests/test_web_verdicts.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_web_verdicts.py`:

```python
def test_post_verdicts_undo_first_click_case_deletes(
    app_client: tuple[TestClient, Config],
) -> None:
    """Undo with empty prior_verdict deletes the row (first-click case)."""
    client, cfg = app_client
    # Seed a verdict
    client.post(
        "/verdicts",
        data={
            "pair_key": "a:b", "detector_type": "contradiction",
            "verdict": "confirmed", "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    # Undo it
    resp = client.post(
        "/verdicts/undo",
        data={
            "pair_key": "a:b", "detector_type": "contradiction", "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    store = AssertionStore(cfg.db_path)
    rows = store._conn.execute("SELECT COUNT(*) FROM reviewer_verdicts").fetchone()
    assert rows[0] == 0
    store.close()


def test_post_verdicts_undo_rejudge_case_restores_prior(
    app_client: tuple[TestClient, Config],
) -> None:
    """Undo with non-empty prior_verdict re-sets to the prior value."""
    client, cfg = app_client
    # Start in confirmed
    client.post(
        "/verdicts",
        data={
            "pair_key": "a:b", "detector_type": "contradiction",
            "verdict": "confirmed", "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    # Re-judge to false_positive
    client.post(
        "/verdicts",
        data={
            "pair_key": "a:b", "detector_type": "contradiction",
            "verdict": "false_positive", "prior_verdict": "confirmed",
        },
        headers={"HX-Request": "true"},
    )
    # Undo: should restore confirmed
    resp = client.post(
        "/verdicts/undo",
        data={
            "pair_key": "a:b", "detector_type": "contradiction",
            "prior_verdict": "confirmed",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    store = AssertionStore(cfg.db_path)
    row = store._conn.execute(
        "SELECT verdict FROM reviewer_verdicts WHERE pair_key = ?", ("a:b",)
    ).fetchone()
    assert row["verdict"] == "confirmed"
    store.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python3 -m pytest tests/test_web_verdicts.py -v -k undo
```

Expected: FAIL.

- [ ] **Step 3: Add the undo route**

In `consistency_checker/web/app.py`, alongside `post_verdict`:

```python
@app.post("/verdicts/undo", response_class=HTMLResponse)
def post_verdict_undo(
    request: Request,
    pair_key: str = Form(...),
    detector_type: str = Form(...),
    prior_verdict: str = Form(""),
) -> HTMLResponse:
    if detector_type not in {"contradiction", "definition_inconsistency", "multi_party"}:
        raise HTTPException(status_code=400, detail=f"unknown detector_type {detector_type!r}")
    store, audit = _open_audit()
    try:
        if prior_verdict == "":
            audit.delete_reviewer_verdict(
                pair_key=pair_key, detector_type=detector_type,  # type: ignore[arg-type]
            )
        else:
            if prior_verdict not in {"confirmed", "false_positive", "dismissed"}:
                raise HTTPException(
                    status_code=400, detail=f"unknown prior_verdict {prior_verdict!r}"
                )
            audit.set_reviewer_verdict(
                pair_key=pair_key,
                detector_type=detector_type,  # type: ignore[arg-type]
                verdict=prior_verdict,  # type: ignore[arg-type]
            )
    finally:
        store.close()
    # Caller has hx-target="#cc-tab-content" hx-swap="innerHTML" — tab re-renders
    # itself from a fresh GET. We return an empty body and signal the redirect via
    # an HX-Trigger response so the existing tab's polling/swap logic re-fetches.
    # Simpler: return empty 204-equivalent; the client's hx-target is the tab body
    # and the response will overwrite it with empty, but the user expects the
    # full tab. So we return an HX-Redirect to the current tab path.
    referer = request.headers.get("HX-Current-URL") or request.headers.get("Referer", "/")
    response = HTMLResponse(content="")
    response.headers["HX-Redirect"] = referer
    return response
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python3 -m pytest tests/test_web_verdicts.py -v
python3 -m ruff check consistency_checker/web/app.py
```

Expected: 4 passed, lint clean.

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/web/app.py tests/test_web_verdicts.py
git commit -m "feat(web): POST /verdicts/undo with delete-or-restore branching"
```

---

## Phase 3 — Templates + base scaffolding

### Task 7: Verdict-buttons partial

**Files:**
- Create: `consistency_checker/web/templates/cc__verdict_buttons.html`
- Test: covered by tab-integration tests in Tasks 10-11; no isolated test.

- [ ] **Step 1: Create the partial**

```html
{# Renders three icon-only verdict buttons for a single finding row.
   Required context: pair_key, detector_type, prior_verdict (possibly empty). #}
<div class="cc-verdict-buttons" role="group" aria-label="Reviewer actions">
  <button class="cc-verdict-btn cc-verdict-btn--confirmed"
          type="button"
          aria-label="Mark as real issue"
          title="Real issue — model flagged this correctly (C)"
          data-verdict="confirmed"
          hx-post="/verdicts"
          hx-vals='{"pair_key": "{{ pair_key }}",
                    "detector_type": "{{ detector_type }}",
                    "verdict": "confirmed",
                    "prior_verdict": "{{ prior_verdict|default('') }}"}'
          hx-target="closest tr"
          hx-swap="outerHTML">✓</button>
  <button class="cc-verdict-btn cc-verdict-btn--false-positive"
          type="button"
          aria-label="Mark as not an issue"
          title="Not an issue — model flagged this incorrectly (F)"
          data-verdict="false_positive"
          hx-post="/verdicts"
          hx-vals='{"pair_key": "{{ pair_key }}",
                    "detector_type": "{{ detector_type }}",
                    "verdict": "false_positive",
                    "prior_verdict": "{{ prior_verdict|default('') }}"}'
          hx-target="closest tr"
          hx-swap="outerHTML">✗</button>
  <button class="cc-verdict-btn cc-verdict-btn--dismissed"
          type="button"
          aria-label="Skip for now"
          title="Skip — real but not yours to resolve (D)"
          data-verdict="dismissed"
          hx-post="/verdicts"
          hx-vals='{"pair_key": "{{ pair_key }}",
                    "detector_type": "{{ detector_type }}",
                    "verdict": "dismissed",
                    "prior_verdict": "{{ prior_verdict|default('') }}"}'
          hx-target="closest tr"
          hx-swap="outerHTML">—</button>
</div>
```

- [ ] **Step 2: Commit**

```bash
git add consistency_checker/web/templates/cc__verdict_buttons.html
git commit -m "feat(web): cc__verdict_buttons.html — inline verdict actions"
```

---

### Task 8: Base template — toast region + keyboard JS + focus management

**Files:**
- Modify: `consistency_checker/web/templates/cc_base.html`
- Test: `tests/test_web_verdicts.py` (extend with one snapshot-style assertion)

- [ ] **Step 1: Modify `cc_base.html`**

Add the toast region inside `<body>` (before the `<dialog>`):

```html
<div id="cc-toast-region" class="cc-toast-region" role="status" aria-live="polite"></div>
```

The progress count regions get rendered inside each tab template, so no global element needed.

Add a small `<script>` block at the end of `<body>` (after the `<dialog>` element):

```html
<script>
  // Keyboard shortcuts on focused finding rows.
  document.addEventListener('keydown', (e) => {
    const row = document.activeElement.closest('tr.cc-finding-row');
    if (!row) return;
    const verdictMap = { 'c': 'confirmed', 'f': 'false_positive', 'd': 'dismissed' };
    const verdict = verdictMap[e.key.toLowerCase()];
    if (!verdict) return;
    const btn = row.querySelector('[data-verdict="' + verdict + '"]');
    if (btn) { e.preventDefault(); btn.click(); }
  });

  // After a row is removed by an HTMX swap, move focus to the next
  // finding row in the same table (not to the top of the page).
  document.body.addEventListener('htmx:afterSwap', (e) => {
    if (!e.detail.target || !e.detail.target.classList) return;
    const tbody = e.detail.target.closest('tbody');
    if (!tbody) return;
    const nextRow = tbody.querySelector('tr.cc-finding-row');
    if (nextRow) nextRow.focus();
  });
</script>
```

- [ ] **Step 2: Verify the base template still renders**

```bash
python3 -m pytest tests/test_web_verdicts.py -v
python3 -m pytest tests/test_web_runs.py tests/test_web_stats.py --tb=no -q
```

Expected: all existing web tests still pass (no template regressions).

- [ ] **Step 3: Commit**

```bash
git add consistency_checker/web/templates/cc_base.html
git commit -m "feat(web): toast region + keyboard shortcuts + focus management"
```

---

## Phase 4 — Tab integration

### Task 9: Contradictions tab — pair-contradictions section integration

**Files:**
- Modify: `consistency_checker/web/app.py` (the `index` route)
- Modify: `consistency_checker/web/templates/cc_contradictions.html`
- Test: `tests/test_web_verdicts.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_web_verdicts.py`:

```python
def _seed_pair_contradiction(cfg: Config) -> tuple[str, str]:
    """Helper: ingest one pair-contradiction finding into the store. Returns (a_id, b_id)."""
    from consistency_checker.audit.logger import AuditLogger
    from consistency_checker.check.gate import CandidatePair
    from consistency_checker.check.llm_judge import JudgeVerdict
    from consistency_checker.check.nli_checker import NliResult
    from consistency_checker.extract.schema import Assertion, Document
    store = AssertionStore(cfg.db_path)
    store.migrate()
    doc_a = Document.from_content("A body.", source_path="a.md", title="Doc A")
    doc_b = Document.from_content("B body.", source_path="b.md", title="Doc B")
    store.add_document(doc_a)
    store.add_document(doc_b)
    a = Assertion.build(doc_a.doc_id, "Revenue grew 12%.")
    b = Assertion.build(doc_b.doc_id, "Revenue declined 5%.")
    store.add_assertions([a, b])
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    logger.record_finding(
        run_id,
        candidate=CandidatePair(a=a, b=b, score=0.9),
        nli=NliResult.from_scores(p_contradiction=0.85, p_entailment=0.05, p_neutral=0.10),
        verdict=JudgeVerdict(
            assertion_a_id=a.assertion_id,
            assertion_b_id=b.assertion_id,
            verdict="contradiction",
            confidence=0.9,
            rationale="opposite revenue signs",
            evidence_spans=[],
        ),
    )
    logger.end_run(run_id, n_assertions=2, n_pairs_gated=1, n_pairs_judged=1)
    store.close()
    return a.assertion_id, b.assertion_id


def test_contradictions_tab_hides_reviewed_by_default(
    app_client: tuple[TestClient, Config],
) -> None:
    client, cfg = app_client
    a_id, b_id = _seed_pair_contradiction(cfg)
    pair_key = ":".join(sorted([a_id, b_id]))
    # Mark as confirmed
    client.post(
        "/verdicts",
        data={
            "pair_key": pair_key, "detector_type": "contradiction",
            "verdict": "confirmed", "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    # Default tab GET — row should be absent
    resp = client.get("/")
    assert resp.status_code == 200
    assert pair_key not in resp.text


def test_contradictions_tab_shows_reviewed_when_toggle_on(
    app_client: tuple[TestClient, Config],
) -> None:
    client, cfg = app_client
    a_id, b_id = _seed_pair_contradiction(cfg)
    pair_key = ":".join(sorted([a_id, b_id]))
    client.post(
        "/verdicts",
        data={
            "pair_key": pair_key, "detector_type": "contradiction",
            "verdict": "confirmed", "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    # With ?show_reviewed_contradiction=1
    resp = client.get("/?show_reviewed_contradiction=1")
    assert resp.status_code == 200
    assert pair_key in resp.text
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python3 -m pytest tests/test_web_verdicts.py -v -k contradictions_tab
```

Expected: FAIL (default tab GET doesn't filter; toggle doesn't exist).

- [ ] **Step 3: Update the `index` route in `web/app.py`**

Find the `index` function. Add a query parameter and use it to filter:

```python
@app.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    show_reviewed_contradiction: bool = False,
    show_reviewed_multi_party: bool = False,
) -> Response:
    """Contradictions tab — the main page (ADR-0007)."""
    store, audit = _open_audit()
    try:
        if store.stats()["documents"] == 0:
            store.close()
            if _is_htmx(request):
                resp: Response = Response(status_code=200)
                resp.headers["HX-Redirect"] = "/tabs/ingest"
                return resp
            return Response(status_code=303, headers={"Location": "/tabs/ingest"})
        run = audit.most_recent_run()
        pair_findings: list[dict[str, Any]] = []
        multi_party_findings: list[dict[str, Any]] = []
        if run is not None:
            raw_pair_findings = [
                *audit.iter_findings(run_id=run.run_id, verdict="contradiction"),
                *audit.iter_findings(run_id=run.run_id, verdict="numeric_short_circuit"),
            ]
            # Build pair_key lookup table
            keys = [
                (":".join(sorted([r.assertion_a_id, r.assertion_b_id])), "contradiction")
                for r in raw_pair_findings
            ]
            verdicts_by_pk = audit.get_reviewer_verdicts_bulk(keys)
            assertion_ids = [
                aid for raw in raw_pair_findings
                for aid in (raw.assertion_a_id, raw.assertion_b_id)
            ]
            assertions = store.get_assertions_bulk(assertion_ids)
            doc_ids = list({a.doc_id for a in assertions.values()})
            documents = store.get_documents_bulk(doc_ids)
            for raw in raw_pair_findings:
                a = assertions.get(raw.assertion_a_id)
                b = assertions.get(raw.assertion_b_id)
                if a is None or b is None:
                    continue
                pk = ":".join(sorted([raw.assertion_a_id, raw.assertion_b_id]))
                rv = verdicts_by_pk.get((pk, "contradiction"))
                # Filter out reviewed rows when toggle off
                if rv is not None and not show_reviewed_contradiction:
                    continue
                pair_findings.append({
                    "finding_id": raw.finding_id,
                    "pair_key": pk,
                    "verdict": raw.judge_verdict,
                    "confidence": raw.judge_confidence,
                    "doc_a_label": _document_label(documents.get(a.doc_id), a.doc_id),
                    "doc_b_label": _document_label(documents.get(b.doc_id), b.doc_id),
                    "rationale_first_line": (raw.judge_rationale or "").splitlines()[:1],
                    "reviewer_verdict": rv.verdict if rv else None,
                    "reviewer_label": VERDICT_LABELS[rv.verdict] if rv else None,
                })
            pair_findings.sort(key=lambda f: -(f["confidence"] or 0.0))

            # Multi-party section: same pattern
            raw_mp = list(audit.iter_multi_party_findings(
                run_id=run.run_id, verdict="multi_party_contradiction"
            ))
            mp_keys = [
                (":".join(sorted(mp.assertion_ids)), "multi_party")
                for mp in raw_mp
            ]
            mp_verdicts_by_pk = audit.get_reviewer_verdicts_bulk(mp_keys)
            for mp in raw_mp:
                pk = ":".join(sorted(mp.assertion_ids))
                rv = mp_verdicts_by_pk.get((pk, "multi_party"))
                if rv is not None and not show_reviewed_multi_party:
                    continue
                multi_party_findings.append({
                    "finding_id": mp.finding_id,
                    "pair_key": pk,
                    "confidence": mp.judge_confidence,
                    "rationale_first_line": ((mp.judge_rationale or "").splitlines()[:1]),
                    "n_docs": len({d for d in mp.doc_ids}),
                    "reviewer_verdict": rv.verdict if rv else None,
                    "reviewer_label": VERDICT_LABELS[rv.verdict] if rv else None,
                })
            multi_party_findings.sort(key=lambda f: -(f["confidence"] or 0.0))

        # Progress counts (full audit-DB totals, independent of toggle state)
        pair_total = _count_total_findings(store, "contradiction")
        pair_reviewed = sum(audit.count_reviewer_verdicts(detector_type="contradiction").values())
        mp_total = _count_total_findings(store, "multi_party")
        mp_reviewed = sum(audit.count_reviewer_verdicts(detector_type="multi_party").values())
    finally:
        store.close()

    return templates.TemplateResponse(
        request,
        "cc_contradictions.html",
        {
            "htmx": _is_htmx(request),
            "active_tab": "contradictions",
            "run": {
                "run_id": run.run_id,
                "n_assertions": run.n_assertions,
                "n_pairs_judged": run.n_pairs_judged,
            } if run is not None else None,
            "pair_findings": pair_findings,
            "multi_party_findings": multi_party_findings,
            "show_reviewed_contradiction": show_reviewed_contradiction,
            "show_reviewed_multi_party": show_reviewed_multi_party,
            "pair_reviewed": pair_reviewed,
            "pair_total": pair_total,
            "mp_reviewed": mp_reviewed,
            "mp_total": mp_total,
        },
    )
```

- [ ] **Step 4: Update `cc_contradictions.html`**

Replace the `<h3>Statement contradictions</h3>` section and its table with this:

```html
<h3>Statement contradictions</h3>
<div class="cc-section-state">
  <span id="cc-progress-count-contradiction" class="cc-progress-count">
    {{ pair_reviewed }} of {{ pair_total }} reviewed
  </span>
  <label class="cc-toggle">
    <input type="checkbox"
           name="show_reviewed_contradiction"
           value="1"
           {% if show_reviewed_contradiction %}checked{% endif %}
           hx-get="/"
           hx-include="this"
           hx-target="#cc-tab-content"
           hx-push-url="true">
    Show reviewed
  </label>
</div>
{% if pair_findings %}
  <table class="cc-findings-table">
    <thead>
      <tr>
        <th>Confidence</th>
        <th>Verdict</th>
        <th>Doc A</th>
        <th>Doc B</th>
        <th>Rationale</th>
        <th>Diff</th>
        <th>Mark</th>
      </tr>
    </thead>
    <tbody>
      {% for f in pair_findings %}
        <tr class="cc-finding-row" tabindex="0">
          <td>{{ "%.2f"|format(f.confidence or 0.0) }}</td>
          <td><code>{{ f.verdict }}</code></td>
          <td>{{ f.doc_a_label }}</td>
          <td>{{ f.doc_b_label }}</td>
          <td>{{ f.rationale_first_line | join(' ') }}</td>
          <td>
            <button type="button" class="cc-button"
                    hx-get="/findings/{{ f.finding_id }}/diff"
                    hx-target="#cc-diff-content"
                    hx-on::after-request="document.getElementById('cc-diff-dialog').showModal()">Diff</button>
          </td>
          <td>
            {% if f.reviewer_verdict %}
              <span class="cc-verdict-badge cc-verdict-badge--{{ f.reviewer_verdict }}">{{ f.reviewer_label }}</span>
            {% else %}
              {% include "cc__verdict_buttons.html" with context %}
            {% endif %}
          </td>
        </tr>
      {% endfor %}
    </tbody>
  </table>
{% else %}
  <p class="cc-muted">All findings reviewed.</p>
{% endif %}
```

For the include to find `pair_key` and `detector_type` in the row context, set them on the dict you pass:

In the route, each `pair_findings` row already has `pair_key`. Add a context-passing wrapper in the template — Jinja's `{% include ... with context %}` already shares the row scope. Inside the partial, `{{ pair_key }}` resolves to `f.pair_key` because Jinja exports the loop variable's attributes if they're explicitly set via `set`. Use a small `set` block in the row:

```html
{% set pair_key = f.pair_key %}
{% set detector_type = "contradiction" %}
{% set prior_verdict = "" %}
{% include "cc__verdict_buttons.html" %}
```

Replace the include-line with this 4-line set+include block.

Do the same shape for the multi-party section (Task 11).

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python3 -m pytest tests/test_web_verdicts.py -v
python3 -m pytest tests/test_web_runs.py tests/test_web_stats.py --tb=no -q
```

Expected: all green, no regressions.

- [ ] **Step 6: Commit**

```bash
git add consistency_checker/web/app.py consistency_checker/web/templates/cc_contradictions.html tests/test_web_verdicts.py
git commit -m "feat(web): Contradictions tab — verdict buttons, hide-by-default, progress count"
```

---

### Task 10: Definitions tab integration

**Files:**
- Modify: `consistency_checker/web/app.py` (the `tab_definitions` route)
- Modify: `consistency_checker/web/templates/cc_definitions.html`
- Test: `tests/test_web_verdicts.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_web_verdicts.py`:

```python
def _seed_definition_finding(cfg: Config) -> tuple[str, str]:
    """Helper: ingest one definition-inconsistency finding."""
    from consistency_checker.audit.logger import AuditLogger
    from consistency_checker.check.definition_checker import DefinitionFinding, DefinitionPair
    from consistency_checker.check.definition_judge import DefinitionJudgeVerdict
    from consistency_checker.extract.schema import Assertion, Document

    store = AssertionStore(cfg.db_path)
    store.migrate()
    doc_a = Document.from_content("A.", source_path="a.md", title="Doc A")
    doc_b = Document.from_content("B.", source_path="b.md", title="Doc B")
    store.add_document(doc_a)
    store.add_document(doc_b)
    a = Assertion.build(doc_a.doc_id, '"MAE" means A.', kind="definition", term="MAE", definition_text="A")
    b = Assertion.build(doc_b.doc_id, '"MAE" means B.', kind="definition", term="MAE", definition_text="B")
    store.add_assertions([a, b])
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    logger.record_definition_finding(
        run_id,
        finding=DefinitionFinding(
            pair=DefinitionPair(a=a, b=b, canonical_term="mae"),
            verdict=DefinitionJudgeVerdict(
                assertion_a_id=min(a.assertion_id, b.assertion_id),
                assertion_b_id=max(a.assertion_id, b.assertion_id),
                verdict="definition_divergent",
                confidence=0.9, rationale="scope shift", evidence_spans=[],
            ),
        ),
    )
    logger.end_run(run_id, n_assertions=2, n_pairs_gated=0, n_pairs_judged=0)
    store.close()
    return a.assertion_id, b.assertion_id


def test_definitions_tab_hides_reviewed_by_default(
    app_client: tuple[TestClient, Config],
) -> None:
    client, cfg = app_client
    a_id, b_id = _seed_definition_finding(cfg)
    pair_key = ":".join(sorted([a_id, b_id]))
    client.post(
        "/verdicts",
        data={
            "pair_key": pair_key, "detector_type": "definition_inconsistency",
            "verdict": "confirmed", "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    resp = client.get("/tabs/definitions")
    assert "MAE" not in resp.text or "All findings reviewed" in resp.text
    resp_all = client.get("/tabs/definitions?show_reviewed_definition_inconsistency=1")
    assert "MAE" in resp_all.text
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python3 -m pytest tests/test_web_verdicts.py -v -k definitions_tab
```

Expected: FAIL.

- [ ] **Step 3: Update the `tab_definitions` route**

In `consistency_checker/web/app.py`, modify `tab_definitions` mirroring the contradictions pattern:

```python
@app.get("/tabs/definitions", response_class=HTMLResponse)
def tab_definitions(
    request: Request,
    show_reviewed_definition_inconsistency: bool = False,
) -> HTMLResponse:
    """Definition-inconsistencies tab (ADR-0009)."""
    store, audit = _open_audit()
    try:
        run = audit.most_recent_run()
        findings: list[dict[str, Any]] = []
        if run is not None:
            raw = list(audit.iter_findings(
                run_id=run.run_id,
                verdict="definition_divergent",
                detector_type="definition_inconsistency",
            ))
            keys = [
                (":".join(sorted([r.assertion_a_id, r.assertion_b_id])),
                 "definition_inconsistency")
                for r in raw
            ]
            verdicts_by_pk = audit.get_reviewer_verdicts_bulk(keys)
            assertion_ids = [aid for r in raw for aid in (r.assertion_a_id, r.assertion_b_id)]
            assertions = store.get_assertions_bulk(assertion_ids)
            doc_ids = list({a.doc_id for a in assertions.values()})
            documents = store.get_documents_bulk(doc_ids)
            for r in raw:
                a = assertions.get(r.assertion_a_id)
                b = assertions.get(r.assertion_b_id)
                if a is None or b is None:
                    continue
                pk = ":".join(sorted([r.assertion_a_id, r.assertion_b_id]))
                rv = verdicts_by_pk.get((pk, "definition_inconsistency"))
                if rv is not None and not show_reviewed_definition_inconsistency:
                    continue
                findings.append({
                    "finding_id": r.finding_id,
                    "pair_key": pk,
                    "term": a.term or "",
                    "confidence": r.judge_confidence,
                    "doc_a_label": _document_label(documents.get(a.doc_id), a.doc_id),
                    "doc_b_label": _document_label(documents.get(b.doc_id), b.doc_id),
                    "def_a_text": a.assertion_text,
                    "def_b_text": b.assertion_text,
                    "rationale": r.judge_rationale or "",
                    "reviewer_verdict": rv.verdict if rv else None,
                    "reviewer_label": VERDICT_LABELS[rv.verdict] if rv else None,
                })
            findings.sort(key=lambda f: (f["term"].lower(), -(f["confidence"] or 0.0)))

        total = _count_total_findings(store, "definition_inconsistency")
        reviewed = sum(audit.count_reviewer_verdicts(
            detector_type="definition_inconsistency"
        ).values())
    finally:
        store.close()

    return templates.TemplateResponse(
        request,
        "cc_definitions.html",
        {
            "htmx": _is_htmx(request),
            "active_tab": "definitions",
            "run": {"run_id": run.run_id} if run is not None else None,
            "findings": findings,
            "show_reviewed": show_reviewed_definition_inconsistency,
            "reviewed_count": reviewed,
            "total_count": total,
        },
    )
```

- [ ] **Step 4: Update `cc_definitions.html`**

Update the template to add the section-state strip and verdict buttons in each row. The exact same pattern as Contradictions:

```html
<div class="cc-section-state">
  <span id="cc-progress-count-definition_inconsistency" class="cc-progress-count">
    {{ reviewed_count }} of {{ total_count }} reviewed
  </span>
  <label class="cc-toggle">
    <input type="checkbox"
           name="show_reviewed_definition_inconsistency"
           value="1"
           {% if show_reviewed %}checked{% endif %}
           hx-get="/tabs/definitions"
           hx-include="this"
           hx-target="#cc-tab-content"
           hx-push-url="true">
    Show reviewed
  </label>
</div>
```

And in the row template, add a `Mark` column that either shows the badge (if reviewed) or the buttons partial (if not), with the `set` + `include` pattern from Task 9.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python3 -m pytest tests/test_web_verdicts.py tests/test_web_definitions.py -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add consistency_checker/web/app.py consistency_checker/web/templates/cc_definitions.html tests/test_web_verdicts.py
git commit -m "feat(web): Definitions tab — verdict buttons, hide-by-default, progress count"
```

---

## Phase 5 — Report integration

### Task 11: Markdown report — filter `false_positive` + add Reviewer column/tag

**Files:**
- Modify: `consistency_checker/audit/report.py`
- Test: `tests/test_report.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_report.py`:

```python
def test_report_excludes_false_positive_findings(seeded_store: AssertionStore) -> None:
    _populate_three_doc_fixture(seeded_store)
    logger = AuditLogger(seeded_store)
    run_id = logger.begin_run()
    _record_three_findings(seeded_store, logger, run_id)
    logger.end_run(run_id, n_assertions=4, n_pairs_gated=3, n_pairs_judged=3)
    # Mark the high-confidence revenue contradiction as false_positive
    a_list = list(seeded_store.iter_assertions())
    by_text = {a.assertion_text: a for a in a_list}
    a1 = by_text["Revenue grew 12% in fiscal 2025."]
    b1 = by_text["Revenue declined 5% in fiscal 2025."]
    pair_key = ":".join(sorted([a1.assertion_id, b1.assertion_id]))
    logger.set_reviewer_verdict(
        pair_key=pair_key, detector_type="contradiction", verdict="false_positive"
    )
    out = render_report(seeded_store, logger, run_id=run_id)
    # The false_positive finding's rationale should not appear
    assert "Opposite revenue signs" not in out
    # But the other contradiction should still appear
    assert "Different start years" in out


def test_report_renders_reviewer_tag_for_confirmed(seeded_store: AssertionStore) -> None:
    _populate_three_doc_fixture(seeded_store)
    logger = AuditLogger(seeded_store)
    run_id = logger.begin_run()
    _record_three_findings(seeded_store, logger, run_id)
    logger.end_run(run_id, n_assertions=4, n_pairs_gated=3, n_pairs_judged=3)
    a_list = list(seeded_store.iter_assertions())
    by_text = {a.assertion_text: a for a in a_list}
    a1 = by_text["Revenue grew 12% in fiscal 2025."]
    b1 = by_text["Revenue declined 5% in fiscal 2025."]
    pair_key = ":".join(sorted([a1.assertion_id, b1.assertion_id]))
    logger.set_reviewer_verdict(
        pair_key=pair_key, detector_type="contradiction", verdict="confirmed"
    )
    out = render_report(seeded_store, logger, run_id=run_id)
    assert "**Reviewer:** Real issue" in out


def test_report_renders_reviewer_tag_for_dismissed(seeded_store: AssertionStore) -> None:
    _populate_three_doc_fixture(seeded_store)
    logger = AuditLogger(seeded_store)
    run_id = logger.begin_run()
    _record_three_findings(seeded_store, logger, run_id)
    logger.end_run(run_id, n_assertions=4, n_pairs_gated=3, n_pairs_judged=3)
    a_list = list(seeded_store.iter_assertions())
    by_text = {a.assertion_text: a for a in a_list}
    a1 = by_text["Revenue grew 12% in fiscal 2025."]
    b1 = by_text["Revenue declined 5% in fiscal 2025."]
    pair_key = ":".join(sorted([a1.assertion_id, b1.assertion_id]))
    logger.set_reviewer_verdict(
        pair_key=pair_key, detector_type="contradiction", verdict="dismissed"
    )
    out = render_report(seeded_store, logger, run_id=run_id)
    assert "**Reviewer:** Dismissed" in out


def test_report_renders_reviewer_tag_for_unreviewed(seeded_store: AssertionStore) -> None:
    _populate_three_doc_fixture(seeded_store)
    logger = AuditLogger(seeded_store)
    run_id = logger.begin_run()
    _record_three_findings(seeded_store, logger, run_id)
    logger.end_run(run_id, n_assertions=4, n_pairs_gated=3, n_pairs_judged=3)
    out = render_report(seeded_store, logger, run_id=run_id)
    assert "**Reviewer:** Pending review" in out
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python3 -m pytest tests/test_report.py -v -k "reviewer or false_positive"
```

Expected: FAIL.

- [ ] **Step 3: Update `report.py`**

In `consistency_checker/audit/report.py`, add the label map:

```python
VERDICT_LABELS: dict[str, str] = {
    "confirmed": "Real issue",
    "false_positive": "Not an issue",
    "dismissed": "Dismissed",
}
```

In the contradictions section, after building `contradictions`, filter false-positives:

```python
# Pull reviewer verdicts for all contradiction findings in one query.
contradiction_keys = [
    (":".join(sorted([f.assertion_a_id, f.assertion_b_id])), "contradiction")
    for f in contradictions
]
verdicts_by_pk = audit_logger.get_reviewer_verdicts_bulk(contradiction_keys)
# Filter false_positive
contradictions = [
    f for f in contradictions
    if (verdicts_by_pk.get(
        (":".join(sorted([f.assertion_a_id, f.assertion_b_id])), "contradiction")
    ) is None
       or verdicts_by_pk[
            (":".join(sorted([f.assertion_a_id, f.assertion_b_id])), "contradiction")
       ].verdict != "false_positive")
]
```

In the per-finding rendering loop, add the reviewer tag line:

```python
pk = ":".join(sorted([finding.assertion_a_id, finding.assertion_b_id]))
rv = verdicts_by_pk.get((pk, "contradiction"))
reviewer_label = (
    VERDICT_LABELS[rv.verdict] if rv else "Pending review"
)
lines.append(f"**Reviewer:** {reviewer_label}")
lines.append("")
```

Apply the same filter + tag in `_append_multi_party_section` and `_append_definition_section`.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python3 -m pytest tests/test_report.py -v
python3 -m ruff check consistency_checker/audit/report.py
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/audit/report.py tests/test_report.py
git commit -m "feat(report): filter false_positive findings; per-finding reviewer tag"
```

---

## Phase 6 — CSS + polish

### Task 12: CSS for verdict buttons, badges, toast, hidden state

**Files:**
- Modify: `consistency_checker/web/static/cc_style.css`

- [ ] **Step 1: Add CSS rules**

Append to `consistency_checker/web/static/cc_style.css`:

```css
/* --- Reviewer-verdict surface ------------------------------------------- */

.cc-section-state {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 0.5em 0;
  font-size: 0.9em;
}

.cc-progress-count {
  color: #555;
}

.cc-toggle {
  user-select: none;
}

.cc-toggle input[type="checkbox"] {
  margin-right: 0.3em;
}

.cc-verdict-buttons {
  display: flex;
  gap: 0.3em;
}

.cc-verdict-btn {
  width: 1.8em;
  height: 1.8em;
  line-height: 1;
  font-size: 1em;
  padding: 0;
  border: 1px solid #aaa;
  background: #f8f8f8;
  border-radius: 3px;
  cursor: pointer;
}

.cc-verdict-btn:hover {
  background: #e8e8e8;
}

.cc-verdict-btn--confirmed { color: #1a7a1a; }
.cc-verdict-btn--false-positive { color: #b03030; }
.cc-verdict-btn--dismissed { color: #555; }

.cc-verdict-badge {
  display: inline-block;
  padding: 0.15em 0.5em;
  border-radius: 3px;
  font-size: 0.85em;
  white-space: nowrap;
}

.cc-verdict-badge--confirmed { background: #e0f0e0; color: #1a5a1a; }
.cc-verdict-badge--false_positive { background: #f0e0e0; color: #803030; }
.cc-verdict-badge--dismissed { background: #eee; color: #555; }

.cc-toast-region {
  position: fixed;
  bottom: 1em;
  right: 1em;
  z-index: 1000;
}

.cc-toast {
  background: #333;
  color: white;
  padding: 0.6em 1em;
  border-radius: 4px;
  display: flex;
  align-items: center;
  gap: 0.8em;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

.cc-toast-undo {
  background: transparent;
  color: #8cf;
  border: none;
  cursor: pointer;
  text-decoration: underline;
  padding: 0;
}

.cc-toast-close {
  background: transparent;
  color: white;
  border: none;
  cursor: pointer;
  font-size: 1.2em;
  padding: 0 0.2em;
}

tr.cc-finding-row:focus {
  outline: 2px solid #2a6;
  outline-offset: -2px;
}
```

- [ ] **Step 2: Visual smoke-check**

If you can run `consistency-check serve --open` locally, do — confirm:
- Buttons render as small icon squares.
- Toast appears bottom-right on click, persists, dismissible via `×`.
- Focused row has a green outline.
- Verdict badges have distinct colors in the "Show reviewed" view.

If CSS-only changes aren't testable in your env, skip the visual check; the templates already pass their tests, and CSS doesn't have hermetic test coverage in this project.

- [ ] **Step 3: Commit**

```bash
git add consistency_checker/web/static/cc_style.css
git commit -m "feat(web): CSS for verdict buttons, badges, toast, focused-row state"
```

---

## Phase 7 — Wrap-up

### Task 13: futureplans.md update + PR

**Files:**
- Modify: `futureplans.md`

- [ ] **Step 1: Move item #9 to Completed**

In `futureplans.md`, remove or shorten the item #9 entry under "v0.4 — precision and provenance" and add to the Completed section:

```markdown
- **v0.4 (reviewer workflow, Phase A)** — item #9: inline verdict buttons on
  Contradictions / Definitions / Cross-document tabs; content-keyed
  `reviewer_verdicts` table that survives re-runs; hide-by-default with
  "Show reviewed" toggle; persistent undo toast; markdown report filters
  `false_positive` and adds a Reviewer column/tag. Phase B (dedicated
  review-queue page) and Phase A.1 (note column UI, findings CSV with
  `reviewer_verdict`) tracked under item #9b.
```

Also add an item #9b for the deferred Phase B + small follow-ups:

```markdown
### 9b. Reviewer workflow — Phase B (dedicated queue + small extensions)
Parked from the v0.4 Phase A build. Two distinct pieces:

- **Dedicated "Review" tab** — focused per-finding queue with big buttons,
  skip/back navigation, optional batch-mode keyboard flow. The schema and
  setter API land in Phase A; this is a UI surface that uses them.
- **Note column UI** — column already exists in `reviewer_verdicts.note`,
  no v1 UI. Queue page is the natural place to surface it.
- **Findings CSV export** with `reviewer_verdict` column for downstream
  tooling.
```

- [ ] **Step 2: Run the full hermetic suite**

```bash
python3 -m pytest -m "not slow and not live" --tb=no
python3 -m ruff check consistency_checker/ tests/
python3 -m ruff format --check consistency_checker/ tests/
```

Expected: all green (pre-existing env failures unrelated to this work are OK).

- [ ] **Step 3: Commit**

```bash
git add futureplans.md
git commit -m "docs: futureplans — item #9 (reviewer workflow Phase A) shipped; #9b parked"
```

- [ ] **Step 4: Push + PR**

```bash
git push -u origin <feature-branch>
gh pr create --title "Reviewer workflow (item #9, Phase A)" --body "$(cat <<'EOF'
## Summary

Adds inline reviewer verdicts to the web UI so the user can mark each finding
as Real issue (`confirmed`), Not an issue (`false_positive`), or Skip
(`dismissed`). Verdicts persist across re-runs of the same corpus
(content-keyed, not run-scoped). Markdown report filters false-positives and
tags surviving findings.

Spec: docs/superpowers/specs/2026-05-15-reviewer-workflow-design.md
Plan: docs/superpowers/plans/2026-05-15-reviewer-workflow.md

## What landed

- Migration 0009 — `reviewer_verdicts` table with composite PK
  `(pair_key, detector_type)` and CHECK constraints on both enum columns.
- `consistency_checker/audit/reviewer.py` — `ReviewerVerdict` dataclass,
  Literal type aliases, `build_pair_key` helper.
- `AuditLogger.{set,delete,get_bulk,count}_reviewer_verdict` — four CRUD
  methods.
- New `POST /verdicts` and `POST /verdicts/undo` routes; first-click undo
  deletes, re-judge undo restores prior verdict.
- New partials: `cc__verdict_buttons.html`, `cc__verdict_toast.html`,
  `cc__progress_count.html`.
- Base template gains the toast region, keyboard shortcuts (C/F/D when a row
  has focus), and post-swap focus management.
- Contradictions + Definitions tabs gain inline verdict buttons,
  hide-reviewed-by-default with a "Show reviewed" toggle per section, and a
  progress count above each table.
- Markdown report filters `false_positive` findings out of the
  contradictions, multi-party, and definitions sections; surviving findings
  carry a `**Reviewer:** Real issue` / `Dismissed` / `Pending review` tag.

## Test plan

- [x] Hermetic suite: all new tests pass; pre-existing env failures untouched.
- [x] Ruff + ruff format clean.
- [ ] CI green.
- [ ] Manual: web UI buttons click, toast persists, undo works for both
      first-click and re-judge cases.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review

**Spec coverage:**
- Migration 0009 with composite PK + CHECK constraints → Task 1.
- `build_pair_key` + `ReviewerVerdict` dataclass + type aliases → Task 2.
- `set_reviewer_verdict` + `delete_reviewer_verdict` → Task 3.
- `get_reviewer_verdicts_bulk` + `count_reviewer_verdicts` → Task 4.
- `POST /verdicts` with OOB-swap response → Task 5.
- `POST /verdicts/undo` with delete-or-restore branching → Task 6.
- Verdict-buttons partial → Task 7.
- Toast partial + progress-count partial → Task 5 (alongside the route).
- Base template (toast region + keyboard JS + focus mgmt) → Task 8.
- Contradictions tab integration (pair + cross-document) → Task 9.
- Definitions tab integration → Task 10.
- Report: filter false_positive, reviewer tag, summary table column → Task 11.
- CSS → Task 12.
- futureplans.md update → Task 13.

**Placeholder scan:** Three steps reference Phase B / queue work; those are explicit out-of-scope notes in Task 13, not placeholders. No "TBD" / "implement later" / "similar to X" anywhere.

**Type consistency:**
- `pair_key: str`, `detector_type: DetectorType`, `verdict: ReviewerVerdictLabel` are used consistently across the audit logger methods, the route signatures (as `str` post-validation), and the dataclass.
- `VERDICT_LABELS` dict is defined twice (web app + report module) by design — the alternative is a shared module and an import everywhere. Two ~5-line constants is fine.
- Setter and delete are keyword-only everywhere (Tasks 3, 5, 6, audit calls).
- `_count_total_findings` helper introduced in Task 5 is reused by Tasks 9 and 10.

---

## Execution Handoff

**Plan complete and saved to** `docs/superpowers/plans/2026-05-15-reviewer-workflow.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks. Best when the codebase is large enough that catching architectural drift early matters. Tasks 9-11 are the most architectural; SDD review there pays off.

**2. Inline Execution** — execute tasks in this session with checkpoint pauses at phase boundaries. Lower per-task overhead.

Given the prior pattern (hybrid worked well — inline for the easy schema/CRUD tasks, SDD for the integration ones), suggest the same shape here: inline for Phase 0-1 (Tasks 1-4, all mechanical), then your choice for Phase 2+ depending on appetite.

Which approach?
