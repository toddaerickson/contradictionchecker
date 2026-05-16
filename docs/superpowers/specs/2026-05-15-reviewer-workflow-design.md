# Reviewer-workflow detector design

**Date:** 2026-05-15
**Status:** Approved by user, pending implementation plan
**Tracks:** [`futureplans.md`](../../../futureplans.md) item #9 (Reviewer workflow)

## Summary

Turn the web UI from a read-only finding viewer into a workflow tool by letting
the user mark each finding with a three-value verdict. Verdicts persist across
re-runs of the same corpus (keyed by content, not run id), feed back into the
markdown report (`false_positive` findings are filtered out), and unlock a
labeled-feedback dataset for future prompt iteration.

This is the **Phase A** ("inline") build. A dedicated review-queue page is
deferred to Phase B.

## Motivation

The existing web UI shows findings but offers no way to act on them. A reviewer
who concludes "this is a false positive" has no place to record it; the next
report includes the noise; the iteration loop is broken.

The futureplans entry has reserved the conceptual seam (a `reviewer_verdict`
field) for some time. This design realizes it with three correctness fixes
caught during brainstorming review:

- The verdict belongs in a content-keyed table, not a run-scoped column, so
  re-checking the same corpus doesn't wipe review work.
- The primary key must include `detector_type` so a single content pair flagged
  by both the contradiction and definition detectors can carry independent
  verdicts.
- The UI uses plain-English button labels ("Real issue" / "Not an issue" /
  "Skip for now") even though the internal enum values stay technical.

## Scope

### In scope (Phase A)

Inline verdict buttons on the existing Contradictions, Definitions, and
Cross-document tabs. Verdicts persist across runs. Markdown report filters
`false_positive` and renders a per-finding reviewer tag. CSV unchanged.

Three finding types are covered: pair contradictions, definition
inconsistencies, multi-party (triangle) contradictions.

### Out of scope (Phase B / future)

- Dedicated "Review" tab with focused queue UI.
- Note column surfaced in the UI (column exists in schema, no UI in v1).
- Findings CSV export with a `reviewer_verdict` column.
- Verdict history / audit-grade trail (current design hard-deletes on undo).
- Reviewer identity (single-user localhost tool).
- `--include-rejected` CLI flag (declined this round).
- Persona-aware verdict bias (out of scope per ADR-0008).

## Decisions confirmed during brainstorm

1. **UX shape — inline buttons on existing tabs.** Phase B will add a queue.
2. **Scope — all three finding types** get the verdict surface.
3. **Verdict vocabulary — three values.** `confirmed`, `false_positive`,
   `dismissed`. NULL = unreviewed.
4. **Row behavior — hide reviewed by default**, with a "Show reviewed" toggle.
5. **Report/CSV — exclude `false_positive` from the report**, include
   confirmed/dismissed/unreviewed with a per-finding reviewer tag. CSV
   unchanged in this PR.
6. **Persistence — verdicts in a separate `reviewer_verdicts` table keyed by
   `(pair_key, detector_type)`**, surviving re-runs.
7. **Metadata stored** — auto-`set_at` timestamp + nullable `note` column.
   No reviewer identity. Note column has no v1 UI.

## Data model

### Migration `0009_reviewer_verdicts.sql`

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

**No foreign key** to `assertions(assertion_id)` — intentional. Verdicts must
survive document-delete-and-reingest cycles; cascading FK would defeat the
persistence design. Orphan tolerance is the correct behavior.

**No separate `idx_reviewer_verdicts_detector` index** — `detector_type` is part
of the primary key, so the PK B-tree covers detector-filtered queries.

### Setter SQL (used by `AuditLogger.set_reviewer_verdict`)

```sql
INSERT INTO reviewer_verdicts (pair_key, detector_type, verdict, note)
VALUES (?, ?, ?, ?)
ON CONFLICT (pair_key, detector_type)
DO UPDATE SET verdict = excluded.verdict,
              note = excluded.note,
              set_at = CURRENT_TIMESTAMP;
```

Explicit upsert, not `INSERT OR REPLACE` — safer when columns are added in
future migrations (the old form would silently null out any unsupplied column).

### Render-time join SQL (pair findings; mirrors for multi-party)

```sql
LEFT JOIN reviewer_verdicts rv
  ON rv.pair_key = CASE
       WHEN f.assertion_a_id < f.assertion_b_id
       THEN f.assertion_a_id || ':' || f.assertion_b_id
       ELSE f.assertion_b_id || ':' || f.assertion_a_id
     END
  AND rv.detector_type = f.detector_type
```

## Audit logger surface

### New module: `consistency_checker/audit/reviewer.py`

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
        raise ValueError("pair_key needs at least 2 assertion ids")
    return ":".join(sorted(assertion_ids))


@dataclass(frozen=True, slots=True)
class ReviewerVerdict:
    pair_key: str
    detector_type: DetectorType
    verdict: ReviewerVerdictLabel
    set_at: datetime | None = None
    note: str | None = None
```

### New methods on `AuditLogger` (in `audit/logger.py`)

```python
def set_reviewer_verdict(
    self,
    *,
    pair_key: str,
    detector_type: DetectorType,
    verdict: ReviewerVerdictLabel,
    note: str | None = None,
) -> None:
    """UPSERT a verdict. Refreshes set_at on every call."""

def delete_reviewer_verdict(
    self, *, pair_key: str, detector_type: DetectorType
) -> None:
    """Remove a verdict (backs the undo toast for first-click cases)."""

def get_reviewer_verdicts_bulk(
    self,
    keys: Sequence[tuple[str, DetectorType]],
) -> dict[tuple[str, str], ReviewerVerdict]:
    """Fetch verdicts for the given (pair_key, detector_type) tuples.

    Uses SQLite IN (VALUES (?, ?), ...) for single-query batch lookup.
    Returns dict keyed by (pair_key, detector_type); missing keys are absent.
    """

def count_reviewer_verdicts(
    self, *, detector_type: DetectorType | None = None
) -> dict[str, int]:
    """Counts per verdict label. Used for 'X of N reviewed' progress."""
```

Setter and delete are keyword-only — `pair_key` and `detector_type` are both
strings; positional calls would silently swap them.

## Web UI

### Per-row inline buttons

Each finding row in Contradictions, Definitions, and the Cross-document section
of Contradictions gets a new rightmost column with three icon-only buttons.
Button glyphs are minimal; labels and shortcuts live in tooltips and aria
attributes.

| Button | aria-label | Tooltip | Enum | Shortcut |
|---|---|---|---|---|
| ✓ | "Mark as real issue" | "Real issue — model flagged this correctly (C)" | `confirmed` | `c` |
| ✗ | "Mark as not an issue" | "Not an issue — model flagged this incorrectly (F)" | `false_positive` | `f` |
| — | "Skip for now" | "Skip — real but not yours to resolve (D)" | `dismissed` | `d` |

A new partial `cc__verdict_buttons.html` is included by each tab's row template
to keep the pattern DRY across all three tabs.

### Above-table state

Each finding section gets its own header strip (so the Contradictions tab has
two — one above the pair-contradictions sub-table, one above the
cross-document sub-table — and the Definitions tab has one):

```text
12 of 47 reviewed                                        ☐ Show reviewed
```

- **Progress count** is `count_reviewer_verdicts(detector_type=...)` divided
  by the visible run's finding count of that type. It reflects the
  underlying audit state, not what's currently rendered — toggling "Show
  reviewed" never changes the count.
- **"Show reviewed" checkbox** — labeled, persistent. Default off;
  query-string `?show_reviewed=1` (per detector type, e.g.
  `?show_reviewed_contradiction=1`) flips that section's WHERE clause. Wired
  to `hx-get` with `hx-include="this"` so checkbox state is sent with each
  load.

### Click flow

```text
[User clicks ✓ on a row]
   ↓
POST /verdicts                       (HTMX, hx-vals: pair_key, detector_type,
                                      verdict, prior_verdict)
   ↓
Server:
  • set_reviewer_verdict(...)
  • returns HTMX response with three swaps:
      1. The row → empty (hx-target=row, hx-swap="outerHTML")
      2. OOB swap into #cc-toast-region (undo toast)
      3. OOB swap into #cc-progress-count (updated count)
   ↓
Page:
  • Row vanishes
  • Toast appears: "Marked as 'Real issue'. [Undo] [×]"
  • Progress count increments
```

**No auto-dismiss.** The toast persists until the user clicks `×` to dismiss,
or until a new verdict action replaces it (the OOB swap into
`#cc-toast-region` overwrites the prior toast in place). Single line of vanilla
JS for the close: `onclick="this.parentElement.remove()"`.

### Toast template (`cc__verdict_toast.html`)

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

### Undo

The undo endpoint reads the `prior_verdict` from the toast's hidden form value:

- Empty `prior_verdict` (first-click case) → `delete_reviewer_verdict`.
- Non-empty `prior_verdict` (re-judge case) → `set_reviewer_verdict` with the
  prior value.

Response refreshes the whole tab content (`hx-target="#cc-tab-content"`).
Slight visual flash is acceptable for an explicit user action; optimization
path (single-row partial + HTMX prepend) is available if it becomes
annoying.

### Empty state

When all findings of a kind are reviewed and "Show reviewed" is off, the table
renders:

```text
All findings reviewed.
```

Plain text. No celebration banner.

### Accessibility

- `<div id="cc-toast-region" role="status" aria-live="polite">` in
  `cc_base.html` ensures screen readers announce the toast
  non-interruptively.
- Verdict badges in the "Show reviewed" view use text + color, never color
  alone (WCAG 1.4.1):
  `<span class="cc-verdict-badge cc-verdict-badge--confirmed">Real issue</span>`.
- After a row is removed, page-level JS on `htmx:afterSwap` moves focus to the
  next `tr.cc-finding-row` in the same table, not to the top of the page.
- Each finding row gets `tabindex="0"`. ~10 LOC of vanilla JS in `cc_base.html`
  listens for `c` / `f` / `d` when a row has focus and dispatches the matching
  button's click.
- No confirm dialogs anywhere — the toast Undo is the safety net.

### New routes

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/verdicts` | `pair_key`, `detector_type`, `verdict`, `prior_verdict?` | HTMX OOB-swap response (row removal + toast + progress) |
| `POST` | `/verdicts/undo` | `pair_key`, `detector_type`, `prior_verdict?` | Full tab refresh into `#cc-tab-content` |

### New / modified templates

- **New**:
  - `cc__verdict_buttons.html` — three icon buttons + `data-verdict` attrs.
  - `cc__verdict_toast.html` — toast snippet, OOB-swappable.
  - `cc__progress_count.html` — "X of N reviewed" snippet, OOB-swappable.
- **Modified**:
  - `cc_contradictions.html` — include verdict-buttons partial in row template;
    add above-table progress + checkbox; treat Cross-document section the
    same.
  - `cc_definitions.html` — same as above for the definitions table.
  - `cc_base.html` — add `#cc-toast-region` live region; add small
    keyboard-shortcut + focus-management JS block.

## Report and CSV integration

### Markdown report (`audit/report.py`)

Three changes:

1. **Query-time filter** — each section's query LEFT JOINs `reviewer_verdicts`
   and adds `WHERE rv.verdict IS NULL OR rv.verdict != 'false_positive'`.
2. **Per-finding reviewer tag** — each finding's rendered block gains one
   new line above the rationale:

   ```markdown
   **Reviewer:** Real issue
   ```

   Mapping:
   - `confirmed` → `Real issue`
   - `dismissed` → `Dismissed`
   - NULL → `Pending review`

   `false_positive` never appears because those rows are filtered at query
   time.
3. **Summary table column** — the per-section summary table gains a
   rightmost `Reviewer` column with the same plain-English label.

The same treatment applies to `_append_multi_party_section`.

### CSV export

No change to assertion CSV in this PR. A future `findings_csv` export with a
`reviewer_verdict` column is the natural Phase B addition.

### CLI

No new CLI flags. The user direction was that `--include-rejected` is YAGNI
because the web UI's "Show reviewed" toggle covers the audit case.

## Testing strategy

All hermetic (no `slow`, no `live`). Five test files:

### `tests/test_migrations.py` (extend)

- `test_migration_0009_adds_reviewer_verdicts_table` — table exists, composite
  PK is `(pair_key, detector_type)`, CHECK constraints reject bogus values.
- `test_migration_0009_idempotent` — re-running is a no-op.

### `tests/test_reviewer.py` (new)

- `test_build_pair_key_two_assertions_sorted` — alphabetical sort.
- `test_build_pair_key_three_assertions_sorted` — works for triangles.
- `test_build_pair_key_rejects_singleton` — raises on <2 ids.

### `tests/test_audit_logger.py` (extend)

- `test_set_reviewer_verdict_inserts_row`
- `test_set_reviewer_verdict_upserts_on_conflict` — second `set` overwrites;
  `set_at` refreshes; `note` carries through.
- `test_set_reviewer_verdict_same_pair_different_detector` — both rows coexist
  because the PK is composite.
- `test_set_reviewer_verdict_rejects_invalid_verdict` — CHECK constraint
  rejects `"banana"`.
- `test_delete_reviewer_verdict_removes_row`.
- `test_get_reviewer_verdicts_bulk_returns_present_only` — missing keys absent
  from dict.
- `test_get_reviewer_verdicts_bulk_empty_input` — returns empty dict.
- `test_count_reviewer_verdicts_groups_by_verdict`.
- `test_count_reviewer_verdicts_filtered_by_detector`.

### `tests/test_web_verdicts.py` (new)

- `test_post_verdicts_inserts_and_returns_oob_swaps`.
- `test_post_verdicts_undo_first_click_case_deletes`.
- `test_post_verdicts_undo_rejudge_case_restores_prior`.
- `test_contradictions_tab_hides_reviewed_by_default`.
- `test_definitions_tab_hides_reviewed_by_default`.
- `test_cross_document_section_hides_reviewed_by_default`.
- `test_progress_count_reflects_reviewed_total`.

### `tests/test_report.py` (extend)

- `test_report_excludes_false_positive_findings`.
- `test_report_renders_reviewer_tag_for_confirmed`.
- `test_report_renders_reviewer_tag_for_dismissed`.
- `test_report_renders_reviewer_tag_for_unreviewed`.
- `test_report_summary_table_has_reviewer_column`.
- `test_multi_party_section_excludes_false_positives`.

Total: ~25 new tests, all hermetic.

## Acceptance

This feature ships when:

- Migration 0009 applied; idempotent; CHECK constraints active.
- `AuditLogger` exposes the four new methods + `build_pair_key` helper.
- The three finding tabs render inline verdict buttons; clicking writes to
  `reviewer_verdicts`; the row hides; the toast appears; undo restores.
- "Show reviewed" toggle works on all three tabs.
- Progress count above each table reflects current verdict state.
- Keyboard shortcuts C/F/D work when a row has focus.
- Markdown report excludes `false_positive` findings; includes the Reviewer
  column and per-finding tag.
- All ~25 new hermetic tests pass; existing suite stays green; ruff + mypy
  clean.
