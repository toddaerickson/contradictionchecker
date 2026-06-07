# Collapse the 7-tab UI to a single-page corpora+findings layout

**Status**: planned
**Branch**: `feat/ui-collapse` (Phase 1) — each phase opens its own PR
**Date**: 2026-06-01
**Motivates**: replace the workflow-ordered tab nav (ADR-0011, shipped 2026-05-17) with a results-first single-page that matches how an auditor actually works. Closes two of three documented gaps from ADR-0011 (Action Items query, `/uploads` legacy endpoint) by subsuming them; CSV export deferred.

## Context

The current UI is 7 tabs (Ingest → Process → Assertions → Contradictions → Definitions → Action Items → Stats). The workflow ordering was correct for a brand-new user but is wrong for the user's actual loop, which is:

> Load a corpus once → run check once → live in the findings list, marking verdicts, occasionally drilling into raw assertions or stats to confirm a finding.

The tab nav forces a click-out-and-back pattern for every verdict mark, every drill-down, every cross-reference. Three operational signals confirm the gap:

- **Action Items tab** ships with the query unwired — the verdict-marking UI has nowhere to read findings from because findings live under a tab the Action Items query doesn't traverse.
- **CSV export button removed** — there was no clear home for a "export these findings" action.
- **`/uploads` legacy endpoint** still exists because the Ingest tab subsumed it without deleting it.

All three smell like "the tabs are wrong for the workflow." A single-page layout that puts findings front-and-center with corpora as the sidebar resolves the structural cause, not just the symptoms.

Cost has also been promoted to a first-class concern (PR #73, ADR-0016: `--max-cost`, provider-aware pricing). A persistent cost gauge in the header is now load-bearing UX, not a nice-to-have.

## Decision

Collapse to a single-page UI with three regions:

- **Header** (top): product name, persistent cost gauge (`spent $X / ceiling $Y` from the active corpus's most recent run), settings menu.
- **Sidebar** (left, ~280px): corpora list. Each row shows name, fact count, run count, and (for active runs) an inline SSE progress bar. `[+ New corpus]` opens a modal.
- **Main pane** (right, fluid): the selected corpus's findings stream. Header has `[Run check ▾]`, `[Assertions]`, `[Definitions]`, `[Stats]` buttons. Findings list below with inline verdict-marking buttons on each card.

Tabs collapse as follows:

| Old tab | New home |
|---|---|
| Ingest | `[+ New corpus]` modal |
| Process | `[Run check ▾]` modal + SSE progress on the sidebar row |
| Assertions | Drawer (slide-over from the right, escape-to-close) |
| Contradictions | Subsumed by Findings list (filter chip: "contradictions") |
| Definitions | Drawer |
| Action Items | Inline verdict buttons on each finding card + filter chips |
| Stats | Drawer |

API surface: every existing route stays. New routes are HTML fragments returned to HTMX:

- `GET /` — new single-page shell (replaces existing `/` tab nav)
- `GET /corpora/{id}/findings` — main-pane findings stream (paginated)
- `GET /corpora/{id}/drawer/{view}` — drawer fragments (`view ∈ {assertions, definitions, stats}`)
- `GET /corpora/{id}/progress` — SSE channel for ingest + check progress

Stack stays HTMX. No React. The sidebar/main-pane/drawer pattern is achievable with `hx-swap`, `hx-target`, and `<dialog>` for modals; the drawers are CSS transforms triggered by HTMX class swaps.

Two of the three ADR-0011 gaps close as a side effect:
- **Action Items query** — findings ARE the query; no separate tab, no separate query path.
- **`/uploads` legacy endpoint** — Phase 6 deletes it; the New Corpus modal subsumes it.

The CSV export gap stays open; routing it through a future "export findings" button on the main pane is a separate PR.

## Files

(Across all 6 phases. Each phase scopes a subset.)

### New
| File | Purpose |
|---|---|
| `consistency_checker/web/templates/cc_single.html` | New single-page shell (header + sidebar + main pane + drawer container) |
| `consistency_checker/web/templates/cc_sidebar.html` | Sidebar fragment (corpora list + new-corpus button) |
| `consistency_checker/web/templates/cc_findings.html` | Findings stream fragment (cards + inline verdict buttons + filter chips) |
| `consistency_checker/web/templates/cc_drawer.html` | Drawer container (used for assertions/definitions/stats) |
| `consistency_checker/web/templates/cc_new_corpus_modal.html` | New-corpus modal form |
| `consistency_checker/web/templates/cc_new_run_modal.html` | Run-check modal (pairwise/definitions/deep/max-cost toggles) |
| `consistency_checker/web/cc_collapsed.css` | New styles (sidebar, drawer transitions, finding cards) |
| `docs/decisions/0017-ui-collapse.md` | ADR documenting this redesign |
| `tests/test_web_ui_collapse.py` | End-to-end tests for each phase |

### Modified
| File | Change |
|---|---|
| `consistency_checker/web/app.py` | Wire new routes; flag old routes behind `?legacy=1` until Phase 6 |
| `consistency_checker/web/templates/cc_base.html` | Keep behind `?legacy=1` until Phase 6; then delete |
| `consistency_checker/web/templates/cc_assertions.html` | Repurpose body as the drawer fragment in Phase 4 |
| `consistency_checker/web/templates/cc_definitions.html` | Same |
| `consistency_checker/web/templates/cc_stats.html` | Same |
| `futureplans.md` | Move the ADR-0011 gaps under this Completed entry as they close |

### Deleted (Phase 6)
| File | Reason |
|---|---|
| `consistency_checker/web/templates/cc_ingest.html` | Replaced by New Corpus modal |
| `consistency_checker/web/templates/cc_process.html` | Replaced by Run Check modal + sidebar SSE |
| `consistency_checker/web/templates/cc_action_items.html` | Replaced by inline verdict buttons |
| `/uploads` route in `consistency_checker/web/app.py` | Subsumed by New Corpus modal |

## Phases (each one PR)

Each phase is shippable behind a `?new_ui=1` query-string flag. Phase 6 flips the default to on and deletes the legacy. Per [[feedback-review-every-third-commit]], run `/review silent-code-killer adversary` at the end of Phase 3.

### Phase 1 — Shell + sidebar + findings list (PR #1)

Build the bare bones. `GET /?new_ui=1` returns the new shell. Sidebar lists corpora from `store.list_corpora()`. Main pane shows findings from the most recent run of the most recently active corpus.

- ADR-0017 written.
- New routes: `GET /?new_ui=1`, `GET /corpora/{id}/findings` (HTMX fragment).
- Findings query reuses existing `audit_logger` reads.
- Cost gauge in header is a placeholder displaying `--` (Phase 5 wires the real query).
- Verdict buttons are placeholder shapes (Phase 5 wires submission).
- Drawer buttons are inert (Phase 4 wires them).
- Run Check button is inert (Phase 3 wires the modal).
- New Corpus button is inert (Phase 2 wires the modal).
- Old tabs still work at `/?legacy=1` (or no flag — flag defaults to legacy in Phase 1).
- Tests: a corpus exists → `GET /?new_ui=1` returns 200 with sidebar HTML containing the corpus name + findings panel HTML.

### Phase 2 — New Corpus modal (PR #2)

`[+ New corpus]` opens a modal (HTML `<dialog>` via HTMX). Form fields: judge provider, corpus name, ingest source (file upload OR directory path). Submits to a new `POST /corpora/new` endpoint that wraps the existing ingest pipeline (which already accepts `corpus_id`). Modal closes on success; sidebar HTMX-refreshes with the new row.

- New route: `POST /corpora/new` (multipart for file uploads, JSON for directory path).
- Background ingest fires via existing `_ingest_uploaded_paths()` plumbing.
- Sidebar row appears immediately in "ingesting" state.
- Tests: POST creates a corpus row in SQLite + spawns background ingest + returns the new sidebar row HTMX fragment.
- Old `/uploads` endpoint still works; Phase 6 deletes it.

### Phase 3 — Run Check modal + sidebar SSE (PR #3)

Per-corpus `[Run check ▾]` button opens a modal with toggles for `--pairwise`, `--no-definitions`, `--deep`, and a number input for `--max-cost`. Submits to the existing `POST /runs` (with the new flag values threaded through). Progress streams via a new `GET /corpora/{id}/progress` SSE channel; the sidebar row shows a progress bar.

- Re-use existing `_run_check_in_background` plumbing; just thread the new flags.
- SSE event types: `ingest_progress`, `check_progress`, `cost_update`, `done`, `failed`, `cost_ceiling_exceeded`.
- The cost gauge in the header now updates live from the `cost_update` events.
- Tests: POST /runs from the modal → background task starts → SSE emits at least one progress event → run reaches "done" status.
- **Run `/review silent-code-killer adversary` at the end of this phase per the rolling-review rule.** Address any critical/moderate findings before opening PR #3.

### Phase 4 — Drawers for Assertions, Definitions, Stats (PR #4)

`[Assertions]`, `[Definitions]`, `[Stats]` buttons in the main-pane header. Each opens a slide-over drawer (right-side panel, ~600px wide, escape-to-close, click-outside-to-close). Drawers fetch HTML fragments from `GET /corpora/{id}/drawer/{view}`.

- New route: `GET /corpora/{id}/drawer/{view}`.
- Repurpose existing `cc_assertions.html`, `cc_definitions.html`, `cc_stats.html` bodies as the drawer content (strip their `cc_base.html` wrapping).
- CSS for the drawer slide-in/out and the escape-key handler.
- Tests: drawer-open route returns 200; clicking a row in Assertions/Definitions deep-links to the matching finding in the main pane.

### Phase 5 — Inline verdict marking + filter chips (PR #5)

Each finding card grows `[✓ Resolved]`, `[✗ False positive]`, `[? Uncertain]` buttons. Clicking submits to the existing verdict endpoint and the button swaps to "Marked X · undo." Filter chips above the findings list: "All / Open / Resolved / FP / Uncertain." The header cost gauge becomes live: it reads `run.spent_usd / cfg.max_cost_usd` from the most recent run of the active corpus and updates on `cost_update` SSE events.

- The verdict POST endpoint already exists from the previous redesign.
- Filter chips are HTMX hx-get with a query param.
- Tests: clicking a verdict button POSTs the verdict + swaps the button label; filter chip changes the result set; cost gauge updates from a fake SSE event.

### Phase 6 — Delete old templates + flip flag (PR #6)

Delete `cc_ingest.html`, `cc_process.html`, `cc_action_items.html`. Delete the tabs nav from `cc_base.html` (or delete `cc_base.html` entirely if `cc_single.html` is the new base). Flip `?new_ui=1` to default on; `?legacy=1` no longer renders anything (route returns 410 Gone with a one-line note). Delete `/uploads` legacy endpoint. Update README screenshots if any.

- Final cleanup PR. No new feature work.
- Update `futureplans.md`: move ADR-0011 gaps to Completed under this entry; CSV export remains in the active list.
- Tests: GET old tab routes return 410 or redirect to the new shell.

## Non-goals

- **CSV export.** Deferred to a follow-up. The new header will have an empty space where the export button will eventually live.
- **A React rewrite.** HTMX is sufficient. The sidebar + drawer + modal pattern doesn't justify a SPA framework, and the existing tests would all break.
- **Multi-tenant / multi-user UI.** Single-user assumption preserved from ADR-0011.
- **Mobile-first.** Desktop-first. The drawer pattern works on tablet but is not a phone-screen-first layout.
- **Backwards-compatible URL aliases.** Old tab URLs (`/tabs/stats`, `/tabs/assertions`, etc.) return 410 in Phase 6 with a one-line "this UI was replaced; visit `/`" message.

## Risks

- **SSE multiplexing.** Multiple concurrent runs (one per corpus) need separate SSE channels keyed by corpus_id. The existing `/runs` SSE uses a global channel. Phase 3 needs a per-corpus rekey.
- **Drawer accessibility.** Slide-over drawers must be keyboard-navigable (Tab traps, Escape closes, focus returns to the trigger button). This was a documented miss in ADR-0011; this redesign must do better.
- **Legacy flag transition.** Phase 6 deletes the legacy templates. Users with bookmarked tab URLs hit 410. Acceptable for single-user, but the 410 message must point at the new shell.
- **CSS specificity wars.** `cc_style.css` from ADR-0011 has high specificity. Phase 1 adds `cc_collapsed.css` as a new sheet; conflicts get resolved per-finding, not by `!important` everywhere.
- **Test flake on SSE.** The existing SSE tests use `TestClient` which can race. Phase 3 should snapshot SSE events synchronously where possible (poll the audit log instead of trusting SSE timing).

## Verification at end (per phase)

```bash
uv run pytest -m "not slow and not live"
uv run ruff check .
uv run ruff format --check .
uv run mypy consistency_checker
uv build
```

All four green. Plus a manual browser pass: open the dev server, click through the workflow (new corpus → run check → mark verdicts → drill into drawer → cost gauge updates).

## Out-of-scope follow-ups

- CSV export.
- Per-finding comment threads (multi-user feature).
- Bulk verdict marking ("mark all FP within filter").
- Saved filter presets.
- Mobile responsive layout.

These get separate plans if they come up.
