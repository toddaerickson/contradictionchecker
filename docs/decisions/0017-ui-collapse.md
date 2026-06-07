# ADR 0017 — Collapse the 7-tab UI to a single-page corpora+findings layout

**Status**: Accepted

## Context

The current web UI (ADR-0011, shipped 2026-05-17) is a 7-tab nav: Ingest → Process → Assertions → Contradictions → Definitions → Action Items → Stats. The ordering was workflow-correct for a first-time user but is wrong for the actual loop the tool is used in:

> Load a corpus once → run check once → live in the findings list, marking verdicts and occasionally drilling into raw assertions, definitions, or stats to confirm a finding.

Three operational signals confirm the gap:

- **Action Items tab** ships with the query unwired. The verdict-marking surface has nowhere to read findings from because findings live under tabs the Action Items query doesn't traverse.
- **CSV export button removed.** There is no clear home for an "export these findings" action under the current tab decomposition.
- **`/uploads` legacy endpoint** still exists alongside the Ingest tab — the Ingest tab subsumed it without deleting it.

All three smell like "the tabs are wrong for the workflow." A single-page layout that puts findings front-and-center with corpora as the sidebar resolves the structural cause, not just the symptoms.

Cost was also just promoted to a first-class concern via ADR-0016 (`--max-cost`, provider-aware pricing). A persistent cost gauge in the header — `spent $X / ceiling $Y` for the active corpus's most recent run — is now load-bearing UX, not a nice-to-have.

## Decision

- Collapse to a single-page UI with three regions: a header (cost gauge + settings cog), a ~280px left sidebar listing corpora, and a fluid main pane that streams the active corpus's findings.
- Tabs map to new locations:
  - Ingest → `[+ New corpus]` modal (sidebar).
  - Process → `[Run check ▾]` modal + inline SSE progress on the sidebar row.
  - Assertions / Definitions / Stats → slide-over drawers from the right.
  - Action Items → inline verdict buttons on each finding card + filter chips above the list.
  - Contradictions → subsumed by the main-pane findings list, with a filter chip ("contradictions").
- New routes return HTML fragments to HTMX. No React; no SPA framework.
- Each phase ships behind `?new_ui=1`. Phase 6 flips the default to on, deletes the legacy templates, and deletes the `/uploads` route.
- Stack stays HTMX + Jinja templates + plain CSS. New styles land in `cc_collapsed.css`, layered on top of `cc_style.css` until Phase 6 prunes dead rules.
- **SSE consumed via the htmx-sse extension** (`static/htmx-sse.min.js`, ~370 lines vendored from htmx@1.9.12). Rationale per the "no net complexity" guardrail: an inline `<script>` using the native `EventSource` would also work, but the sidebar already wires every other event through HTMX attributes (`hx-get`, `hx-trigger="corpus-created from:body"`, `hx-swap`). Adding declarative `hx-ext="sse"` + `sse-connect` + `sse-swap` on the corpus row keeps the wiring consistent — readers don't have to flip between attribute-driven HTMX and imperative DOM event handlers to follow the live-update flow. A future Phase-6 audit may revisit and inline if the file proves load-bearing for nothing else, but Phase 3 keeps the declarative pattern intact.

## Alternatives considered

- **Fix the three gaps in place without restructuring.** Rejected. The gaps share a structural cause; fixing each as a one-off would not address the workflow-ordering mismatch and would compound complexity (one more half-wired tab, one more empty button hunt, one more legacy endpoint).
- **React SPA rewrite.** Rejected. HTMX is sufficient for the sidebar/main-pane/drawer pattern; React adds toolchain weight, breaks every existing template-rendering test, and the project is single-user. The novice-maintainability axis dominates.
- **Drawer-on-the-left, sidebar-on-the-right.** Rejected. Sidebar-left matches Atom/VSCode/Slack conventions; auditors will already have that mental model and the corpora list is the navigational primitive, not a transient overlay.
- **Keep the tabs, just add the cost gauge.** Rejected. Solves cost-gauge-promotion but leaves the workflow-ordering mismatch and Action Items gap unaddressed.

## Consequences

- Two of the three ADR-0011 gaps close as side effects: Action Items query disappears (findings ARE the query), and `/uploads` goes away (subsumed by the New Corpus modal in Phase 2, deleted in Phase 6).
- CSV export remains open; it gets routed to a future "export findings" button on the main pane in a separate PR.
- Old tab URLs (`/tabs/stats`, `/tabs/assertions`, etc.) become 410 Gone in Phase 6 with a one-line "this UI was replaced; visit `/`" message. Acceptable for a single-user tool with no published deep-links.
- A new per-`corpus_id` SSE channel is needed for Phase 3; the existing `/runs` SSE channel is global and cannot multiplex runs from multiple corpora.
- Drawer focus-trap and escape-to-close were a documented miss in ADR-0011; this redesign must do better. Phase 4 owns that quality bar.
- Phases 1-5 ship behind a flag, so a regression is recoverable by removing `?new_ui=1` from the URL. Phase 6 removes that escape hatch.
- **Phase 5 cost-gauge caveat:** the schema has no measured ``spent_usd`` column, so the header gauge renders an upper-bound estimate (`(n_pairs_judged + n_definition_pairs_judged) * per_call_high`, the same formula ADR-0016 uses for the pre-flight ceiling gate). The label is explicit ("Est. spent"). Replacing this with a measured-spend column is a future schema change.

Reference: [`docs/superpowers/archive/plans/2026-06-01-ui-collapse.md`](../superpowers/plans/2026-06-01-ui-collapse.md).
