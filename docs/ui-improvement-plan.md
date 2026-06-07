# UI improvement plan

Synthesis of a four-lens UI audit (accessibility / UX-flows / visual-consistency /
capability-gaps) of the single-page web UI (ADR-0017). Ordered by value. Each phase
is independently shippable; sub-items map cleanly to one PR each.

## Headline

The visual base is good (disciplined `:root` tokens, mono/sans "instrument" system,
real focus-trapped drawer). The deficits cluster in four places:

1. **The review loop is hollow** — the findings card shows *one line*; the two
   conflicting assertions, their source docs, and the full rationale are all
   computed server-side (`_collect_findings_for_run`) and then **thrown away**.
   You mark confirmed/false-positive/dismissed essentially blind.
2. **Accessibility** — no global `:focus-visible`, modals injected with `open`
   instead of `showModal()` (no focus trap/Esc/restore), filter chips use a broken
   `role="tab"` pattern, pagination links have no `href` (keyboard-unreachable).
3. **The visual system forked** — `cc_style.css` + `cc_collapsed.css` re-implement
   parallel modal/input systems; a never-done "Phase 6 prune"; ~4 hardcoded greens
   and several reds bypass tokens; three drawer pages still wear deleted-7-tab chrome
   ("03 — Assertions", "go to the Ingest tab").
4. **Backend power with no UI** — pre-run cost estimate, run history, document list,
   report view, OCR-on-ingest toggle all exist in the backend/CLI but aren't reachable.

---

## Phase 1 — Make the review loop actually work  (highest value; core purpose)

- **1a. Findings card shows source context.** Render Doc A + assertion A ↔ Doc B +
  assertion B + full `judge_rationale` per finding, expandable in place (a `<details>`
  or hx-get detail fragment on each `<li class="cc-finding">`). Data already on the
  server — pure template work. *(UX-H1, Visual-M4)* — **the single highest-leverage fix.**
- **1b. One verdict component.** Today the main pane uses text pills
  (`cc__finding_actions.html`: "✓ Confirmed / ✗ False positive / ? Dismissed`") and the
  Definitions drawer uses icon squares (`cc__verdict_buttons.html`: ✓/✗/—) for the same
  three verdicts, with different labels ("Dismissed" vs "Skip"). Consolidate to one
  component + one vocabulary everywhere. *(UX-M4, Visual-H1)*
- **1c. Wire the advertised hotkeys.** The verdict buttons advertise "(C)/(F)/(D)" in
  their titles but no key handler exists. Bind C/F/D to the focused finding + j/k row
  nav for fast triage of large corpora. *(UX-H3)*
- **1d. Fix stale/ambiguous states.** Remove "go to the **Ingest tab** / **Action
  Items**" copy from the three drawer pages (those tabs were deleted in ADR-0017).
  Distinguish "run found no contradictions ✓" (success) from "no run yet" in the empty
  state. *(UX-M5, Visual-M3)*

## Phase 2 — Accessibility baseline (WCAG 2.2 AA; mostly cheap)

- **2a. Global `:focus-visible`** outline (≥2px, ≥3:1) — one CSS block; fixes the worst
  keyboard failure. Light variant for dark-header controls. *(A11y-C1 / SC 2.4.7, 2.4.11)*
- **2b. Modals via `showModal()`** + store/restore `activeElement` on the htmx
  `afterSwap`/close. Gives focus trap, Esc, and a real backdrop for free; today they're
  injected with `open` and leak focus to the page behind. *(A11y-C2 / SC 2.4.3, 4.1.2)*
- **2c. Filter chips:** drop `role="tab"`/`tablist`/`aria-selected` (no tabpanel/arrow
  contract exists) → plain `<nav>` links with `aria-current="true"`. *(A11y-S1)*
- **2d. Pagination:** Prev/Next are `<a hx-get>` with **no `href`** → keyboard-unreachable;
  add `href` (progressive enhancement) or make them `<button>`. *(A11y-M8 / SC 2.1.1)*
- **2e. Live regions:** scope a polite `aria-live` to the SSE progress *label* (+
  `role="progressbar"`); de-nest the toast (region-only `role="status"` + `aria-atomic`);
  keep the cost gauge silent (announce ceiling-exceeded via the toast). *(A11y-S2/S3/M3)*
- **2f. Contrast tokens:** `--cc-muted #767672` is 4.11:1 on the page bg (fails 4.5),
  `--cc-accent` text on tints fails, `--cc-border-strong` 2.24:1 fails 3:1 for control
  borders. Darken the tokens. *(A11y-M7 / SC 1.4.3, 1.4.11)*
- **2g. Misc:** `@media (prefers-reduced-motion)`; min 24×24 target size on small/icon
  buttons; demote `.cc-tab-title` eyebrow `<h2>`→`<p>` (heading order); `aria-hidden`
  decorative glyphs; label the disabled cog; re-query drawer focusables at Tab-time so
  in-drawer hx-swaps don't drop focus. *(A11y-S5/S6/M4/M6, M1)*

## Phase 3 — Visual system consolidation (pays down the fork)

- **3a. Merge / re-layer the two stylesheets.** Finish the never-done "Phase 6 prune":
  move all `cc-modal-*` + input rules into one file so the other is purely layout; kill
  the duplicate input padding and the orphan `.cc-dialog`/`.cc-findings-table` rules.
  Total <1000 lines, shipped on every page. *(Visual-H2)*
- **3b. Tokenize status colors.** Add `--cc-success-fg / --cc-error-fg / --cc-danger /
  --cc-accent-hover`; replace the ~4 hardcoded greens and several reds. *(Visual-M2)*
- **3c. Action row hierarchy.** 8 buttons of mixed sizes in one bar. Group: **[Run check]**
  · [Add files] · [Assertions/Definitions/Stats] · overflow "⋯" [Rename, Delete, Export];
  one consistent secondary size; destructive Delete out of the flat row. *(UX-M1, Visual-H3)*
- **3d. Modernize the 3 drawer pages.** Drop "NN —" prefixes and the card-on-card
  `.cc-section` wrapper inside drawers (drawer already provides surface+padding). *(Visual-M3)*
- **3e. Responsive shell.** `.cc-shell` (`280px 1fr`) has no media query — add one that
  stacks/collapses the sidebar under ~700px (and revisit body `overflow:hidden` at 400%
  zoom). *(Visual-M5, A11y-M10 / SC 1.4.10)*
- **3f. Findings scannability.** Status-colored left border + verdict badge on each
  finding so confirmed/open/dismissed is glanceable. *(Visual-M4)*

## Phase 4 — Capability gaps (UI over existing backend; little/no backend work)

- **4a. Pre-run cost estimate** in the Run modal (hx-get → `run_estimate_cost`, which
  exists) so `max_cost` isn't set blind. *(Functional-2; backend exists)* — small, high value.
- **4b. Run history / run picker.** Everything keys off the latest run only; a re-run
  silently replaces what you see and orphans prior verdicts. Add `list_runs(corpus_id)` +
  a picker re-targeting `?run=<id>`. *(UX-H2, Functional-3)*
- **4c. Run-in-progress banner on the main pane** (wired to the same SSE the sidebar
  uses) — the primary surface is currently silent while a run progresses elsewhere; this
  is the recurring "is it actually working?" gap. *(UX-L3)*
- **4d. Document list view** (read) — see files in a corpus, org label, assertion count;
  later single-doc delete/re-ingest (needs a small new store method). *(Functional-1)*
- **4e. OCR toggle on the ingest modals** (new-corpus + add-files) — matters for scanned
  PDFs (the Atkins case). Tiny. *(Functional-8)*
- **4f. Findings search + detector-type filter + sort** for large corpora. *(Functional-4, UX-H3)*
- **4g. Report view + reviewer-decision/eval export** — `render_report`, `iter_eval_rows`,
  `compute_detector_precision` exist with no UI. *(Functional-5/6)*
- **4h. Disable "Run check" when the corpus has 0 assertions** (scanned-PDF dead-end) and
  point the empty CTA at "Add files". *(UX-H4)*

## Phase 5 — Settings (needs a config-persistence story first)

The settings cog is permanently disabled; provider-after-creation, default max-cost,
default pairwise, OCR default, thresholds live only in YAML/`CC_*`. Config is **frozen
Pydantic** (`model_copy` per request), so a settings drawer needs a persistence
mechanism (write-back to `config.yml` or a settings table) — design that first. Until
then, hide the dead cog (Phase 2g) rather than ship a disabled control. *(UX-M2, Functional-8)*

---

## Suggested execution order

1. **Phase 1** (review loop) — directly serves the product's purpose; mostly template work.
2. **Phase 2** (a11y) — cheap, high-correctness; many items are one CSS block.
3. **Phase 3** (visual consolidation) — pays down the stylesheet fork before more UI lands.
4. **Phase 4** (capability gaps) — pick by value; 4a/4c/4e/4h are small and high-impact.
5. **Phase 5** (settings) — deferred; needs a persistence decision.

Every UI PR runs the standing two passes (consistency + adversarial density) and never
trades label/field legibility for space (global CLAUDE.md rule).
