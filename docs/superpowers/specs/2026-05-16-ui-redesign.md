# UI Redesign: Workflow-Based Tab Navigation

**Date:** 2026-05-16  
**Goal:** Reorganize the web UI from a scattered tab layout to a linear workflow that guides users from corpus setup through contradiction analysis and action items.

**Current State:** Six tabs (Contradictions, Definitions, Documents, Statements, Stats, Ingest) with unclear organization. Users don't know what order to use them in.

**Target State:** Seven workflow-ordered tabs (Ingest → Process → Assertions → Contradictions → Definitions → Action Items → Stats) that follow natural user progression and NLP best practices.

---

## Design Principles

1. **Workflow-first**: Tabs are ordered left-to-right by when users need them
2. **Detail + Action split**: Exploration tabs (Assertions, Contradictions, Definitions) for understanding; unified Action Items tab for doing
3. **Linguistic semantics**: Contradictions and definitions are separate phenomena requiring different fixes
4. **Permanent corpora**: Filesystem-based, reusable, can be extended with new files
5. **Real-time transparency**: Streaming progress during processing so users see what's happening

---

## Tab Structure (Left to Right)

### Tab 1: Ingest

**Purpose**: Set up the run — choose judge, select or create a corpus, upload/point to files.

**Components**:
- **Judge selector** (dropdown, default: Moonshot, option: Claude)
  - Label: "Judge Provider"
  - Options: Moonshot (default), Claude
  - Help text: "Which LLM should judge contradictions?"
  
- **Corpus management**
  - Label: "Corpus"
  - State A (no prior corpora): Text input to create new corpus name, or button to "Browse existing"
  - State B (show prior corpora): Dropdown list of existing corpus names with "Create New" option
  - Help text: "A corpus is a persistent collection of documents. You can add files to an existing corpus to see how they interact."

- **File input**
  - After corpus is selected/created, show file upload
  - Drag-and-drop or file picker
  - Accepted formats: .txt, .md, .pdf, .docx
  - Help text: "Upload documents or point to a directory"
  - Option for "Use existing directory" (path input)

- **Action button**: [Start Processing]
  - Disabled until corpus is selected and files are provided
  - On click: Advance to Process tab and start the pipeline

**Behavior**:
- Validate corpus name (no invalid filesystem chars)
- Create corpus directory if it doesn't exist: `data/corpora/<corpus_name>/`
- Store uploaded files in that directory
- Remember last-used corpus for next session (cookie or config)

---

### Tab 2: Process

**Purpose**: Real-time progress streaming during document parsing, assertion extraction, and contradiction detection.

**Components**:
- **Progress stream** (scrollable log)
  - Shows messages as they arrive from the backend:
    - "Parsing document 1/12: report.pdf"
    - "Extracted 45 assertions from report.pdf"
    - "Building semantic index..."
    - "Running contradiction detection (NLI gate)..."
    - "Checking definition consistency..."
    - "✓ Done. Summary: 120 assertions, 8 contradictions, 3 definition issues found"
  - Style: Monospace, scrollable, auto-scroll to bottom
  - Use HTMX or SSE to stream messages in real-time

- **Status indicator**
  - Current phase (e.g., "Parsing documents", "Running judges", "Complete")
  - Elapsed time
  - Estimated time remaining (if available)

- **Action buttons**
  - [View Results] — Advances to Assertions tab (or Action Items if user prefers)
  - [Download Log] — Saves progress stream as .txt for debugging

**Behavior**:
- Backend streams progress via SSE or HTMX polling
- Auto-advance to Assertions tab on completion (with option to stay on Process)
- If error: show error message, allow retry or go back to Ingest

---

### Tab 3: Assertions (Exploration)

**Purpose**: Review what was extracted. Understand the raw assertion data.

**Components**:
- **Table**
  - Columns: Assertion Text, Source Document, Kind (Fact/Definition), Doc section/paragraph, Assertion ID (for reference)
  - Sortable by all columns
  - Filterable by Kind, Source
  - Searchable (text search across assertion text)
  - Pagination or lazy-load for large corpora (100+ assertions)

- **Detail view** (click row)
  - Full assertion text
  - Source doc with context
  - Kind and metadata
  - Close button or click-outside to dismiss

**Behavior**:
- Read-only exploration
- No user actions here (this tab is for understanding, not fixing)

---

### Tab 4: Contradictions (Exploration)

**Purpose**: Review contradiction pairs detected by the judge. Understand the semantic conflicts.

**Components**:
- **Table**
  - Columns: Assertion A, Assertion B, Judge Verdict (Contradiction/Not Contradiction/Uncertain), Confidence, Rationale
  - Sortable by verdict, confidence
  - Filterable by verdict
  - Searchable
  - Pagination for large result sets

- **Detail view** (click row)
  - Both assertion texts side-by-side
  - Judge rationale (full text)
  - Confidence score visualized (bar or percentage)
  - Source docs for each assertion
  - Close to dismiss

**Behavior**:
- Read-only exploration
- Shows judge's reasoning; users will set their own verdict in Action Items tab

---

### Tab 5: Definitions (Exploration)

**Purpose**: Review definition inconsistencies. Understand term harmonization issues.

**Components**:
- **Table**
  - Columns: Canonical Term, Definition A, Definition B, Judge Verdict (Consistent/Inconsistent/Uncertain), Confidence, Rationale
  - Sortable by term, verdict, confidence
  - Filterable by verdict
  - Searchable by term
  - Pagination

- **Detail view** (click row)
  - Canonical term (normalized)
  - Both definition texts side-by-side
  - Judge rationale
  - Confidence score
  - Source docs
  - Close to dismiss

**Behavior**:
- Read-only exploration
- Users will reconcile definition issues in Action Items tab

---

### Tab 6: Action Items (Unified & Actionable)

**Purpose**: Single source of truth for all issues needing user verdict/reconciliation. Export for corpus cleanup.

**Components**:
- **Unified findings table**
  - Columns: 
    - Type (badge: Contradiction / Definition / Multi-party / [Future type])
    - Issue (Assertion A vs Assertion B, or "Term: X vs Y")
    - Judge Verdict (what the LLM said)
    - User Verdict (Confirmed / False Positive / Dismissed / Pending — dropdown)
    - Status (Open / Resolved — based on user verdict)
    - Confidence (numeric, sortable)
    - Actions (buttons)
  
  - Rows color-coded or icon-marked by Type:
    - Contradiction: Red/Orange
    - Definition: Blue
    - Multi-party: Purple
    - Future: Gray

  - Sortable by all columns
  - Filterable by Type, Judge Verdict, User Verdict, Status
  - Searchable across issue text
  - Pagination or infinite scroll

- **Inline verdict setting** (dropdown per row)
  - User clicks cell in "User Verdict" column
  - Dropdown appears: Confirmed / False Positive / Dismissed / Pending
  - On select: Updates DB, row status changes, row moves (if filtering by Open/Resolved)
  - Option to add a note (small text input or click-to-expand)

- **Bulk actions** (header controls)
  - Select all / deselect all (checkboxes)
  - Bulk set verdict (select rows, choose verdict from dropdown)
  - Mark all as resolved

- **Export controls**
  - [Download as CSV] — All visible rows, all columns, for spreadsheet review
  - [Download as JSON] — Structured export for external tools
  - Filter/sort applied to export

**Behavior**:
- On load: Show all issues, default sorted by Confidence descending
- Verdict changes persist immediately to DB (or batch-save on page unload)
- Resolved issues remain in table but can be filtered out (checkbox: "Show resolved")
- No verdict-changing capability elsewhere; this is the single source of truth

---

### Tab 7: Stats (Summary & Results)

**Purpose**: Review corpus health and run summary. Understand the big picture of what was found.

**Components**:

- **Corpus Overview** (section)
  - Documents: N uploaded, M parsed successfully
  - Assertions: N total, K definitions, L facts
  - Coverage: X% of documents have assertions

- **Findings Summary** (section)
  - Contradictions: N found, M confirmed, K false positive, L dismissed, P pending
  - Definitions: N found, M confirmed, etc.
  - Multi-party: N found (if applicable)

- **Judge Metrics** (section)
  - Provider used: Moonshot or Claude
  - Average confidence: X.XX
  - Verdict distribution: Pie chart (Contradiction / Not Contradiction / Uncertain)
  - Processing time: H:MM:SS

- **Charts**
  - Contradiction density by document (bar chart: docs on X, % of assertions in contradictions on Y)
  - User verdict distribution (pie: Confirmed / False Positive / Dismissed / Pending)
  - Definition issues by term (word cloud or bar chart)

- **Export/Archive**
  - [Export Full Report] — Markdown or PDF with all stats, contradictions, definitions
  - [Archive this run] — Save run metadata for historical comparison

**Behavior**:
- Auto-populate on Process tab completion
- Update in real-time if user changes verdicts in Action Items tab
- Read-only; no user actions here except export

---

## Data & Persistence

### Corpus Structure
```
data/corpora/
  ├── corpus_name_1/
  │   ├── documents/            (uploaded files)
  │   │   ├── report.pdf
  │   │   └── notes.txt
  │   ├── .metadata.json         (corpus metadata: creation date, judge provider, etc.)
  │   └── runs/
  │       ├── run_20260516_1/    (each run gets a directory)
  │       │   ├── findings.json
  │       │   ├── assertions.json
  │       │   ├── verdicts.json
  │       │   └── progress.log
  │       └── run_20260516_2/
```

### Session State
- **Current corpus**: Stored in session or cookie, remembered for next session
- **Judge provider selection**: Stored per corpus in metadata
- **User verdicts**: Stored in DB (findings table with `user_verdict` column)
- **Processing state**: Stored in run directory, cleared on new run

---

## User Interactions & Workflows

### Workflow A: New Corpus, From Scratch
1. Ingest: Create "Q1 Audit" corpus, upload 5 PDFs
2. Process: Watch progress stream, see "✓ Done"
3. (Optional) Assertions: Skim extracted data
4. (Optional) Contradictions: Review some contradiction details
5. Action Items: Set verdicts (Confirmed/False Positive/Dismissed)
6. Stats: Review summary, download CSV for cleanup team
7. Exit

### Workflow B: Add Files to Existing Corpus
1. Ingest: Select "Q1 Audit" corpus (from dropdown)
2. Upload 3 new files
3. [Start Processing]
4. Process: Watch progress, see new contradictions found
5. Action Items: Filter by recent findings, set verdicts
6. Stats: Note the updated numbers
7. Exit (corpus persists for future additions)

### Workflow C: Deep Dive into One Issue
1. Ingest → Process → done
2. Contradictions: Find a specific pair, click to see judge rationale
3. Action Items: Find same pair, set verdict, note why (False Positive)
4. Back to Action Items, continue with other issues
5. Exit

---

## Technical Implementation Notes

### Frontend
- Use HTMX for tab switching (cache to avoid re-fetches of static exploration tabs)
- SSE or HTMX polling for Process tab streaming
- Real-time DB updates for verdict changes (AJAX POST to verdict endpoint)
- Lightweight sorting/filtering (could use client-side JS for small datasets, server-side for large)

### Backend
- New endpoints:
  - `POST /corpora` — Create corpus
  - `GET /corpora` — List existing corpora
  - `GET /corpora/<name>` — Get corpus metadata
  - `POST /corpora/<name>/runs` — Start processing run
  - `POST /findings/<id>/verdict` — Set user verdict
  - Export endpoints for CSV/JSON

- Stream progress via SSE endpoint: `GET /runs/<run_id>/progress` (stream)

### Database Schema
- Existing `findings` table: Add `user_verdict` column (Confirmed / False Positive / Dismissed / Pending)
- New `corpora` table: corpus_name, path, judge_provider, created_at
- New `runs` table: corpus_id, started_at, completed_at, status, message_log

---

## Success Criteria

- [ ] Tabs are ordered left-to-right by workflow
- [ ] Users can create and manage persistent corpora
- [ ] Users can add files to existing corpora
- [ ] Process tab streams real-time progress
- [ ] Assertions, Contradictions, Definitions tabs are read-only exploration
- [ ] Action Items tab is the single source of truth for verdicts
- [ ] User verdicts persist and update in real-time
- [ ] Stats tab shows meaningful corpus/run metrics
- [ ] Export (CSV, JSON) works from Action Items tab
- [ ] Old ingest, documents, stats tabs are removed or consolidated
- [ ] No visual clutter; UI is compact and professional

---

## Migration from Current UI

**Tabs to remove**: Documents (content moved to Assertions), old Stats/Ingest (consolidated)  
**Tabs to keep**: Contradictions, Definitions (renamed from structure, kept as exploration)  
**Tabs to create**: Ingest (new, combined upload + corpus + judge), Process (new, streaming), Assertions (renamed from Statements), Action Items (new, unified), Stats (redesigned)

**Backward compatibility**: No API changes; this is purely UI reorganization. Existing DB schema gains `user_verdict` and `corpus` tables, but all existing data is preserved.

---

## Future Extensions (Out of Scope)

- Multi-party contradictions visualization (separate tab if detected)
- Definition harmonization suggestions (AI-assisted term clustering)
- Corpus comparison (run two models on same corpus, compare verdicts)
- Annotation sharing (team review of verdicts)
