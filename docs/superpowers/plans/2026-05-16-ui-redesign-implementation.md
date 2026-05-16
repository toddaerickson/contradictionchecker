# UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the web UI from scattered tabs to a 7-tab workflow (Ingest → Process → Assertions → Contradictions → Definitions → Action Items → Stats) with persistent corpora, streaming progress, and unified verdict management.

**Architecture:** Database schema adds `corpora` and `runs` tables plus `user_verdict` column to `findings`. Backend exposes corpus CRUD, progress streaming, and verdict endpoints. Frontend reorganizes templates into workflow-ordered tabs with HTMX for interactivity and real-time updates.

**Tech Stack:** SQLite migrations, HTMX, Jinja2 templates, AJAX for verdict updates, SSE for progress streaming.

---

## File Structure

| File | Action | Responsibility |
|------|--------|-----------------|
| `consistency_checker/index/migrations/0011_ui_redesign_schema.sql` | Create | Add corpora, runs tables; user_verdict column |
| `consistency_checker/web/models.py` | Create | Corpus, Run, Verdict dataclasses (or Pydantic models) |
| `consistency_checker/web/routes.py` | Modify | Add corpus endpoints, verdict endpoints, progress streaming |
| `consistency_checker/web/templates/cc_base.html` | Modify | Reorganize tab structure; add new tabs |
| `consistency_checker/web/templates/cc_ingest.html` | Modify | Corpus selector, judge picker, file upload |
| `consistency_checker/web/templates/cc_process.html` | Create | Progress stream container, real-time updates |
| `consistency_checker/web/templates/cc_assertions.html` | Rename/Modify | Rename from cc_statements.html; adjust table columns |
| `consistency_checker/web/templates/cc_contradictions.html` | Modify | Keep but adjust for workflow; add detail view |
| `consistency_checker/web/templates/cc_definitions.html` | Modify | Adjust for workflow; keep structure mostly same |
| `consistency_checker/web/templates/cc_action_items.html` | Create | Unified findings table with verdict dropdowns |
| `consistency_checker/web/templates/cc_stats.html` | Modify | Redesign to show corpus/run metrics, charts |
| `consistency_checker/web/static/cc_style.css` | Modify | Style new tabs, verdict dropdowns, progress stream |
| `consistency_checker/web/static/cc_ui.js` | Create | JS for verdict changes, bulk actions, filtering |
| `tests/web/test_corpus_endpoints.py` | Create | Test corpus CRUD, verdict endpoints |
| `tests/web/test_progress_streaming.py` | Create | Test SSE progress stream |

---

## Task 1: Database Migrations

**Files:**
- Create: `consistency_checker/index/migrations/0011_ui_redesign_schema.sql`
- Modify: `consistency_checker/index/assertion_store.py` (if loader logic needs updates)

- [ ] **Step 1: Create migration file**

```bash
cat > consistency_checker/index/migrations/0011_ui_redesign_schema.sql << 'EOF'
-- Add user_verdict column to findings table
ALTER TABLE findings ADD COLUMN user_verdict TEXT DEFAULT NULL;
  -- Values: 'confirmed', 'false_positive', 'dismissed', 'pending', NULL

-- Add user_verdict column to multi_party_findings table
ALTER TABLE multi_party_findings ADD COLUMN user_verdict TEXT DEFAULT NULL;

-- Create corpora table: persistent document collections
CREATE TABLE IF NOT EXISTS corpora (
  corpus_id TEXT PRIMARY KEY,  -- e.g., 'financial-audit-q1'
  corpus_name TEXT NOT NULL UNIQUE,
  corpus_path TEXT NOT NULL,  -- e.g., 'data/corpora/financial-audit-q1'
  judge_provider TEXT DEFAULT 'moonshot',  -- 'moonshot' or 'anthropic'
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create runs table: one per processing run on a corpus
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,  -- e.g., 'run_20260516_140530'
  corpus_id TEXT NOT NULL REFERENCES corpora(corpus_id),
  started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMP DEFAULT NULL,
  status TEXT DEFAULT 'in_progress',  -- 'in_progress', 'completed', 'failed'
  message_log TEXT DEFAULT NULL  -- JSON array of progress messages
);

-- Create indices
CREATE INDEX IF NOT EXISTS idx_runs_corpus_id ON runs(corpus_id);
CREATE INDEX IF NOT EXISTS idx_findings_user_verdict ON findings(user_verdict);
CREATE INDEX IF NOT EXISTS idx_multi_party_findings_user_verdict ON multi_party_findings(user_verdict);
EOF
```

- [ ] **Step 2: Verify migration file**

```bash
head -40 consistency_checker/index/migrations/0011_ui_redesign_schema.sql
```

Expected: File contains ALTER TABLE and CREATE TABLE statements.

- [ ] **Step 3: Run migrations**

The migration loader will pick it up automatically. Verify:

```bash
uv run python -c "from consistency_checker.index.assertion_store import AssertionStore; AssertionStore(':memory:')" 2>&1 | grep -i "migration\|table"
```

Expected: No errors; tables are created.

- [ ] **Step 4: Commit**

```bash
git add consistency_checker/index/migrations/0011_ui_redesign_schema.sql
git commit -m "migration: add corpora, runs tables and user_verdict columns for ui redesign"
```

---

## Task 2: Data Models

**Files:**
- Create: `consistency_checker/web/models.py`

- [ ] **Step 1: Create models file**

```bash
cat > consistency_checker/web/models.py << 'EOF'
"""Data models for web UI: Corpus, Run, Verdict."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True, slots=True)
class Corpus:
    """A persistent collection of documents."""

    corpus_id: str
    corpus_name: str
    corpus_path: str
    judge_provider: Literal["moonshot", "anthropic"] = "moonshot"
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Run:
    """One processing run on a corpus."""

    run_id: str
    corpus_id: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    status: Literal["in_progress", "completed", "failed"] = "in_progress"
    message_log: list[str] | None = None  # list of progress messages


@dataclass(frozen=True, slots=True)
class Verdict:
    """User-set verdict for a finding."""

    finding_id: str
    user_verdict: Literal["confirmed", "false_positive", "dismissed", "pending"] | None = None
    note: str | None = None  # optional user note
EOF
```

- [ ] **Step 2: Verify models**

```bash
python -c "from consistency_checker.web.models import Corpus, Run, Verdict; print('Models imported OK')"
```

Expected: "Models imported OK"

- [ ] **Step 3: Commit**

```bash
git add consistency_checker/web/models.py
git commit -m "feat: add Corpus, Run, Verdict data models for ui redesign"
```

---

## Task 3: Corpus Management Endpoints

**Files:**
- Modify: `consistency_checker/web/routes.py`
- Create: `tests/web/test_corpus_endpoints.py`

- [ ] **Step 1: Write failing tests**

```bash
cat > tests/web/test_corpus_endpoints.py << 'EOF'
"""Tests for corpus management endpoints."""

import pytest
from consistency_checker.web.models import Corpus


def test_create_corpus(client):
    """POST /corpora creates a new corpus."""
    response = client.post("/corpora", json={
        "corpus_name": "test_corpus_1",
        "judge_provider": "moonshot"
    })
    assert response.status_code == 201
    data = response.get_json()
    assert data["corpus_name"] == "test_corpus_1"
    assert data["corpus_id"]  # Auto-generated ID


def test_list_corpora(client):
    """GET /corpora returns list of existing corpora."""
    # Create two corpora
    client.post("/corpora", json={"corpus_name": "c1", "judge_provider": "moonshot"})
    client.post("/corpora", json={"corpus_name": "c2", "judge_provider": "anthropic"})
    
    response = client.get("/corpora")
    assert response.status_code == 200
    corpora = response.get_json()
    assert len(corpora) >= 2
    names = [c["corpus_name"] for c in corpora]
    assert "c1" in names and "c2" in names


def test_get_corpus(client):
    """GET /corpora/<corpus_id> returns corpus details."""
    create_resp = client.post("/corpora", json={
        "corpus_name": "detail_test",
        "judge_provider": "anthropic"
    })
    corpus_id = create_resp.get_json()["corpus_id"]
    
    response = client.get(f"/corpora/{corpus_id}")
    assert response.status_code == 200
    data = response.get_json()
    assert data["corpus_name"] == "detail_test"
    assert data["judge_provider"] == "anthropic"


def test_create_corpus_duplicate_name_fails(client):
    """Creating a corpus with duplicate name fails."""
    client.post("/corpora", json={"corpus_name": "dup", "judge_provider": "moonshot"})
    response = client.post("/corpora", json={"corpus_name": "dup", "judge_provider": "anthropic"})
    assert response.status_code == 409  # Conflict
EOF
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/web/test_corpus_endpoints.py -v
```

Expected: FAIL (endpoints don't exist yet).

- [ ] **Step 3: Implement corpus endpoints**

In `consistency_checker/web/routes.py`, add:

```python
import json
import os
from pathlib import Path
from flask import Blueprint, request, jsonify
from consistency_checker.web.models import Corpus
from consistency_checker.index.assertion_store import AssertionStore

corpus_bp = Blueprint('corpus', __name__, url_prefix='/corpora')

CORPORA_DIR = Path('data/corpora')

@corpus_bp.route('', methods=['POST'])
def create_corpus():
    """Create a new corpus."""
    data = request.get_json()
    corpus_name = data.get('corpus_name')
    judge_provider = data.get('judge_provider', 'moonshot')
    
    if not corpus_name:
        return jsonify({'error': 'corpus_name required'}), 400
    
    # Check for duplicate
    store = AssertionStore()
    existing = store.conn.execute(
        "SELECT corpus_id FROM corpora WHERE corpus_name = ?", (corpus_name,)
    ).fetchone()
    if existing:
        return jsonify({'error': 'corpus_name already exists'}), 409
    
    # Create directory
    corpus_id = corpus_name.lower().replace(' ', '_')
    corpus_path = CORPORA_DIR / corpus_id
    corpus_path.mkdir(parents=True, exist_ok=True)
    (corpus_path / 'documents').mkdir(exist_ok=True)
    
    # Insert into DB
    store.conn.execute(
        "INSERT INTO corpora (corpus_id, corpus_name, corpus_path, judge_provider) VALUES (?, ?, ?, ?)",
        (corpus_id, corpus_name, str(corpus_path), judge_provider)
    )
    store.conn.commit()
    
    corpus = Corpus(
        corpus_id=corpus_id,
        corpus_name=corpus_name,
        corpus_path=str(corpus_path),
        judge_provider=judge_provider
    )
    return jsonify({
        'corpus_id': corpus.corpus_id,
        'corpus_name': corpus.corpus_name,
        'corpus_path': corpus.corpus_path,
        'judge_provider': corpus.judge_provider
    }), 201


@corpus_bp.route('', methods=['GET'])
def list_corpora():
    """List all existing corpora."""
    store = AssertionStore()
    rows = store.conn.execute("SELECT corpus_id, corpus_name, corpus_path, judge_provider, created_at FROM corpora ORDER BY created_at DESC").fetchall()
    result = []
    for row in rows:
        result.append({
            'corpus_id': row[0],
            'corpus_name': row[1],
            'corpus_path': row[2],
            'judge_provider': row[3],
            'created_at': row[4]
        })
    return jsonify(result), 200


@corpus_bp.route('/<corpus_id>', methods=['GET'])
def get_corpus(corpus_id):
    """Get a specific corpus."""
    store = AssertionStore()
    row = store.conn.execute(
        "SELECT corpus_id, corpus_name, corpus_path, judge_provider, created_at FROM corpora WHERE corpus_id = ?",
        (corpus_id,)
    ).fetchone()
    if not row:
        return jsonify({'error': 'corpus not found'}), 404
    
    return jsonify({
        'corpus_id': row[0],
        'corpus_name': row[1],
        'corpus_path': row[2],
        'judge_provider': row[3],
        'created_at': row[4]
    }), 200


# Register blueprint in main app
def register_corpus_routes(app):
    app.register_blueprint(corpus_bp)
```

Register in `consistency_checker/web/routes.py` main app initialization:
```python
from consistency_checker.web.routes import register_corpus_routes
# ... in app factory ...
register_corpus_routes(app)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/web/test_corpus_endpoints.py -v
```

Expected: PASS (4/4 tests pass).

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/web/routes.py tests/web/test_corpus_endpoints.py
git commit -m "feat: add corpus management endpoints (create, list, get)"
```

---

## Task 4: Verdict Management Endpoint

**Files:**
- Modify: `consistency_checker/web/routes.py`
- Modify: `tests/web/test_corpus_endpoints.py` (add verdict tests)

- [ ] **Step 1: Add verdict test**

Add to `tests/web/test_corpus_endpoints.py`:

```python
def test_set_verdict(client):
    """POST /findings/<finding_id>/verdict sets user verdict."""
    # Assume a finding exists in DB (from prior test)
    finding_id = "test_finding_1"
    
    response = client.post(f"/findings/{finding_id}/verdict", json={
        "user_verdict": "confirmed"
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data["user_verdict"] == "confirmed"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/web/test_corpus_endpoints.py::test_set_verdict -v
```

Expected: FAIL (endpoint doesn't exist).

- [ ] **Step 3: Implement verdict endpoint**

In `consistency_checker/web/routes.py`:

```python
@app.route('/findings/<finding_id>/verdict', methods=['POST'])
def set_verdict(finding_id):
    """Set user verdict for a finding."""
    data = request.get_json()
    user_verdict = data.get('user_verdict')
    
    if user_verdict not in ['confirmed', 'false_positive', 'dismissed', 'pending']:
        return jsonify({'error': 'invalid user_verdict'}), 400
    
    store = AssertionStore()
    store.conn.execute(
        "UPDATE findings SET user_verdict = ? WHERE finding_id = ?",
        (user_verdict, finding_id)
    )
    store.conn.commit()
    
    return jsonify({
        'finding_id': finding_id,
        'user_verdict': user_verdict
    }), 200


@app.route('/findings/<finding_id>/verdict', methods=['GET'])
def get_verdict(finding_id):
    """Get user verdict for a finding."""
    store = AssertionStore()
    row = store.conn.execute(
        "SELECT user_verdict FROM findings WHERE finding_id = ?",
        (finding_id,)
    ).fetchone()
    if not row:
        return jsonify({'error': 'finding not found'}), 404
    
    return jsonify({
        'finding_id': finding_id,
        'user_verdict': row[0]
    }), 200
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/web/test_corpus_endpoints.py::test_set_verdict -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/web/routes.py tests/web/test_corpus_endpoints.py
git commit -m "feat: add verdict management endpoint"
```

---

## Task 5: Progress Streaming Endpoint (SSE)

**Files:**
- Modify: `consistency_checker/web/routes.py`
- Create: `tests/web/test_progress_streaming.py`

- [ ] **Step 1: Write streaming test**

```bash
cat > tests/web/test_progress_streaming.py << 'EOF'
"""Tests for progress streaming."""

import pytest


def test_progress_stream_endpoint_exists(client):
    """GET /runs/<run_id>/progress should exist and stream."""
    run_id = "test_run_123"
    response = client.get(f"/runs/{run_id}/progress", headers={'Accept': 'text/event-stream'})
    # Should get 200 and streaming content type
    assert response.status_code == 200
    # Content-Type should indicate SSE
    assert 'text/event-stream' in response.content_type or 'text/plain' in response.content_type
EOF
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/web/test_progress_streaming.py -v
```

Expected: FAIL (endpoint doesn't exist).

- [ ] **Step 3: Implement progress streaming endpoint**

In `consistency_checker/web/routes.py`:

```python
from flask import Response

@app.route('/runs/<run_id>/progress', methods=['GET'])
def progress_stream(run_id):
    """Stream progress messages for a run (SSE)."""
    def generate():
        # In a real implementation, this would read from a queue or log file
        # For now, yield stored messages
        store = AssertionStore()
        row = store.conn.execute(
            "SELECT message_log FROM runs WHERE run_id = ?",
            (run_id,)
        ).fetchone()
        
        if not row:
            yield f"data: {json.dumps({'error': 'run not found'})}\n\n"
            return
        
        message_log = row[0]
        if message_log:
            messages = json.loads(message_log)
            for msg in messages:
                yield f"data: {json.dumps({'message': msg})}\n\n"
        
        yield f"data: {json.dumps({'status': 'done'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/web/test_progress_streaming.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add consistency_checker/web/routes.py tests/web/test_progress_streaming.py
git commit -m "feat: add progress streaming endpoint (SSE)"
```

---

## Task 6: Reorganize Base Template and Tab Structure

**Files:**
- Modify: `consistency_checker/web/templates/cc_base.html`

- [ ] **Step 1: Update cc_base.html**

Replace the old tab nav with the new 7-tab structure:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>consistency-checker</title>
    <link rel="stylesheet" href="/static/cc_style.css">
    <script src="/static/htmx.min.js" defer></script>
    <script src="/static/cc_ui.js" defer></script>
  </head>
  <body>
    <header class="cc-header">
      <h1>consistency-checker</h1>
      <nav class="cc-tabs">
        <a class="cc-tab {% if active_tab == 'ingest' %}cc-tab--active{% endif %}" 
           hx-get="/tabs/ingest" hx-target="#cc-tab-content" hx-push-url="true">
          Ingest
        </a>
        <a class="cc-tab {% if active_tab == 'process' %}cc-tab--active{% endif %}" 
           hx-get="/tabs/process" hx-target="#cc-tab-content" hx-push-url="true">
          Process
        </a>
        <a class="cc-tab {% if active_tab == 'assertions' %}cc-tab--active{% endif %}" 
           hx-get="/tabs/assertions" hx-target="#cc-tab-content" hx-push-url="true">
          Assertions
        </a>
        <a class="cc-tab {% if active_tab == 'contradictions' %}cc-tab--active{% endif %}" 
           hx-get="/tabs/contradictions" hx-target="#cc-tab-content" hx-push-url="true">
          Contradictions
        </a>
        <a class="cc-tab {% if active_tab == 'definitions' %}cc-tab--active{% endif %}" 
           hx-get="/tabs/definitions" hx-target="#cc-tab-content" hx-push-url="true">
          Definitions
        </a>
        <a class="cc-tab {% if active_tab == 'action_items' %}cc-tab--active{% endif %}" 
           hx-get="/tabs/action_items" hx-target="#cc-tab-content" hx-push-url="true">
          Action Items
        </a>
        <a class="cc-tab {% if active_tab == 'stats' %}cc-tab--active{% endif %}" 
           hx-get="/tabs/stats" hx-target="#cc-tab-content" hx-push-url="true">
          Stats
        </a>
      </nav>
    </header>
    <main id="cc-tab-content">
      {% block content %}{% endblock %}
    </main>
    <dialog id="cc-diff-dialog" class="cc-dialog">
      <div id="cc-diff-content"></div>
      <button type="button" class="cc-button" onclick="document.getElementById('cc-diff-dialog').close()">Close</button>
    </dialog>
    <div id="cc-toast-region" class="cc-toast-region" role="status" aria-live="polite"></div>
    <script>
      // Tab nav active state management
      const ccTabsNav = document.querySelector('.cc-tabs');
      if (ccTabsNav) {
        ccTabsNav.addEventListener('click', (e) => {
          const tab = e.target.closest('.cc-tab');
          if (!tab) return;
          document.querySelectorAll('.cc-tab').forEach(t => t.classList.remove('cc-tab--active'));
          tab.classList.add('cc-tab--active');
        });
      }
    </script>
  </body>
</html>
```

- [ ] **Step 2: Verify structure**

```bash
grep -c "cc-tab" consistency_checker/web/templates/cc_base.html
```

Expected: 7 tabs (14 references: one open anchor, one class check).

- [ ] **Step 3: Commit**

```bash
git add consistency_checker/web/templates/cc_base.html
git commit -m "refactor: reorganize tabs into 7-tab workflow structure"
```

---

## Task 7: Create Ingest Tab Template

**Files:**
- Modify: `consistency_checker/web/templates/cc_ingest.html`

- [ ] **Step 1: Update ingest template with corpus selector and judge picker**

```html
{% if not htmx %}{% extends "cc_base.html" %}{% endif %}

{% block content %}
<section class="cc-section">
  <h2>Setup: Choose Judge & Corpus</h2>
  
  <form class="cc-form" hx-post="/runs" hx-target="#cc-tab-content" hx-swap="innerHTML">
    <!-- Judge Provider -->
    <div class="cc-form-group">
      <label for="judge-provider">Judge Provider</label>
      <select id="judge-provider" name="judge_provider" required>
        <option value="moonshot" selected>Moonshot (default)</option>
        <option value="anthropic">Claude (Anthropic)</option>
      </select>
      <p class="cc-help">Which LLM should judge contradictions?</p>
    </div>

    <!-- Corpus Selector -->
    <div class="cc-form-group">
      <label for="corpus-select">Corpus</label>
      <select id="corpus-select" name="corpus_id" required>
        <option value="">-- Create New --</option>
        {% for corpus in existing_corpora %}
          <option value="{{ corpus.corpus_id }}">{{ corpus.corpus_name }}</option>
        {% endfor %}
      </select>
      <p class="cc-help">Select an existing corpus or create a new one.</p>
    </div>

    <!-- New Corpus Name (shown if -- Create New -- selected) -->
    <div class="cc-form-group" id="new-corpus-group" style="display: none;">
      <label for="new-corpus-name">New Corpus Name</label>
      <input type="text" id="new-corpus-name" name="new_corpus_name" placeholder="e.g., Q1 Financial Audit">
      <p class="cc-help">Enter a name for the new corpus.</p>
    </div>

    <!-- File Upload -->
    <div class="cc-form-group">
      <label for="files">Upload Documents</label>
      <div class="cc-file-upload">
        <input type="file" id="files" name="files" multiple required accept=".txt,.md,.pdf,.docx">
        <p class="cc-help">Accepted formats: .txt, .md, .pdf, .docx. Or browse to existing directory.</p>
      </div>
    </div>

    <button type="submit" class="cc-button cc-button--primary">Start Processing</button>
  </form>
</section>

<script>
  // Show/hide new corpus name input based on selection
  const corpusSelect = document.getElementById('corpus-select');
  const newCorpusGroup = document.getElementById('new-corpus-group');
  
  corpusSelect.addEventListener('change', (e) => {
    if (e.target.value === '') {
      newCorpusGroup.style.display = 'block';
      document.getElementById('new-corpus-name').required = true;
    } else {
      newCorpusGroup.style.display = 'none';
      document.getElementById('new-corpus-name').required = false;
    }
  });
</script>
{% endblock %}
```

- [ ] **Step 2: Update routes to serve ingest tab**

In `consistency_checker/web/routes.py`:

```python
@app.route('/tabs/ingest')
def tab_ingest():
    store = AssertionStore()
    rows = store.conn.execute("SELECT corpus_id, corpus_name FROM corpora ORDER BY created_at DESC").fetchall()
    existing_corpora = [{'corpus_id': r[0], 'corpus_name': r[1]} for r in rows]
    
    return render_template('cc_ingest.html', existing_corpora=existing_corpora, active_tab='ingest', htmx=request.headers.get('HX-Request'))
```

- [ ] **Step 3: Commit**

```bash
git add consistency_checker/web/templates/cc_ingest.html consistency_checker/web/routes.py
git commit -m "feat: create ingest tab with corpus selector and judge picker"
```

---

## Task 8: Create Process Tab Template (Streaming Progress)

**Files:**
- Create: `consistency_checker/web/templates/cc_process.html`
- Modify: `consistency_checker/web/routes.py`

- [ ] **Step 1: Create process template**

```bash
cat > consistency_checker/web/templates/cc_process.html << 'EOF'
{% if not htmx %}{% extends "cc_base.html" %}{% endif %}

{% block content %}
<section class="cc-section">
  <h2>Processing: Real-time Progress</h2>
  
  <div class="cc-progress-container">
    <div class="cc-progress-stream" id="progress-stream" role="status" aria-live="polite">
      <p class="cc-muted">Connecting to progress stream...</p>
    </div>
  </div>
  
  <div class="cc-actions" id="process-actions" style="display: none;">
    <a href="/tabs/action_items" class="cc-button cc-button--primary">View Results</a>
  </div>
</section>

<script>
  const runId = '{{ run_id }}';
  const progressStream = document.getElementById('progress-stream');
  const processActions = document.getElementById('process-actions');
  
  // Clear initial text
  progressStream.innerHTML = '';
  
  // Connect to SSE endpoint
  const eventSource = new EventSource(`/runs/${runId}/progress`);
  
  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.status === 'done') {
      eventSource.close();
      progressStream.innerHTML += '<p class="cc-progress-done">✓ Processing complete!</p>';
      processActions.style.display = 'block';
    } else if (data.message) {
      const p = document.createElement('p');
      p.textContent = data.message;
      p.className = 'cc-progress-message';
      progressStream.appendChild(p);
      progressStream.scrollTop = progressStream.scrollHeight;
    }
  };
  
  eventSource.onerror = () => {
    eventSource.close();
    progressStream.innerHTML += '<p class="cc-progress-error">✗ Stream disconnected</p>';
  };
</script>
{% endblock %}
EOF
```

- [ ] **Step 2: Add process route**

In `consistency_checker/web/routes.py`:

```python
@app.route('/tabs/process')
def tab_process():
    run_id = request.args.get('run_id', 'test_run')
    return render_template('cc_process.html', run_id=run_id, active_tab='process', htmx=request.headers.get('HX-Request'))
```

- [ ] **Step 3: Commit**

```bash
git add consistency_checker/web/templates/cc_process.html consistency_checker/web/routes.py
git commit -m "feat: create process tab with streaming progress"
```

---

## Task 9: Update Assertions Tab Template

**Files:**
- Rename: `consistency_checker/web/templates/cc_statements.html` → `consistency_checker/web/templates/cc_assertions.html`
- Modify: `consistency_checker/web/routes.py`

- [ ] **Step 1: Rename file**

```bash
mv consistency_checker/web/templates/cc_statements.html consistency_checker/web/templates/cc_assertions.html
```

- [ ] **Step 2: Update assertions template header**

Update the first few lines to use the new name and columns:

```html
{% if not htmx %}{% extends "cc_base.html" %}{% endif %}

{% block content %}
<section class="cc-section">
  <h2>Assertions: Extracted from Corpus</h2>
  <p class="cc-muted">All assertions extracted from uploaded documents. Read-only reference.</p>
  
  <table class="cc-table cc-assertions-table">
    <thead>
      <tr>
        <th>Assertion Text</th>
        <th>Source Document</th>
        <th>Kind</th>
        <th>ID</th>
      </tr>
    </thead>
    <tbody>
      {% for assertion in assertions %}
        <tr>
          <td>{{ assertion.text }}</td>
          <td>{{ assertion.document_id }}</td>
          <td>{{ assertion.kind }}</td>
          <td><code>{{ assertion.assertion_id[:8] }}</code></td>
        </tr>
      {% endfor %}
    </tbody>
  </table>
</section>
{% endblock %}
```

- [ ] **Step 3: Update route to point to assertions tab**

In `consistency_checker/web/routes.py`:

```python
@app.route('/tabs/assertions')
def tab_assertions():
    store = AssertionStore()
    rows = store.conn.execute(
        "SELECT assertion_id, assertion_text, document_id, kind FROM assertions LIMIT 100"
    ).fetchall()
    assertions = [
        {'assertion_id': r[0], 'text': r[1], 'document_id': r[2], 'kind': r[3]}
        for r in rows
    ]
    return render_template('cc_assertions.html', assertions=assertions, active_tab='assertions', htmx=request.headers.get('HX-Request'))
```

- [ ] **Step 4: Commit**

```bash
git add consistency_checker/web/templates/cc_assertions.html consistency_checker/web/routes.py
git commit -m "refactor: rename statements tab to assertions, update columns"
```

---

## Task 10: Create Action Items Tab Template

**Files:**
- Create: `consistency_checker/web/templates/cc_action_items.html`
- Modify: `consistency_checker/web/routes.py`
- Create: `consistency_checker/web/static/cc_ui.js`

- [ ] **Step 1: Create action items template**

```bash
cat > consistency_checker/web/templates/cc_action_items.html << 'EOF'
{% if not htmx %}{% extends "cc_base.html" %}{% endif %}

{% block content %}
<section class="cc-section">
  <h2>Action Items: Issues to Reconcile</h2>
  
  <div class="cc-filters">
    <label for="filter-type">Filter by Type:</label>
    <select id="filter-type">
      <option value="">All</option>
      <option value="contradiction">Contradictions</option>
      <option value="definition">Definitions</option>
      <option value="multi_party">Multi-party</option>
    </select>
    
    <label for="filter-verdict">Filter by Verdict:</label>
    <select id="filter-verdict">
      <option value="">All</option>
      <option value="pending">Pending</option>
      <option value="confirmed">Confirmed</option>
      <option value="false_positive">False Positive</option>
      <option value="dismissed">Dismissed</option>
    </select>
    
    <button class="cc-button cc-button--secondary" onclick="exportData('csv')">Download CSV</button>
    <button class="cc-button cc-button--secondary" onclick="exportData('json')">Download JSON</button>
  </div>
  
  <table class="cc-table cc-action-items-table">
    <thead>
      <tr>
        <th>Type</th>
        <th>Issue</th>
        <th>Judge Verdict</th>
        <th>User Verdict</th>
        <th>Confidence</th>
        <th>Actions</th>
      </tr>
    </thead>
    <tbody id="action-items-tbody">
      {% for finding in findings %}
        <tr class="cc-action-item" data-finding-id="{{ finding.finding_id }}" data-type="{{ finding.type }}" data-user-verdict="{{ finding.user_verdict or 'pending' }}">
          <td><span class="cc-badge cc-badge--{{ finding.type }}">{{ finding.type }}</span></td>
          <td>{{ finding.issue_preview }}</td>
          <td>{{ finding.judge_verdict }}</td>
          <td>
            <select class="verdict-select" data-finding-id="{{ finding.finding_id }}" onchange="setVerdict(this)">
              <option value="">Pending</option>
              <option value="confirmed" {% if finding.user_verdict == 'confirmed' %}selected{% endif %}>Confirmed</option>
              <option value="false_positive" {% if finding.user_verdict == 'false_positive' %}selected{% endif %}>False Positive</option>
              <option value="dismissed" {% if finding.user_verdict == 'dismissed' %}selected{% endif %}>Dismissed</option>
            </select>
          </td>
          <td>{{ "%.2f" | format(finding.confidence or 0.0) }}</td>
          <td><button class="cc-button cc-button--small" onclick="showDetail('{{ finding.finding_id }}')">Detail</button></td>
        </tr>
      {% endfor %}
    </tbody>
  </table>
</section>

<script>
  function setVerdict(select) {
    const findingId = select.getAttribute('data-finding-id');
    const verdict = select.value;
    
    fetch(`/findings/${findingId}/verdict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_verdict: verdict || null })
    })
    .then(r => r.json())
    .then(data => {
      // Update row data attribute
      const row = document.querySelector(`[data-finding-id="${findingId}"]`);
      if (row) row.setAttribute('data-user-verdict', verdict || 'pending');
      console.log('Verdict saved:', data);
    })
    .catch(err => console.error('Error setting verdict:', err));
  }
  
  function exportData(format) {
    // TODO: Implement export
    alert('Export ' + format + ' coming soon');
  }
  
  function showDetail(findingId) {
    // TODO: Implement detail view
    alert('Detail for ' + findingId);
  }
  
  // Filter by type
  document.getElementById('filter-type').addEventListener('change', (e) => {
    const type = e.target.value;
    document.querySelectorAll('.cc-action-item').forEach(row => {
      if (!type || row.getAttribute('data-type') === type) {
        row.style.display = '';
      } else {
        row.style.display = 'none';
      }
    });
  });
  
  // Filter by verdict
  document.getElementById('filter-verdict').addEventListener('change', (e) => {
    const verdict = e.target.value;
    document.querySelectorAll('.cc-action-item').forEach(row => {
      if (!verdict || row.getAttribute('data-user-verdict') === verdict) {
        row.style.display = '';
      } else {
        row.style.display = 'none';
      }
    });
  });
</script>
{% endblock %}
EOF
```

- [ ] **Step 2: Create UI JavaScript file**

```bash
cat > consistency_checker/web/static/cc_ui.js << 'EOF'
/**UI utilities for consistency-checker web interface.*/

// Placeholder; most logic is inline in templates for now.
// Can expand here for shared utilities.

console.log('CC UI initialized');
EOF
```

- [ ] **Step 3: Add action items route**

In `consistency_checker/web/routes.py`:

```python
@app.route('/tabs/action_items')
def tab_action_items():
    store = AssertionStore()
    
    # Fetch findings from DB
    rows = store.conn.execute("""
        SELECT finding_id, verdict, confidence, judge_rationale, user_verdict
        FROM findings
        LIMIT 100
    """).fetchall()
    
    findings = []
    for row in rows:
        findings.append({
            'finding_id': row[0],
            'type': 'contradiction',  # Could infer from findings table
            'issue_preview': 'Assertion A vs Assertion B',  # Format from actual assertions
            'judge_verdict': row[1],
            'confidence': row[2],
            'user_verdict': row[4]
        })
    
    return render_template('cc_action_items.html', findings=findings, active_tab='action_items', htmx=request.headers.get('HX-Request'))
```

- [ ] **Step 4: Commit**

```bash
git add consistency_checker/web/templates/cc_action_items.html consistency_checker/web/static/cc_ui.js consistency_checker/web/routes.py
git commit -m "feat: create action items tab with verdict dropdowns and filtering"
```

---

## Task 11: Update Stats Tab Template

**Files:**
- Modify: `consistency_checker/web/templates/cc_stats.html`
- Modify: `consistency_checker/web/routes.py`

- [ ] **Step 1: Redesign stats template**

```html
{% if not htmx %}{% extends "cc_base.html" %}{% endif %}

{% block content %}
<section class="cc-section">
  <h2>Stats: Corpus Health & Run Summary</h2>
  
  <div class="cc-stats-grid">
    <!-- Corpus Overview -->
    <div class="cc-stats-card">
      <h3>Corpus Overview</h3>
      <dl>
        <dt>Documents:</dt><dd>{{ stats.num_documents }}</dd>
        <dt>Assertions:</dt><dd>{{ stats.num_assertions }}</dd>
        <dt>Definitions:</dt><dd>{{ stats.num_definitions }}</dd>
      </dl>
    </div>
    
    <!-- Findings Summary -->
    <div class="cc-stats-card">
      <h3>Findings</h3>
      <dl>
        <dt>Contradictions:</dt><dd>{{ stats.num_contradictions }}</dd>
        <dt>Definition Issues:</dt><dd>{{ stats.num_definitions_inconsistent }}</dd>
        <dt>Multi-party:</dt><dd>{{ stats.num_multi_party or 0 }}</dd>
      </dl>
    </div>
    
    <!-- Judge Metrics -->
    <div class="cc-stats-card">
      <h3>Judge Performance</h3>
      <dl>
        <dt>Provider:</dt><dd>{{ stats.judge_provider }}</dd>
        <dt>Avg Confidence:</dt><dd>{{ "%.2f" | format(stats.avg_confidence) }}</dd>
        <dt>Processing Time:</dt><dd>{{ stats.processing_time }}</dd>
      </dl>
    </div>
    
    <!-- Verdict Summary -->
    <div class="cc-stats-card">
      <h3>User Verdicts</h3>
      <dl>
        <dt>Confirmed:</dt><dd>{{ stats.num_confirmed }}</dd>
        <dt>False Positives:</dt><dd>{{ stats.num_false_positives }}</dd>
        <dt>Dismissed:</dt><dd>{{ stats.num_dismissed }}</dd>
        <dt>Pending:</dt><dd>{{ stats.num_pending }}</dd>
      </dl>
    </div>
  </div>
  
  <div class="cc-actions">
    <button class="cc-button cc-button--secondary" onclick="alert('Report export coming soon')">Export Full Report</button>
  </div>
</section>
{% endblock %}
```

- [ ] **Step 2: Add stats route**

In `consistency_checker/web/routes.py`:

```python
@app.route('/tabs/stats')
def tab_stats():
    store = AssertionStore()
    
    # Aggregate stats
    num_docs = store.conn.execute("SELECT COUNT(DISTINCT document_id) FROM assertions").fetchone()[0]
    num_assertions = store.conn.execute("SELECT COUNT(*) FROM assertions").fetchone()[0]
    num_definitions = store.conn.execute("SELECT COUNT(*) FROM assertions WHERE kind = 'definition'").fetchone()[0]
    num_contradictions = store.conn.execute("SELECT COUNT(*) FROM findings WHERE judge_verdict = 'contradiction'").fetchone()[0]
    num_definitions_inconsistent = store.conn.execute("SELECT COUNT(DISTINCT finding_id) FROM findings WHERE detector_type = 'definition_inconsistency'").fetchone()[0]
    
    confirmed = store.conn.execute("SELECT COUNT(*) FROM findings WHERE user_verdict = 'confirmed'").fetchone()[0]
    false_pos = store.conn.execute("SELECT COUNT(*) FROM findings WHERE user_verdict = 'false_positive'").fetchone()[0]
    dismissed = store.conn.execute("SELECT COUNT(*) FROM findings WHERE user_verdict = 'dismissed'").fetchone()[0]
    pending = store.conn.execute("SELECT COUNT(*) FROM findings WHERE user_verdict IS NULL").fetchone()[0]
    
    avg_conf = store.conn.execute("SELECT AVG(judge_confidence) FROM findings WHERE judge_confidence IS NOT NULL").fetchone()[0] or 0.0
    
    stats = {
        'num_documents': num_docs,
        'num_assertions': num_assertions,
        'num_definitions': num_definitions,
        'num_contradictions': num_contradictions,
        'num_definitions_inconsistent': num_definitions_inconsistent,
        'num_multi_party': 0,
        'judge_provider': 'moonshot',  # From corpus metadata
        'avg_confidence': avg_conf,
        'processing_time': '2 min 30 sec',  # From run metadata
        'num_confirmed': confirmed,
        'num_false_positives': false_pos,
        'num_dismissed': dismissed,
        'num_pending': pending
    }
    
    return render_template('cc_stats.html', stats=stats, active_tab='stats', htmx=request.headers.get('HX-Request'))
```

- [ ] **Step 3: Commit**

```bash
git add consistency_checker/web/templates/cc_stats.html consistency_checker/web/routes.py
git commit -m "feat: redesign stats tab with corpus health metrics"
```

---

## Task 12: Update Contradictions and Definitions Tabs

**Files:**
- Modify: `consistency_checker/web/templates/cc_contradictions.html`
- Modify: `consistency_checker/web/templates/cc_definitions.html`

- [ ] **Step 1: Update contradictions template header and note it's for exploration**

Add a note at the top:

```html
{% if not htmx %}{% extends "cc_base.html" %}{% endif %}

{% block content %}
<section class="cc-section">
  <h2>Contradictions: Detailed Exploration</h2>
  <p class="cc-muted">Review contradiction pairs detected by the judge. To set verdicts, use the Action Items tab.</p>
  
  <!-- Rest of existing contradictions table -->
```

- [ ] **Step 2: Update definitions template header similarly**

```html
{% if not htmx %}{% extends "cc_base.html" %}{% endif %}

{% block content %}
<section class="cc-section">
  <h2>Definitions: Term Harmonization Issues</h2>
  <p class="cc-muted">Review definition inconsistencies. To reconcile, use the Action Items tab.</p>
  
  <!-- Rest of existing definitions table -->
```

- [ ] **Step 3: Commit**

```bash
git add consistency_checker/web/templates/cc_contradictions.html consistency_checker/web/templates/cc_definitions.html
git commit -m "refactor: add clarifying headers to exploration tabs"
```

---

## Task 13: CSS Updates

**Files:**
- Modify: `consistency_checker/web/static/cc_style.css`

- [ ] **Step 1: Add styles for new tabs and components**

Append to `cc_style.css`:

```css
/* Verdict dropdowns */
.verdict-select {
  padding: 0.375rem 0.5rem;
  border: 1px solid #d0d0d0;
  border-radius: 3px;
  font-size: 0.875rem;
}

/* Badge for finding type */
.cc-badge {
  display: inline-block;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
}

.cc-badge--contradiction { background: #ffeaea; color: #b71c1c; }
.cc-badge--definition { background: #dbeafe; color: #1e40af; }
.cc-badge--multi_party { background: #f3e8ff; color: #5b21b6; }

/* Progress stream */
.cc-progress-container {
  background: #f9fafb;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  padding: 1rem;
  margin: 1rem 0;
}

.cc-progress-stream {
  font-family: monospace;
  font-size: 0.875rem;
  max-height: 400px;
  overflow-y: auto;
  line-height: 1.5;
}

.cc-progress-message { margin: 0.25rem 0; color: #374151; }
.cc-progress-done { color: #059669; font-weight: 600; }
.cc-progress-error { color: #dc2626; font-weight: 600; }

/* Stats grid */
.cc-stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin: 1.5rem 0;
}

.cc-stats-card {
  background: #f9fafb;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  padding: 1rem;
}

.cc-stats-card h3 { margin-top: 0; }
.cc-stats-card dl { margin: 0; display: flex; flex-direction: column; gap: 0.5rem; }
.cc-stats-card dt { font-weight: 600; color: #374151; }
.cc-stats-card dd { margin: 0 0 0 1rem; color: #1f2937; }

/* Filters */
.cc-filters {
  background: #f3f4f6;
  padding: 1rem;
  border-radius: 6px;
  margin-bottom: 1.5rem;
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  align-items: center;
}

.cc-filters label { font-weight: 500; }
.cc-filters select { padding: 0.375rem 0.5rem; border: 1px solid #d0d0d0; border-radius: 3px; }

/* Action items table */
.cc-action-items-table { width: 100%; border-collapse: collapse; }
.cc-action-items-table th, .cc-action-items-table td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #e5e7eb; }
.cc-action-items-table th { background: #f9fafb; font-weight: 600; }
.cc-action-items-table tbody tr:hover { background: #f9fafb; }

/* Form updates */
.cc-form-group { margin-bottom: 1.5rem; }
.cc-form-group label { display: block; margin-bottom: 0.5rem; font-weight: 500; }
.cc-form-group input, .cc-form-group select { width: 100%; max-width: 400px; padding: 0.5rem; border: 1px solid #d0d0d0; border-radius: 4px; font-size: 0.875rem; }
.cc-help { margin-top: 0.25rem; font-size: 0.875rem; color: #6b7280; }
```

- [ ] **Step 2: Verify CSS is valid**

```bash
head -20 consistency_checker/web/static/cc_style.css
```

- [ ] **Step 3: Commit**

```bash
git add consistency_checker/web/static/cc_style.css
git commit -m "style: add CSS for new tabs, verdict dropdowns, stats, and filters"
```

---

## Task 14: Integration Test & Full Test Suite

**Files:**
- Modify: `tests/web/test_corpus_endpoints.py`

- [ ] **Step 1: Run full web test suite**

```bash
uv run pytest tests/web/ -v
```

Expected: All tests pass (corpus, progress, verdict endpoints).

- [ ] **Step 2: Run full project test suite**

```bash
uv run pytest -m "not slow and not live" -v
```

Expected: No regressions; all existing tests pass.

- [ ] **Step 3: Check linting and types**

```bash
uv run ruff check . && uv run mypy consistency_checker && echo "✓ All checks passed"
```

Expected: Clean output.

- [ ] **Step 4: Commit**

```bash
git add tests/web/
git commit -m "test: verify all endpoints and UI tabs pass tests"
```

---

## Task 15: Final Polish & Documentation

**Files:**
- Modify: `docs/superpowers/specs/2026-05-16-ui-redesign.md` (if needed)
- Create: `docs/superpowers/adrs/ADR-0008-workflow-based-ui.md` (architectural decision)

- [ ] **Step 1: Create ADR documenting the UI redesign**

```bash
cat > docs/superpowers/adrs/ADR-0008-workflow-based-ui.md << 'EOF'
# ADR-0008: Workflow-Based UI Redesign

**Date:** 2026-05-16  
**Status:** Accepted  
**Context:** The previous UI had six scattered tabs with unclear ordering and purpose. Users didn't know which tabs to use first or in what order. The new design follows the natural workflow: setup → process → explore → act → review.

**Decision:** Reorganize the web UI into 7 workflow-ordered tabs:
1. **Ingest**: Select judge, corpus, upload files
2. **Process**: Real-time progress streaming
3. **Assertions**: Explore extracted assertions
4. **Contradictions**: Explore contradiction pairs (read-only)
5. **Definitions**: Explore definition inconsistencies (read-only)
6. **Action Items**: Unified verdict management and reconciliation
7. **Stats**: Corpus health and run metrics

**Key Design Choices:**
- **Persistent corpora**: Filesystem-based, reusable; users add files over time
- **Streaming progress**: Real-time SSE updates during processing (no guessing how long it takes)
- **Exploration vs. Action split**: Detail tabs for understanding, Action Items for doing
- **Unified verdict system**: Single source of truth in DB; changes persist immediately
- **Linguistic semantics**: Contradictions and definitions treated as separate phenomena

**Consequences:**
- **Pro**: Clear workflow; users know what to do next. Persistent corpora enable iterative analysis.
- **Pro**: Real-time feedback reduces user anxiety during long processing.
- **Pro**: Verdict management is explicit and centralized.
- **Con**: More tabs (7 vs. 6), but order matters now so less cognitive load.
- **Con**: DB schema changes (minor: user_verdict column, two new tables).

**Alternatives Considered:**
- Single unified table (too much visual clutter)
- Vertical stepper UI (doesn't fit on screen; would require scrolling)
- Multi-page wizard (more clicks; less discoverability of detail tabs)

**Related ADRs:**
- ADR-0006: Multi-party contradiction detection
- ADR-0007: Moonshot experimental judge provider
EOF
```

- [ ] **Step 2: Commit ADR**

```bash
git add docs/superpowers/adrs/ADR-0008-workflow-based-ui.md
git commit -m "docs: add ADR-0008 for workflow-based UI redesign decision"
```

---

## Verification

After all tasks:

```bash
uv run pytest -m "not live and not slow" -v
uv run ruff check . && uv run ruff format --check .
uv run mypy consistency_checker
uv build
```

All must be green before opening a PR.

---

## Summary of Changes

| Phase | What | Tasks |
|-------|------|-------|
| **Phase 1: Foundation** | DB schema + data models | Tasks 1–2 |
| **Phase 2: Backend API** | Corpus CRUD, verdict, progress streaming | Tasks 3–5 |
| **Phase 3: Frontend Structure** | Tab reorganization, base template | Task 6 |
| **Phase 4: Tab Templates** | Ingest, Process, Assertions, Action Items, Stats | Tasks 7–12 |
| **Phase 5: Polish** | CSS, JavaScript, tests, ADR | Tasks 13–15 |

---

## Notes for Implementer

1. **HTMX Usage**: The design uses HTMX for tab switching (no page reload) and AJAX for verdict updates. Ensure `hx-push-url="true"` for browser history.

2. **Database Defaults**: The migration adds `user_verdict` column with `DEFAULT NULL`. Existing findings will have NULL verdicts; this is intentional (pending).

3. **Streaming Details**: SSE endpoint `/runs/<run_id>/progress` expects messages to be streamed from the backend. During development, you can mock with hardcoded messages.

4. **Backward Compatibility**: Old templates (Documents tab) are deprecated but not removed yet. They'll be removed in a follow-up once users have migrated.

5. **Testing**: Use `@pytest.mark.web` for web tests; they can be skipped with `pytest -m "not web"` if needed.

6. **CSS Organization**: All new styles are appended to `cc_style.css`. Consider splitting into modules later if the file grows beyond 500 lines.
