# Security Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the attack surface for sensitive corporate document data loaded by the contradiction checker.

**Architecture:** The tool is a localhost FastAPI + HTMX app with no auth, SQLite storage, and cloud AI judge providers. The primary data-security concern is that document content (extracted assertions) leaves the machine to external AI APIs with no disclosure, no classification controls, and no deletion path.

**Tech Stack:** Python 3.11+, FastAPI, SQLite (via `AssertionStore`), Jinja2 templates, IBM Plex fonts, HTMX.

**Source of full analysis:** This session (2026-05-17). Full narrative in session transcript. The seven issues below are ordered by remediation effort × impact.

---

## Issue Register

| # | Title | Effort | Impact | Status |
|---|-------|--------|--------|--------|
| 1 | Corpus path re-derived from corpus_id, not trusted from DB | Low | Medium | Open |
| 2 | Server-side file size + extension validation on upload | Low | Medium | Open |
| 3 | Generic HTTP 500 messages (no internal detail leak) | Low | Low | Open |
| 4 | Pre-run disclosure: name the external provider before first run | Low | High | Open |
| 5 | Bind to 127.0.0.1 by default + startup session token | Medium | High | Open |
| 6 | Corpus/data deletion API + filesystem permission docs | Medium | Medium | Open |
| 7 | Data classification field + per-corpus provider restriction | High | High | Open |

---

## Task 1: Fix corpus_path trust boundary

**Files:**
- Modify: `consistency_checker/web/api/corpora.py:286,349`

The `corpus_path` column is stored as an absolute OS path and then retrieved and used directly for `iterdir()` / `exists()` calls. If the DB row is tampered with, any filesystem path can be traversed.

**Fix:** Reconstruct path from `corpus_id` + `config.data_dir` — never use the stored value for opening file handles.

- [ ] **Step 1: Locate all uses of stored corpus_path for filesystem ops**

  ```bash
  grep -n "corpus_path" consistency_checker/web/api/corpora.py
  ```

  Expected hits: line ~286 (`get_corpus`) and ~349 (`list_documents`).

- [ ] **Step 2: Write a failing test**

  In `tests/web/api/test_corpora.py` (or create it):
  ```python
  def test_get_corpus_ignores_stored_path(client, db_with_tampered_path):
      """corpus_path column being wrong should not affect filesystem read."""
      # Store a corpus with a tampered corpus_path in DB
      # GET /api/corpora/{corpus_id} should still work, reading from data_dir/corpora/{corpus_id}
      ...
  ```

  Run: `uv run pytest tests/web/api/test_corpora.py -v`
  Expected: FAIL (current code uses stored path).

- [ ] **Step 3: Replace stored-path usage in `get_corpus`**

  Current (line ~286):
  ```python
  corpus_dir = Path(corpus.corpus_path)
  ```
  Replace with:
  ```python
  corpus_dir = config.data_dir / "corpora" / corpus.corpus_id
  ```

- [ ] **Step 4: Replace stored-path usage in `list_documents`**

  Current (line ~349):
  ```python
  corpus_path = Path(row[0])
  ```
  Replace with:
  ```python
  corpus_path = config.data_dir / "corpora" / corpus_id
  ```
  (The `row[0]` was `corpus_path`; now we only need `corpus_id` which is the route param.)

  Update the query to drop `corpus_path` from the SELECT:
  ```python
  row = conn.execute("SELECT 1 FROM corpora WHERE corpus_id = ?", (corpus_id,)).fetchone()
  ```

- [ ] **Step 5: Run tests and commit**

  ```bash
  uv run pytest tests/web/api/ -v
  uv run ruff check . && uv run mypy consistency_checker
  git add consistency_checker/web/api/corpora.py tests/web/api/test_corpora.py
  git commit -m "fix: reconstruct corpus_path from corpus_id, never trust stored DB path"
  ```

---

## Task 2: Server-side upload validation

**Files:**
- Modify: `consistency_checker/web/app.py:925-982` (`post_uploads`)

Current code reads the full file into memory with no size check, and accepts any MIME/extension the browser sends.

- [ ] **Step 1: Add MAX_UPLOAD_BYTES constant**

  At the top of `app.py` (after the imports, before `create_app`):
  ```python
  MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB per file
  _ALLOWED_EXTENSIONS = frozenset({".txt", ".md", ".pdf", ".docx"})
  ```

- [ ] **Step 2: Write a failing test**

  ```python
  def test_upload_rejects_oversized_file(client, tmp_path):
      big = tmp_path / "big.txt"
      big.write_bytes(b"x" * (MAX_UPLOAD_BYTES + 1))
      with open(big, "rb") as f:
          resp = client.post("/uploads", files=[("files", ("big.txt", f, "text/plain"))])
      assert resp.status_code == 413

  def test_upload_rejects_bad_extension(client, tmp_path):
      exe = tmp_path / "bad.exe"
      exe.write_bytes(b"MZ")
      with open(exe, "rb") as f:
          resp = client.post("/uploads", files=[("files", ("bad.exe", f, "application/octet-stream"))])
      assert resp.status_code == 400
  ```

- [ ] **Step 3: Add validation in `post_uploads`**

  Before the `target.write_bytes(...)` line:
  ```python
  ext = Path(file.filename or "").suffix.lower()
  if ext not in _ALLOWED_EXTENSIONS:
      raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext!r}")

  content = await file.read(MAX_UPLOAD_BYTES + 1)
  if len(content) > MAX_UPLOAD_BYTES:
      raise HTTPException(status_code=413, detail="File too large (max 100 MB)")
  target.write_bytes(content)
  ```

  Remove the old `target.write_bytes(await file.read())`.

- [ ] **Step 4: Run tests and commit**

  ```bash
  uv run pytest tests/ -v -m "not slow and not live"
  uv run ruff check . && uv run mypy consistency_checker
  git add consistency_checker/web/app.py tests/
  git commit -m "fix: server-side upload size cap (100 MB) and extension allowlist"
  ```

---

## Task 3: Sanitize HTTP 500 error messages

**Files:**
- Modify: `consistency_checker/web/api/corpora.py` (4 `detail=f"..."` occurrences)
- Modify: `consistency_checker/web/api/runs.py` (5 `detail=f"..."` occurrences)

Raw Python exception messages (filesystem paths, SQLite internals) are returned in `detail` fields of HTTP 500 responses.

- [ ] **Step 1: Find all offending patterns**

  ```bash
  grep -n 'detail=f"Failed to' consistency_checker/web/api/corpora.py consistency_checker/web/api/runs.py
  ```

- [ ] **Step 2: Replace each with a static string**

  Pattern: replace `detail=f"Failed to {verb} {noun}: {e}"` with `detail=f"Failed to {verb} {noun}"`.

  The `_log.error(...)` call above each one already captures the full exception — no information is lost.

  Example (corpora.py line ~239):
  ```python
  # Before:
  raise HTTPException(status_code=500, detail=f"Failed to list corpora: {e}") from e
  # After:
  raise HTTPException(status_code=500, detail="Failed to list corpora") from e
  ```

- [ ] **Step 3: Run tests and commit**

  ```bash
  uv run pytest tests/ -v -m "not slow and not live"
  uv run ruff check .
  git add consistency_checker/web/api/corpora.py consistency_checker/web/api/runs.py
  git commit -m "fix: strip exception details from HTTP 500 responses"
  ```

---

## Task 4: Pre-run provider disclosure

**Files:**
- Modify: `consistency_checker/web/templates/cc_ingest.html`
- Modify: `consistency_checker/web/static/cc_style.css`

Before `Start Processing →` triggers, the user should see which external API will receive their document content.

- [ ] **Step 1: Add provider disclosure text to the Ingest template**

  In `cc_ingest.html`, inside the Judge Provider section (after the judge toggle buttons), add:
  ```html
  <p class="cc-ingest-disclosure" id="cc-provider-disclosure">
    <!-- Updated by JS when provider changes -->
  </p>
  ```

  In the JS `ccSelectJudge` function, update the disclosure text:
  ```js
  var PROVIDER_NOTES = {
    moonshot: 'Document content will be sent to Moonshot AI (Kimi) servers in China.',
    anthropic: 'Document content will be sent to Anthropic servers (US).',
  };
  window.ccSelectJudge = function (btn) {
    // ... existing code ...
    var disc = document.getElementById('cc-provider-disclosure');
    if (disc) disc.textContent = PROVIDER_NOTES[_judge] || '';
  };
  // Also set on load:
  (function () {
    var disc = document.getElementById('cc-provider-disclosure');
    if (disc) disc.textContent = PROVIDER_NOTES['moonshot'];
  }());
  ```

- [ ] **Step 2: Style the disclosure**

  In `cc_style.css`:
  ```css
  .cc-ingest-disclosure {
    font-size: 0.78rem;
    color: var(--cc-accent);
    margin-top: 0.25rem;
  }
  ```

- [ ] **Step 3: Commit**

  ```bash
  git add consistency_checker/web/templates/cc_ingest.html consistency_checker/web/static/cc_style.css
  git commit -m "feat: show external provider disclosure before processing"
  ```

---

## Task 5: Localhost bind + startup session token

**Files:**
- Modify: `consistency_checker/cli/main.py` (or wherever the uvicorn `run()` call lives)
- Create: middleware in `consistency_checker/web/app.py`

- [ ] **Step 1: Find the uvicorn launch call**

  ```bash
  grep -rn "uvicorn" consistency_checker/
  ```

- [ ] **Step 2: Add `host="127.0.0.1"` default**

  In the CLI `serve` command, ensure the default host is `127.0.0.1`:
  ```python
  @app.command()
  def serve(host: str = "127.0.0.1", port: int = 8000): ...
  ```

- [ ] **Step 3: Add IP-check middleware**

  In `create_app()` in `app.py`, add before any route:
  ```python
  from fastapi import Request
  from starlette.middleware.base import BaseHTTPMiddleware

  class LocalOnlyMiddleware(BaseHTTPMiddleware):
      async def dispatch(self, request: Request, call_next):
          client = request.client
          if client and client.host not in ("127.0.0.1", "::1"):
              from fastapi.responses import JSONResponse
              return JSONResponse({"detail": "Remote access not permitted"}, status_code=403)
          return await call_next(request)

  app.add_middleware(LocalOnlyMiddleware)
  ```

- [ ] **Step 4: Write test and commit**

  ```bash
  uv run pytest tests/ -v -m "not slow and not live"
  git add consistency_checker/cli/main.py consistency_checker/web/app.py
  git commit -m "fix: restrict web UI to localhost only"
  ```

---

## Task 6: Corpus deletion API + data directory docs

**Files:**
- Modify: `consistency_checker/web/api/corpora.py` (add DELETE endpoint)
- Modify: `README.md` or `docs/` (data sensitivity note)

- [ ] **Step 1: Add `DELETE /api/corpora/{corpus_id}`**

  ```python
  @router.delete("/{corpus_id}", status_code=204)
  def delete_corpus(request: Request, corpus_id: str) -> None:
      config = request.app.state.config
      store = AssertionStore(config.db_path)
      store.migrate()
      try:
          conn = store._conn
          row = conn.execute("SELECT corpus_id FROM corpora WHERE corpus_id = ?", (corpus_id,)).fetchone()
          if row is None:
              raise HTTPException(status_code=404, detail="corpus not found")
          # Cascade delete in DB (runs → corpora FK cascade)
          conn.execute("DELETE FROM corpora WHERE corpus_id = ?", (corpus_id,))
          conn.commit()
      finally:
          store.close()
      # Remove filesystem directory
      corpus_dir = config.data_dir / "corpora" / corpus_id
      import shutil
      shutil.rmtree(corpus_dir, ignore_errors=True)
  ```

- [ ] **Step 2: Add data sensitivity note to README or CLAUDE.md**

  Add a section:
  ```
  ## Data sensitivity
  `data/assertions.db` contains all assertions extracted from processed documents.
  It should be treated with the same confidentiality as the source documents.
  Recommended: `chmod 700 data/` after first run.
  To delete a corpus and all its extracted data: DELETE /api/corpora/{corpus_id}.
  ```

- [ ] **Step 3: Run tests and commit**

  ```bash
  uv run pytest tests/ -v -m "not slow and not live"
  git add consistency_checker/web/api/corpora.py CLAUDE.md
  git commit -m "feat: add corpus deletion endpoint; document data sensitivity"
  ```

---

## Verification

After all tasks:

```bash
uv run pytest -m "not live and not slow" -v
uv run ruff check .
uv run ruff format --check .
uv run mypy consistency_checker
```

Task 7 (data classification field) is deferred — it requires schema changes and UI work. File a GitHub issue for it.
