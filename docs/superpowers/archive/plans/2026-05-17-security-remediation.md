# Security Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the attack surface for sensitive corporate document data loaded by the contradiction checker.

**Architecture:** The tool is a localhost FastAPI + HTMX app with no auth, SQLite storage, and cloud AI judge providers. The primary data-security concern is that document content (extracted assertions) leaves the machine to external AI APIs with no disclosure, no classification controls, and no deletion path.

**Tech Stack:** Python 3.11+, FastAPI, SQLite (via `AssertionStore`), Jinja2 templates, IBM Plex fonts, HTMX.

**Source of full analysis:** Original session 2026-05-17. **Reconciled 2026-06-01** against current code + a fresh web-surface review (six findings). See the 2026-06-01 reconciliation note below before executing — most of the original "Open" tasks turned out to be already-done; the live work is mostly the new findings.

---

## 2026-06-01 reconciliation

The 2026-05-17 register was written before the ADR-0015 pairwise-opt-in change, the ADR-0016 cost ceiling, and the ADR-0017 single-page UI redesign. Re-verifying each item against `origin/main` (13d5d99):

| Orig # | Original title | Verified state on main | Disposition |
|---|---|---|---|
| 1 | corpus_path trust boundary | REST API already reconstructs path from id (`corpora.py:285/349`). **HTMX `/corpora/new` still stores raw `corpus_name`** (`app.py:599`). | **Re-scope → Task C** (HTMX path only) |
| 2 | upload size + extension validation | **Already done** (`app.py:252-253`, `2159-2161`). | Done. New gap = no file-count / total-request cap → **Task D** |
| 3 | generic HTTP 500 messages | api/* `{e}` leaks **already gone**. | Done. New gap = background-run `str(exc)` rendered in UI (`app.py:1806,1888` → `cc__stats_failed.html:8`) → **Task B** |
| 4 | pre-run provider disclosure | `cc_ingest.html` exists but live UI uses `cc_new_corpus_modal.html` / `cc_new_run_modal.html`. | **Deferred** (UI work, lower impact; provider already shown at corpus creation) |
| 5 | bind 127.0.0.1 + session token | host **already defaults** 127.0.0.1; no bind guard. | **Re-scope → Task A** (hard-block non-loopback bind; session token deferred) |
| 6 | corpus deletion API + docs | no DELETE endpoint; `assertion_store.delete_corpus()` exists. | **Deferred** (file follow-up issue) |
| 7 | data classification field | not started. | **Deferred** (file follow-up issue) |

New findings from the 2026-06-01 review:

- **Unauthenticated web mutation surface** — `serve` accepts any `--host`; no auth/CSRF. Covered by **Task A** (loopback hard-block fences the whole surface to localhost).
- **Legacy `/api/runs` mutation surface** — `runs.py` mutators (`append_message`, `update_run_status`, `start_run`) write the legacy `runs` table. **NOT cleanly removable:** `cc_process.html:158` still consumes the router's SSE `/progress` endpoint via `EventSource`. Deleting risks breaking the older process page. Loopback block (Task A) neutralizes the exposure; **removal deferred** to a dedicated cleanup that traces the `cc_process.html` dependency first.
- **ARCHITECTURE.md stale** — `docs/ARCHITECTURE.md:5` calls the tool a "Symmetric pairwise scan," but pairwise is opt-in / default-off since ADR-0015 → **Task E**.

**In-scope for this branch (`sec/web-surface-hardening`), highest-impact first:** A (loopback block), B (error sanitization), C (path hygiene), D (upload DoS caps), E (docs). One branch → one squash-merged PR.

**Deferred (file GitHub issues, do not implement here):** legacy `/api/runs` removal; provider disclosure UI (orig #4); corpus deletion API (orig #6); data classification (orig #7); `uploads/` GC (already roadmapped in `futureplans.md` v0.4).

---

## Updated Issue Register

| # | Title | Effort | Impact | Status |
|---|-------|--------|--------|--------|
| A | Hard-block non-loopback bind unless `--unsafe-no-auth` | Low | High | In scope |
| B | Sanitize background-run failure messages | Low | Medium | In scope |
| C | Path hygiene on HTMX `/corpora/new` | Low | Medium | In scope |
| D | Upload DoS caps (file-count + total bytes) | Low | Medium | In scope |
| E | Docs: un-stale ARCHITECTURE + strengthen README host warning | Low | Low | In scope |
| — | Legacy `/api/runs` removal | Medium | Low | Deferred (issue) |
| — | Provider disclosure UI | Low | Medium | Deferred (issue) |
| — | Corpus deletion API + data-sensitivity docs | Medium | Medium | Deferred (issue) |
| — | Data classification + per-corpus provider restriction | High | High | Deferred (issue) |

---

## Task A: Hard-block non-loopback bind

**Files:**
- Modify: `consistency_checker/cli/main.py` (`serve` command)
- Add: test in `tests/` (CLI-level)

The `serve` command passes `--host` straight to `uvicorn.run` (`main.py:724`). There is no auth/CSRF anywhere, so binding beyond loopback exposes file upload, corpus creation, check start, verdict mutation, and the legacy `/api/runs` mutators to the network.

**Fix:** Refuse to start on a non-loopback host unless the operator passes an explicit `--unsafe-no-auth` escape hatch. Keep the bind-time guard in a small pure helper so it is unit-testable without booting uvicorn.

- [ ] **Step 1: Add `--unsafe-no-auth` option to `serve`** (`bool`, default `False`, help text: "Allow binding to a non-loopback host. NO AUTH IS ADDED — only use on a trusted, isolated network.").

- [ ] **Step 2: Add a pure guard helper** (module-level in `main.py`), e.g.:
  ```python
  _LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

  def _assert_safe_bind(host: str, unsafe_no_auth: bool) -> None:
      if host in _LOOPBACK_HOSTS or unsafe_no_auth:
          return
      raise typer.BadParameter(
          f"Refusing to bind to non-loopback host {host!r}: the web UI has no "
          f"authentication. Re-run with --unsafe-no-auth only if this host is on a "
          f"trusted, isolated network.",
          param_hint="--host",
      )
  ```
  Call it in `serve` immediately after loading config, before `create_app`.

- [ ] **Step 3: Tests** — `_assert_safe_bind` returns for loopback hosts, returns when `unsafe_no_auth=True`, and raises `typer.BadParameter` for `"0.0.0.0"` / a LAN IP with `unsafe_no_auth=False`. (Prefer testing the helper directly; optionally a CliRunner test that `serve --host 0.0.0.0` exits non-zero.)

- [ ] **Step 4: Run gate + commit**
  ```bash
  uv run pytest -m "not slow and not live" -q && uv run ruff check . && uv run mypy consistency_checker
  git commit -am "fix(web): refuse non-loopback bind without --unsafe-no-auth (no auth on surface)"
  ```

---

## Task B: Sanitize background-run failure messages

**Files:**
- Modify: `consistency_checker/web/app.py` (the two `except Exception` handlers that store `error_message=str(exc)`; ~`1806` and ~`1888`)

The generic-exception handlers store `str(exc)` as the run's `error_message`, which `cc__stats_failed.html:8` renders verbatim. `str(exc)` can contain absolute filesystem paths and raw provider/SDK error text.

Note the `CostCeilingExceeded` branch already stores a clean, intentional message — leave it. Only the generic `except Exception` paths leak.

- [ ] **Step 1: Keep the full exception in logs** — confirm each site already does `_log.exception(...)` / `_log.error(...)` with the exception (it does). No information is lost from the operator's logs.

- [ ] **Step 2: Store a sanitized, user-facing message** instead of `str(exc)`, e.g. a generic line plus the exception *type* name only (no message body):
  ```python
  audit_logger.update_run_status(
      run_id, "failed",
      error_message="The check failed. See server logs for details.",
  )
  ```
  Apply at both generic handlers. (Including `type(exc).__name__` is acceptable if it adds triage value without a path/message body — implementer's judgment, but default to the generic line.)

- [ ] **Step 3: Test** — a run whose pipeline raises an exception carrying a fake absolute path (e.g. `RuntimeError("/home/secret/data.db locked")`) must result in a stored `error_message` that does **not** contain the path. Wire a failing checker via the existing fixture seams.

- [ ] **Step 4: Run gate + commit**
  ```bash
  uv run pytest -m "not slow and not live" -q && uv run ruff check . && uv run mypy consistency_checker
  git commit -am "fix(web): do not render raw exception text in run-failure UI"
  ```

---

## Task C: Path hygiene on HTMX `/corpora/new`

**Files:**
- Modify: `consistency_checker/web/app.py` (`post_corpora_new`, ~`542`; the `corpus_path` derivation at ~`599`)

The REST API derives `corpus_path` from a generated id and reconstructs it from `data_dir` for all filesystem ops. The HTMX `/corpora/new` handler instead builds `config.data_dir / "corpora" / corpus_name` from the **raw user-supplied name**. The char filter blocks slashes and Windows-reserved chars but not `..` or reserved dot-names, so the stored metadata path can be misleading/escaped.

`corpus_path` is metadata-only here today (no dir is created, no handle opened on it at creation time), so this is hardening, not an active traversal — but it should match the REST API's id/slug strategy.

- [ ] **Step 1: Study the REST strategy** in `consistency_checker/web/api/corpora.py` (`create_corpus`, ~`100-145`): how it generates `corpus_id`, builds `corpus_path = config.data_dir / "corpora" / corpus_id`, and (where applicable) `.resolve()`-checks it stays under `data_dir`. Reuse the same approach; do not invent a second scheme.

- [ ] **Step 2: In `post_corpora_new`, stop using the raw name for the path.** Generate / obtain the corpus id (the same way `get_or_create_corpus` / the REST path does) and set `corpus_path = config.data_dir / "corpora" / <corpus_id>`. Add a resolve-check that the path stays under `(config.data_dir / "corpora").resolve()`; reject otherwise. Keep the existing user-facing name validation (1–80 chars, invalid-char set) as-is — that's a separate concern.

- [ ] **Step 3: Test** — posting `corpus_name=".."` (and a name with a reserved dot-form) to `/corpora/new` must not produce a stored `corpus_path` that escapes `data_dir/corpora`. Assert the stored path resolves under the corpora root.

- [ ] **Step 4: Run gate + commit**
  ```bash
  uv run pytest -m "not slow and not live" -q && uv run ruff check . && uv run mypy consistency_checker
  git commit -am "fix(web): derive corpus path from id, never raw name, on /corpora/new"
  ```

---

## Task D: Upload DoS caps (file-count + total bytes)

**Files:**
- Modify: `consistency_checker/web/app.py` (`post_uploads`, ~`2134`; constants near `252-253`)

Per-file size (100 MB) and the extension allowlist already exist. The remaining DoS gap: a single request may carry an unbounded **number** of files and an unbounded **total** byte count. (Disk GC / retention is a separate roadmap item — `futureplans.md` v0.4 — do **not** add GC here.)

- [ ] **Step 1: Add two constants** next to `MAX_UPLOAD_BYTES`:
  ```python
  MAX_UPLOAD_FILES = 100                          # files per request
  MAX_UPLOAD_TOTAL_BYTES = 500 * 1024 * 1024      # 500 MB per request
  ```
  (Pick round, clearly-documented values; these are reasonable defaults — adjust only with a stated reason.)

- [ ] **Step 2: Enforce in `post_uploads`.** Reject with HTTP 413 if `len(files) > MAX_UPLOAD_FILES`. Maintain a running total of bytes read across the loop and reject with 413 once it exceeds `MAX_UPLOAD_TOTAL_BYTES`. Preserve the existing cleanup (`shutil.rmtree(upload_dir, ...)`) on rejection so partial writes don't linger.

- [ ] **Step 3: Tests** — (a) a request with `MAX_UPLOAD_FILES + 1` small files → 413; (b) several files whose combined size exceeds the total cap → 413 and the `upload_dir` is removed. Keep files small in tests (the caps can be monkeypatched/temporarily lowered if needed to stay fast and hermetic).

- [ ] **Step 4: Run gate + commit**
  ```bash
  uv run pytest -m "not slow and not live" -q && uv run ruff check . && uv run mypy consistency_checker
  git commit -am "fix(web): cap upload file-count and total request size"
  ```

---

## Task E: Docs — un-stale ARCHITECTURE + strengthen README host warning

**Files:**
- Modify: `docs/ARCHITECTURE.md` (line ~5)
- Modify: `README.md` (the localhost/no-auth known-issue line, ~148)

- [ ] **Step 1: Fix the stale detector description** in `docs/ARCHITECTURE.md:5`. It currently asserts a "Symmetric pairwise scan" as the operating mode. Reword so the **default** is accurate: pairwise NLI is **opt-in (`--pairwise`, default-off since ADR-0015)**; the default `check` runs the definition-inconsistency detector. Keep it to a sentence or two; don't rewrite the whole doc.

- [ ] **Step 2: Strengthen the README host warning** (the existing "bind beyond `127.0.0.1` only after auth lands" line). State explicitly that `serve` now refuses non-loopback hosts unless `--unsafe-no-auth` is passed, and that doing so exposes an unauthenticated upload/mutation surface — only acceptable on a trusted, isolated network. Cross-reference Task A's behavior so docs and code agree.

- [ ] **Step 3: Commit** (docs-only; lint/format still run for markdown-adjacent safety)
  ```bash
  uv run ruff check . >/dev/null 2>&1 || true
  git commit -am "docs: correct default-detector description; harden serve host warning"
  ```

---

## Final verification

After all in-scope tasks:

```bash
uv run pytest -m "not live and not slow" -q
uv run ruff check .
uv run ruff format --check .
uv run mypy consistency_checker
uv build
```

Then: open follow-up GitHub issues for each Deferred item above (legacy `/api/runs` removal with the `cc_process.html` SSE caveat; provider disclosure UI; corpus deletion API; data classification). Finish via superpowers:finishing-a-development-branch → PR → /code-review:code-review → CI/review → squash-merge → delete branch.
