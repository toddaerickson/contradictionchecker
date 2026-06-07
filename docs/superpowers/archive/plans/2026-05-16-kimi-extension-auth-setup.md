# Kimi Extension Auth Setup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure Kimi CLI extension authentication using a hybrid approach (shell env vars + project-local `.env` + optional workspace config).

**Architecture:** Three-tier auth chain: shell `$KIMI_API_KEY` (global, highest priority) → project `.env` (local override) → `.kimi/config.json` (workspace defaults). Secrets are gitignored at all levels to prevent accidental commits.

**Tech Stack:** Bash/Zsh shell, local `.env` file, JSON config file, Kimi CLI extension

---

## File Structure

| File | Status | Responsibility |
|------|--------|-----------------|
| `~/.bashrc` or `~/.zshrc` | Modify | Global shell environment; export `KIMI_API_KEY` |
| `.env` (project root) | Create | Project-local API key override (gitignored) |
| `.kimi/config.json` | Create | Workspace-specific defaults (gitignored) |
| `.gitignore` | Modify | Ensure `.env` and `.kimi/` are excluded |

---

## Task 1: Add KIMI_API_KEY to Shell Profile

**Files:**
- Modify: `~/.bashrc` or `~/.zshrc` (user's home directory)

- [ ] **Step 1: Determine which shell you use**

Run:
```bash
echo $SHELL
```

This returns either `/bin/bash` or `/bin/zsh` (or similar). Note the result.

- [ ] **Step 2: Open your shell profile in an editor**

If bash (`/bin/bash`):
```bash
nano ~/.bashrc
```

If zsh (`/bin/zsh`):
```bash
nano ~/.zshrc
```

- [ ] **Step 3: Add the KIMI_API_KEY export at the end of the file**

Scroll to the end and add:
```bash
export KIMI_API_KEY="sk-your-actual-kimi-api-key-here"
```

Replace `sk-your-actual-kimi-api-key-here` with your real Kimi API key (get it from Moonshot AI's dashboard if you don't have it).

- [ ] **Step 4: Save and exit the editor**

In `nano`: press `Ctrl+O`, then `Enter` to save, then `Ctrl+X` to exit.

- [ ] **Step 5: Reload the shell configuration**

Run:
```bash
source ~/.bashrc
```

Or if zsh:
```bash
source ~/.zshrc
```

- [ ] **Step 6: Verify the export was loaded**

Run:
```bash
echo $KIMI_API_KEY
```

Expected output: `sk-your-actual-kimi-api-key-here` (your actual key, not the placeholder).

If it's empty, the export didn't load. Check that you added the line correctly and re-run `source` from Step 5.

- [ ] **Step 7: Commit shell profile (optional, local only)**

Shell profiles are not in git, so no commit needed. Your shell now has access to `$KIMI_API_KEY` globally across all projects.

---

## Task 2: Create Project-Local .env File

**Files:**
- Create: `/home/terickson/contradictionchecker/.env`

- [ ] **Step 1: Navigate to project root**

```bash
cd /home/terickson/contradictionchecker
```

- [ ] **Step 2: Create the .env file**

```bash
cat > .env << 'EOF'
# Kimi CLI extension API key (local override; uses shell export if not set)
KIMI_API_KEY=sk-your-actual-kimi-api-key-here
EOF
```

Replace `sk-your-actual-kimi-api-key-here` with your actual key. (You can also leave it blank if you prefer to rely on the shell export from Task 1.)

- [ ] **Step 3: Verify the file was created**

```bash
cat .env
```

Expected output:
```
# Kimi CLI extension API key (local override; uses shell export if not set)
KIMI_API_KEY=sk-your-actual-kimi-api-key-here
```

- [ ] **Step 4: Do NOT commit this file yet**

The `.env` file contains secrets and must be gitignored. You'll verify this in Task 3.

---

## Task 3: Update .gitignore to Exclude Secrets

**Files:**
- Modify: `/home/terickson/contradictionchecker/.gitignore`

- [ ] **Step 1: Check if .env is already in .gitignore**

```bash
grep -E '\.env|\.kimi' .gitignore
```

If the output includes `.env` and `.kimi/`, skip to Step 4. If not, continue.

- [ ] **Step 2: Append .env and .kimi/ to .gitignore**

```bash
cat >> .gitignore << 'EOF'

# Kimi CLI extension secrets (local overrides)
.env
.env.local

# Kimi workspace config (gitignored)
.kimi/
EOF
```

- [ ] **Step 3: Verify the additions**

```bash
tail -5 .gitignore
```

Expected output (last few lines should include):
```
# Kimi CLI extension secrets (local overrides)
.env
.env.local

# Kimi workspace config (gitignored)
.kimi/
```

- [ ] **Step 4: Stage and commit the .gitignore update**

```bash
git add .gitignore
git commit -m "chore: add .env and .kimi/ to .gitignore"
```

Expected output: `1 file changed, 4 insertions(+)`

---

## Task 4: Create Workspace Config (.kimi/config.json)

**Files:**
- Create: `/home/terickson/contradictionchecker/.kimi/config.json`

- [ ] **Step 1: Create the .kimi directory**

```bash
mkdir -p .kimi
```

- [ ] **Step 2: Create the config file**

```bash
cat > .kimi/config.json << 'EOF'
{
  "model": "kimi",
  "max_tokens": 4096,
  "temperature": 0.7,
  "workspace_name": "contradictionchecker"
}
EOF
```

- [ ] **Step 3: Verify the file was created**

```bash
cat .kimi/config.json
```

Expected output:
```json
{
  "model": "kimi",
  "max_tokens": 4096,
  "temperature": 0.7,
  "workspace_name": "contradictionchecker"
}
```

- [ ] **Step 4: Verify .kimi/ is gitignored**

```bash
git status --short
```

You should NOT see `.kimi/config.json` in the output. If you do, something is wrong with the `.gitignore` update. Check Task 3 Step 2.

- [ ] **Step 5: Do NOT commit this file**

`.kimi/config.json` is gitignored and will not be committed. It's a local workspace default for this project only.

---

## Task 5: Validate Kimi CLI Extension Authentication

**Files:**
- Test: Verify `kimi` CLI commands work end-to-end

- [ ] **Step 1: Restart your terminal session (or reload shell)**

Close and reopen your terminal, or run:
```bash
exec bash  # or exec zsh if using zsh
```

This ensures all exports from Task 1 are in effect.

- [ ] **Step 2: Verify KIMI_API_KEY is available in the current shell**

```bash
echo $KIMI_API_KEY
```

Expected output: `sk-your-actual-kimi-api-key-here` (your actual key).

If empty, re-run `source ~/.bashrc` (or `~/.zshrc`) from Task 1 Step 5.

- [ ] **Step 3: Test Kimi CLI version**

```bash
kimi --version
```

Expected output: Kimi CLI version (e.g., `Kimi CLI v0.1.0` or similar). If this fails, the auth chain is not working—see Troubleshooting below.

- [ ] **Step 4: Test Kimi CLI status in this workspace**

```bash
cd /home/terickson/contradictionchecker
kimi status
```

Expected output: Should show workspace context (e.g., project name, file count, or similar status info).

- [ ] **Step 5: Test a simple Kimi CLI command**

```bash
kimi --help
```

Expected output: Help text with usage examples and available commands.

- [ ] **Step 6: Verify Kimi extension can read the workspace**

```bash
ls -la .kimi/
```

Expected output:
```
total 8
drwxr-xr-x  3 user user 4096 May 16 10:00 .
drwxr-xr-x 20 user user 4096 May 16 10:00 ..
-rw-r--r--  1 user user  100 May 16 10:00 config.json
```

This confirms `.kimi/config.json` exists and is readable.

---

## Task 6: Verify Auth Chain Priority (Optional Validation)

**Files:**
- Test: Confirm auth chain respects priority (env → .env → config)

- [ ] **Step 1: Temporarily override KIMI_API_KEY in the shell**

```bash
export KIMI_API_KEY="sk-test-override-key"
```

- [ ] **Step 2: Run Kimi CLI and verify it uses the override**

```bash
kimi --version
```

This should still work (the shell export takes priority over `.env`).

- [ ] **Step 3: Unset the shell override**

```bash
unset KIMI_API_KEY
```

- [ ] **Step 4: Verify Kimi falls back to .env**

```bash
kimi --version
```

This should still work because `.env` in the project provides the key.

- [ ] **Step 5: Restore the global export**

```bash
source ~/.bashrc  # or ~/.zshrc
```

Verify:
```bash
echo $KIMI_API_KEY
```

---

## Task 7: Final Commit and Documentation

**Files:**
- Modify: `.gitignore` (already committed in Task 3)
- Create: No new tracked files (`.env` and `.kimi/` are gitignored)

- [ ] **Step 1: Verify no untracked secrets are staged**

```bash
git status
```

Expected output: Should show `.gitignore` as committed (or no changes if already done). `.env` and `.kimi/` should NOT appear in the output.

- [ ] **Step 2: Final validation**

Run all validation steps from Task 5 one more time to confirm everything works:

```bash
echo $KIMI_API_KEY
kimi --version
cd /home/terickson/contradictionchecker && kimi status
```

All three commands should succeed.

- [ ] **Step 3: Document the setup (optional)**

If you want to onboard another developer later, share the spec file with them:

```bash
cat docs/superpowers/specs/2026-05-16-kimi-extension-auth-setup.md
```

This is already committed to git and will be in any clone.

- [ ] **Step 4: Success**

Kimi CLI extension is now authenticated and ready to use in this workspace. No further commits needed.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `kimi: command not found` | Kimi CLI is not installed or not in `$PATH`. Reinstall via Kimi's official docs. |
| `KIMI_API_KEY` env var is empty | Re-run `source ~/.bashrc` (or `~/.zshrc`) from Task 1 Step 5. Check that you added the export to the correct file. |
| `kimi --version` returns auth error | Verify `echo $KIMI_API_KEY` outputs a key. If empty, reload shell. If not empty, check that the key is valid (get a fresh one from Moonshot AI's dashboard). |
| `.env` is showing as unstaged changes in git | `.env` should be gitignored. Run `git rm --cached .env` to stop tracking it, then verify `git status` no longer shows it. |
| `.kimi/` files appear in `git status` | Verify `.kimi/` was added to `.gitignore` in Task 3. If still appearing, try `git rm -r --cached .kimi/` to stop tracking the directory. |

---

## Summary

After completing all tasks:
- ✅ Shell export set: `~/.bashrc` or `~/.zshrc` exports `KIMI_API_KEY`
- ✅ Project `.env` created: `/home/terickson/contradictionchecker/.env` (gitignored)
- ✅ Workspace config created: `/home/terickson/contradictionchecker/.kimi/config.json` (gitignored)
- ✅ `.gitignore` updated: Both `.env` and `.kimi/` are excluded from git
- ✅ Validation passed: `kimi --version`, `kimi status`, and `kimi --help` all succeed
- ✅ Auth chain working: Kimi CLI reads credentials in priority order (env → .env → config)

Kimi CLI extension is ready to use in this workspace.
