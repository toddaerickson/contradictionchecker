# Kimi Extension Auth Setup — Configuration Spec

**Date:** 2026-05-16  
**Goal:** Configure Kimi CLI extension with API key authentication (hybrid: env vars + local `.env` + optional workspace config)

## Problem

Kimi CLI extension is installed but not yet authenticated. Need to set up credentials in a way that:
- Works across all shells and projects (global)
- Supports project-specific overrides (local)
- Keeps secrets out of version control
- Minimal friction for future workspace onboarding

## Solution: Hybrid Auth Chain

Kimi extension will check credentials in priority order:

1. **Shell environment** — `$KIMI_API_KEY` (global, persists across shells)
2. **Project `.env`** — `KIMI_API_KEY=...` in project root (local override)
3. **Workspace config** — `.kimi/config.json` (optional, workspace defaults)

First match wins; subsequent sources are skipped.

## Setup Phases

### Phase 1: Global Shell Setup (One-Time)

Export the API key in your shell profile so it's available to all projects:

**For bash:**
```bash
# ~/.bashrc
export KIMI_API_KEY="your-actual-api-key-here"
```

**For zsh:**
```bash
# ~/.zshrc
export KIMI_API_KEY="your-actual-api-key-here"
```

After editing, reload the shell:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

Verify:
```bash
echo $KIMI_API_KEY
# Should output: your-actual-api-key-here
```

### Phase 2: Project-Local Setup (This Workspace)

#### Step 1: Create `.env` (optional but recommended)

In the project root (`/home/terickson/contradictionchecker/`), create `.env`:

```
# .env
KIMI_API_KEY=your-actual-api-key-here
```

If you want to use the global shell variable, you can leave this file empty or omit it—the extension will fall back to `$KIMI_API_KEY`.

#### Step 2: Update `.gitignore`

Ensure `.env` is gitignored:

```
# .gitignore
.env
.env.local
.kimi/
```

Check if these lines already exist; if not, add them.

#### Step 3 (Optional): Create Workspace Config

Create `.kimi/config.json` for workspace-specific Kimi settings:

```json
{
  "model": "kimi",
  "max_tokens": 4096,
  "temperature": 0.7,
  "workspace_name": "contradictionchecker"
}
```

This is optional. If you skip it, Kimi uses its defaults.

Ensure `.kimi/` is in `.gitignore` (from Step 2).

## Validation

After completing both phases, verify Kimi extension can authenticate:

1. Restart your terminal (or `source ~/.bashrc`/`~/.zshrc`)
2. Run:
   ```bash
   kimi --version
   ```
   Should succeed without auth errors.

3. Verify the extension sees this workspace:
   ```bash
   cd /home/terickson/contradictionchecker
   kimi status
   ```
   Should show workspace context.

4. (Optional) Test a simple Kimi CLI command to confirm the auth chain works:
   ```bash
   kimi --help
   ```

## File Inventory

| File | Action | Gitignored | Notes |
|------|--------|------------|-------|
| `~/.bashrc` or `~/.zshrc` | Add `export KIMI_API_KEY=...` | N/A (home dir) | Global; affects all shells |
| `.env` | Create | Yes | Project-local override; optional |
| `.kimi/config.json` | Create | Yes | Workspace defaults; optional |
| `.gitignore` | Update | — | Ensure `.env` and `.kimi/` are listed |

## Security Notes

- **Never commit `.env` or `.kimi/config.json` to git.** Both are gitignored to prevent accidental secret leaks.
- **Shell exports persist across sessions.** Once you add `export KIMI_API_KEY=...` to `~/.bashrc` or `~/.zshrc`, it's available globally. Keep your API key confidential.
- **Local `.env` overrides globals.** If you test with a different API key in `.env`, it takes precedence over the shell export. Good for testing; remember to revert before committing.

## Success Criteria

- [ ] `KIMI_API_KEY` exported in shell profile
- [ ] `.env` created in project root (or verified empty/omitted)
- [ ] `.gitignore` includes `.env` and `.kimi/`
- [ ] `kimi --version` succeeds without auth errors
- [ ] Kimi extension auto-discovers this workspace

## Rollback

If something goes wrong:

1. **Remove shell export:** Edit `~/.bashrc` or `~/.zshrc`, delete the `export KIMI_API_KEY=...` line, then `source` the file.
2. **Delete project files:** `rm .env .kimi/config.json` (they're not in git anyway).
3. **Restart terminal** and verify `echo $KIMI_API_KEY` returns empty.

---

## Implementation Order

This spec is consumed by the implementation plan (writing-plans skill output). The plan will detail:

1. Step-by-step terminal commands for Phase 1 (shell export)
2. File creation commands for Phase 2 (`.env`, `.kimi/config.json`)
3. Validation test commands
4. Commit strategy (if any config files go into the repo, though they shouldn't)
