# Releasing to PyPI

How a solo maintainer ships `consistency-checker` to PyPI. Publishing is fully
automated by [`.github/workflows/release.yml`](../.github/workflows/release.yml)
and uses **OIDC Trusted Publishing** — there are **no API tokens** stored in the
repo, in GitHub secrets, or anywhere else. GitHub proves its identity to PyPI at
publish time, so nothing can leak.

The whole thing is three steps: one-time setup, an optional TestPyPI dry-run, and
the real release.

---

## 1. One-time: register the Trusted Publisher (do this once, on the PyPI website)

This must be done **before the first publish**. It tells PyPI to trust releases
coming from this specific GitHub repo + workflow.

1. Create a PyPI account at <https://pypi.org/account/register/> (and enable 2FA —
   PyPI requires it).
2. Go to the project's publishing settings:
   - If `consistency-checker` already exists on PyPI:
     **Your projects → consistency-checker → Manage → Publishing**.
   - If it does **not** exist yet (first ever release): go to
     <https://pypi.org/manage/account/publishing/> and use
     **"Add a pending publisher"**. PyPI will create the project automatically on
     the first successful publish.
3. Add a **GitHub** trusted publisher with exactly these values:

   | Field             | Value                     |
   |-------------------|---------------------------|
   | PyPI Project Name | `consistency-checker`     |
   | Owner             | `toddaerickson`           |
   | Repository name   | `contradictionchecker`    |
   | Workflow name     | `release.yml`             |
   | Environment       | *(leave blank)*           |

4. **(Optional, for dry-runs)** Repeat the same registration on
   <https://test.pypi.org/manage/account/publishing/> so the TestPyPI step below
   works. Same values.

You only ever do this once per index.

---

## 2. Optional: dry-run on TestPyPI

Use this to confirm the package builds and uploads cleanly without touching the
real index. It does **not** require a version bump (but TestPyPI also won't let
you re-upload the same version twice, so bump if you've already pushed that
version there).

1. GitHub → **Actions** tab → **Release** workflow → **Run workflow**
   (this is the `workflow_dispatch` trigger; leave target on `testpypi`).
2. The `build` job builds + smoke-installs the wheel, then `publish-testpypi`
   uploads to TestPyPI.
3. Verify the published package installs:

   ```sh
   pipx install \
     --index-url https://test.pypi.org/simple/ \
     --pip-args="--extra-index-url https://pypi.org/simple/" \
     consistency-checker
   ```

   (The `--extra-index-url` is needed because TestPyPI doesn't mirror this
   project's dependencies — they still come from real PyPI.)

---

## 3. Real release

1. Bump `version` in [`pyproject.toml`](../pyproject.toml) (e.g. `0.3.0` →
   `0.3.1`). Commit it and merge to `main`.
2. On GitHub: **Releases → Draft a new release**. Create a tag matching the
   version you bumped in step 1, prefixed with `v` (e.g. `v0.3.1`), target
   `main`, write notes, and click **Publish release**.
3. Publishing the release fires the `release` trigger: `release.yml` builds the
   sdist + wheel, smoke-installs the wheel, and `publish-pypi` uploads to PyPI.
4. Watch the run in the **Actions** tab. When it's green, the version is live at
   <https://pypi.org/project/consistency-checker/>.

> The git tag version and the `pyproject.toml` `version` should match. PyPI
> rejects re-uploading a version that already exists, so always bump first.

---

## Notes

- **No secrets to manage.** Auth is OIDC Trusted Publishing — GitHub mints a
  short-lived identity token at publish time. There are no PyPI API tokens to
  create, rotate, or store.
- If you rename the repo, change the owner, or rename `release.yml`, update the
  Trusted Publisher registration on PyPI to match, or publishing will fail.
