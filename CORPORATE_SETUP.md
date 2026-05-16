# Corporate-network setup

Moving this tool into a corporate environment exposes failure modes that
don't show up on a personal machine. Work through this checklist **before**
the first real run — most of the items are one-time gates that block the
whole pipeline if they fail. Walk top to bottom; later items assume earlier
items pass.

---

## 0. Data-classification policy (the blocker)

The pipeline sends every document chunk to Anthropic (or OpenAI) for atomic-
fact extraction and the LLM judge. **Confirm before any move:**

- Is sending the source documents to a third-party API allowed under your
  firm's data-classification policy?
- Are the documents marked confidential, privileged, or subject to a
  data-residency requirement (GDPR, HIPAA, attorney-client, MNPI)?
- Does your firm have an enterprise Anthropic / OpenAI account that
  covers this use case, or are you using a personal key?

If sending data to a public API is not allowed, **stop** — the
local-LLM judge provider ([futureplans.md item #12](futureplans.md))
must ship first. There is no workaround inside the current architecture.

Document the answer in your team wiki or a `data-policy.md` adjacent to
your config — future-you needs to see what was approved.

---

## 1. Network egress

The tool needs outbound HTTPS to:

| Host | Why | When |
|------|-----|------|
| `api.anthropic.com` | LLM judge + extraction calls | Every `check` / `ingest` run |
| `huggingface.co` and `cdn-lfs.huggingface.co` | DeBERTa NLI model download (~800 MB) | First `check` run only; cached afterwards in `~/.cache/huggingface/` |
| `api.openai.com` (optional) | LLM judge if `judge_provider: openai` | If using OpenAI instead of Anthropic |

Validate from the corp machine before installing:

```sh
curl -sI https://api.anthropic.com/v1/messages | head -1
curl -sI https://huggingface.co | head -1
```

If either fails, you need a corporate proxy (`HTTPS_PROXY`,
`HTTP_PROXY`, `NO_PROXY`) configured, or an exception ticket with
network/security. The Hugging Face download is the most likely to be
blocked — large binary egress to a cloud CDN is exactly the pattern
corp DLP rules target.

**Workaround for blocked Hugging Face egress:** download the NLI model
once from a permitted machine, then copy the contents of
`~/.cache/huggingface/hub/models--<model-name>` to the corp machine.

---

## 2. Python toolchain

Required: **Python 3.11 or newer**. Corporate Linux images often ship 3.9
or 3.10. Check first:

```sh
python3 --version
```

If you don't have 3.11+:

- **Preferred:** install via [`uv`](https://docs.astral.sh/uv/) which
  manages its own Python interpreters and is a single static binary —
  often easier to get through corp software review than a system Python
  upgrade.
- **Fallback:** `pyenv`, conda/mamba, or a corp-IT-managed Python build.

Install `uv`:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: pipx install uv
```

---

## 3. Secrets management

The `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) must be available as an
environment variable at run time. Do not commit it.

**Three places it can live, ranked by safety on a corp machine:**

1. **Corp secret manager** (1Password CLI, HashiCorp Vault, AWS Secrets
   Manager, etc.) — pulled into the shell session at use:
   ```sh
   export ANTHROPIC_API_KEY="$(op read 'op://Private/Anthropic/api_key')"
   ```
2. **A `.env` file in the repo root** that is `.gitignore`'d. Source it
   per-shell:
   ```sh
   echo 'export ANTHROPIC_API_KEY=sk-ant-...' > .env
   source .env
   ```
   This project's `.gitignore` already excludes `.env`. Verify with
   `git check-ignore .env`.
3. **Shell `rc` file** (`~/.bashrc`, `~/.zshrc`) — convenient, but the
   key persists across every shell session on the host. Avoid on shared
   corp machines.

**Never:** put the key into `config.yml` — that file is meant to be
committed.

After setting, sanity-check:

```sh
env | grep ANTHROPIC_API_KEY | wc -l   # should print 1
```

---

## 4. Validate the install with the hermetic smoke test

Before spending a single API dollar, prove the install works end-to-end
using the fixture pipeline (no network, no API calls, no model download):

```sh
uv sync
uv run pytest -m e2e_fixture
```

Expected: tests pass within ~30 seconds. This exercises ingest → check →
report using `FixtureExtractor`, `FixtureNliChecker`, `FixtureJudge`,
and `FixtureDefinitionJudge`. If it passes, the install is correct and
the only remaining unknowns are network egress + API credentials.

---

## 5. Estimate cost before the first real run

Once the smoke test passes:

```sh
# 1. Ingest a small corpus (3-10 docs).
uv run consistency-check ingest path/to/corpus/

# 2. Estimate cost BEFORE running check.
uv run consistency-check estimate-cost
```

`estimate-cost` counts the candidate pairs that would survive the gate
and the definition pairs that would be judged, and prints a rough API-
spend ceiling. Use it to right-size the corpus or back out before
committing.

---

## 6. First real run

```sh
uv run consistency-check check
uv run consistency-check report
```

Or, for the web UI:

```sh
uv run consistency-check serve --open
```

First-run gotchas to expect:

- **~800 MB NLI model download** on the first `check` — visible as a
  long pause. The CLI now prints a one-line warning before the
  download so you don't suspect a hang.
- **Web UI binds to `127.0.0.1:8000` by default.** Corp browsers
  usually allow loopback; if not, switch ports or open the markdown
  report instead.
- **First reports land under `<data_dir>/reports/`** with timestamped
  filenames. The `data_dir` is set in `config.yml` — pick a location
  the corp DLP scanner won't quarantine (often inside your home dir,
  not in a synced folder).

---

## 7. Operational hygiene

- **`uv run consistency-check check --no-definitions`** disables the
  definition-inconsistency detector if you only want contradictions.
- **`uv run consistency-check check --deep`** enables the three-document
  conditional contradiction pass. Roughly 1.5× to 2× the API spend.
- **The audit SQLite DB** under `data_dir/store/` retains every judge
  verdict (including `uncertain` rows hidden from reports). Back it
  up before re-running on the same corpus if you care about
  reproducibility.
- **API spend** scales roughly linearly with the number of candidate
  pairs (~`n_assertions²` after gating). The combined extractor adds
  ~10% overhead from the definition output tokens; the definition
  judge is per-pair-per-term-group, usually a small fraction of total
  spend.
