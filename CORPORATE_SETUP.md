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
| `huggingface.co` and `cdn-lfs.huggingface.co` | DeBERTa NLI model download (~440 MB default, ~1.5 GB if you opt up to large) | Only when the pairwise detector is enabled (`--pairwise` or `pairwise_enabled: true` — off by default per ADR-0015). First `check --pairwise` run only; cached afterwards in `~/.cache/huggingface/` |
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
corp DLP rules target. Note: with pairwise off by default (ADR-0015),
this egress is only required if you plan to run `check --pairwise`;
default `check` runs do not touch Hugging Face.

**Workaround for blocked Hugging Face egress:** download the NLI model
once from a permitted machine (run `check --pairwise` there once),
then copy the contents of `~/.cache/huggingface/hub/models--<model-name>`
to the corp machine.

---

## 1.5 System dependencies

The OCR fallback path uses `unstructured`'s `strategy="hi_res"`, which calls system Tesseract under the hood. First run downloads ~500 MB of layout + OCR model weights into the HuggingFace cache.

Data-classification implication: OCR runs entirely locally. Document images do not leave the machine for OCR — the LLM judge is still the only path that sends content to a third-party API.

Install Tesseract: `apt install tesseract-ocr` (Debian/Ubuntu), `brew install tesseract` (macOS), or the equivalent for your distro.

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

## 4.5 Memory budget

`check --pairwise` loads two large models into the same process; default
`check` (pairwise off per ADR-0015) skips the NLI model entirely:

| Component | Resident RSS |
|-----------|--------------|
| `DeBERTa-v3-base-mnli-fever-anli` (NLI, only when `--pairwise` is enabled) | ~0.6 GB |
| `all-mpnet-base-v2` (embedder, ingest stage) | ~0.5 GB |
| Python / sqlite / faiss baseline | ~0.3 GB |
| **Estimated peak `check --pairwise` RSS (default NLI model)** | **~1.4 GB** |
| **Estimated peak default `check` RSS (definitions only)** | **~0.8 GB** |

If you opt up to `DeBERTa-v3-large-mnli-fever-anli-ling-wanli` for max recall,
add ~1.5 GB to peak (≈ 2.5–2.8 GB total) when running with `--pairwise`.

Recommended floors:

- **Bare metal / VM:** at least **4 GB** of free RAM at the moment you start `check`.
- **VS Code devcontainer / Docker Desktop:** the default container memory cap on
  Mac/Windows is often 2 GB, which will OOM. Raise it to **6 GB**:
  ```jsonc
  // .devcontainer/devcontainer.json
  "runArgs": ["--memory=6g", "--memory-swap=6g"]
  ```
  Or in Docker Desktop → Settings → Resources → Memory.
- **WSL2:** the global cap lives in `~/.wslconfig`:
  ```ini
  [wsl2]
  memory=6GB
  ```

Pre-flight guardrail:

```yaml
# config.yml
max_memory_mb: 3000
```

When set, `check --pairwise` aborts before the NLI model loads if `MemAvailable`
is below the threshold, with a clear message instead of an OOM crash. The
pre-flight is **skipped** entirely when pairwise is off (the dominant memory
consumer doesn't load). Leave unset to skip the check unconditionally (you'll
get a soft warning if available memory looks low).

If you need maximum gate recall and have headroom, swap up to the large model:

```yaml
# config.yml
nli_model: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
```

(~1.4 GB on disk; ~3× higher NLI RSS for a few percentage points of recall.
Stage B's LLM judge catches most of what the base gate would otherwise miss.)

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

- **~440 MB NLI model download** on the first `check --pairwise` — visible
  as a long pause. The CLI now prints a one-line warning before the
  download so you don't suspect a hang. (Larger if you opted up to
  `DeBERTa-v3-large`: ~1.5 GB.) Default `check` runs (pairwise off per
  ADR-0015) skip this download entirely.
- **Web UI binds to `127.0.0.1:8000` by default.** Corp browsers
  usually allow loopback; if not, switch ports or open the markdown
  report instead.
- **First reports land under `<data_dir>/reports/`** with timestamped
  filenames. The `data_dir` is set in `config.yml` — pick a location
  the corp DLP scanner won't quarantine (often inside your home dir,
  not in a synced folder).

---

## 7. Operational hygiene

- **`uv run consistency-check check --pairwise`** enables the pairwise
  contradiction detector (NLI gate → LLM judge). Off by default per
  ADR-0015 — own-corpus eval on legal prose showed near-zero useful yield
  at high compute cost. Re-enable on numeric- or spec-heavy corpora where
  outright sign flips and quantity disagreements live.
- **`uv run consistency-check check --max-cost <USD>`** is a budget
  guardrail (ADR-0016). Set a hard ceiling (e.g. `--max-cost 5.00`) and
  the pipeline runs `estimate-cost` as a pre-flight; if the conservative
  projection exceeds the ceiling, `check` aborts before any NLI or judge
  bootstrap (no spend, no model download). The same value can live in
  `config.yml` as `max_cost_usd`. `estimate-cost` itself now defaults
  per-call costs from your configured `judge_provider`, so Moonshot/Kimi
  users see realistic sub-cent projections (~$0.0001–$0.001/call) instead
  of the Anthropic/OpenAI tier (~$0.003–$0.010/call).
- **`uv run consistency-check check --no-definitions`** disables the
  definition-inconsistency detector if you only want contradictions
  (and only makes sense paired with `--pairwise`).
- **`uv run consistency-check check --pairwise --deep`** enables the
  three-document conditional contradiction pass. Roughly 1.5× to 2× the
  pairwise API spend; `--deep` shares the pairwise NLI gate, so
  `--deep` without `--pairwise` is rejected.
- **The audit SQLite DB** under `data_dir/store/` retains every judge
  verdict (including `uncertain` rows hidden from reports). Back it
  up before re-running on the same corpus if you care about
  reproducibility.
- **API spend** scales roughly linearly with the number of candidate
  pairs (~`n_assertions²` after gating). The combined extractor adds
  ~10% overhead from the definition output tokens; the definition
  judge is per-pair-per-term-group, usually a small fraction of total
  spend.
