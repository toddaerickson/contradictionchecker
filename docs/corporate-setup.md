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
local-LLM judge provider must ship first. There is no workaround inside the
current architecture (see `futureplans.md` item #12).

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

## 4. Memory budget

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