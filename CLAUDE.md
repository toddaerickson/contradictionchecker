# CLAUDE.md

Guidance for Claude Code sessions working on this repository.

## Project shape

`consistency-checker` is a Python 3.11+ tool that detects internal contradictions across a corpus of documents. The pipeline is **two-stage**: a DeBERTa NLI gate (cheap, high-recall) feeds an LLM judge (structured JSON output, high-precision).

Architecture rules to preserve:

- **SQLite is the canonical store.** The FAISS index is a derived view, rebuildable from SQLite via `consistency-check store rebuild-index`. Never let FAISS become the source of truth.
- **Assertion ids are content hashes** (`sha256(doc_id || assertion_text)[:16]`). Re-running extraction on the same content is idempotent by design.
- **Char spans round-trip exactly.** The chunker invariant is `text[chunk.char_start:chunk.char_end] == chunk.text`. Don't break this; tests will catch it.
- **Judge output is schema-validated, never string-parsed.** This is the failure mode of the kb_guardian prior art that the architecture explicitly avoids. Use the provider-native structured-output paths (tool-use for Anthropic, `beta.chat.completions.parse` for OpenAI) plus a Pydantic guard.
- **`uncertain` is a first-class verdict.** It stays in the audit DB for replay and threshold tuning but is excluded from reports. Treat `uncertain` as "don't bother the user" — not as a bug.

## Module map (`consistency_checker/`)

```
corpus/      loader.py, chunker.py                       — ingest .txt/.md, sentence-window chunks
extract/     schema.py, atomic_facts.py, prompts/        — Document/Assertion dataclasses; FActScore-style extraction
index/       assertion_store.py, embedder.py,            — SQLite + FAISS; migrations live here too
             faiss_store.py, migrations/
check/       gate.py, nli_checker.py, llm_judge.py,      — Stage A NLI + Stage B judge with provider abstraction
             providers/{base,anthropic,openai}.py,
             prompts/{judge_system,judge_user}.txt
audit/       logger.py, report.py                        — SQLite-backed findings; deterministic markdown report
cli/         main.py                                     — typer CLI; thin wrapper around pipeline.py
pipeline.py  ingest() and check() orchestration         — what the CLI calls
config.py    Pydantic v2, loaded from YAML              — env overrides via CC_<FIELD>
```

## Commands you'll run constantly

```sh
uv sync                                             # install/update deps
uv run pytest -m "not slow and not live"            # the CI gate
uv run pytest -m slow                               # HF model downloads (~800 MB - 1.5 GB)
uv run pytest -m live                               # real Anthropic / OpenAI calls
uv run ruff check .                                 # lint
uv run ruff format --check .                        # format check (use `format` without --check to fix)
uv run mypy consistency_checker                     # strict types
uv build                                            # wheel + sdist into dist/
```

CI runs `ruff check`, `ruff format --check`, `mypy`, `pytest -m "not live and not slow"`, then `uv build` + a wheel smoke-install. All four must stay green.

## Pytest marks

- (default, no mark): hermetic. Used by CI. Must never download models or hit external APIs.
- `slow`: downloads a HF model. Skip in CI.
- `live`: hits Anthropic or OpenAI. Requires `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`. Skip in CI; gated by env var checks.
- `e2e_fixture`: end-to-end test using `Fixture*` checkers. Runs in CI.

When adding a test that downloads models or hits external APIs, mark it appropriately. Both marks combine — a slow live test gets both.

## Hermetic test scaffolding

`tests/conftest.py::HashEmbedder` produces deterministic float32 vectors derived from sha256 (via uint32 → float64 → normalised). It's the standard test embedder. Don't make tests depend on real sentence-transformers unless they're marked `slow`.

The check stage has fixture-mode for every component:
- `extract.atomic_facts.FixtureExtractor`
- `check.nli_checker.FixtureNliChecker`
- `check.llm_judge.FixtureJudge`

When testing pipeline behaviour, wire these in via the dependency-injection seams in `pipeline.ingest()` / `pipeline.check()`, or via monkey-patching the factories in `consistency_checker.cli.main`.

## Common gotchas

- **Cosine similarity range is `[-1, 1]`.** `AnnGate`'s threshold validator was widened to allow that range; the config default is `0.7` for real embeddings but tests using `HashEmbedder` need `-1.0` or close to it.
- **Pydantic config is frozen.** Use `cfg.model_copy(update={...})` to derive a tweaked config; don't try to mutate.
- **PR rebase before opening.** The development branch (`claude/design-consistency-checker-IwJv4` historically) collects squash-merged trees that differ in lineage from `main`. Rebase the working branch onto fresh `origin/main` before pushing a new PR, otherwise GitHub's squash-merge will report conflicts.
- **typer defaults trigger ruff's `B008`.** That rule is ignored only for `consistency_checker/cli/*.py` via per-file-ignores. Don't blanket-disable it elsewhere.
- **Anthropic and Hugging Face APIs are typed loosely.** Where mypy can't reason about a SDK return (e.g. transformers' `pipeline(...)` returning a union), prefer `Any`-typing the attribute (`self._pipe: Any = ...`) over scattered `# type: ignore` comments. There's one such case in `nli_checker.py` and one in `atomic_facts.py`.
- **First `check` run downloads ~800 MB.** That's the DeBERTa NLI model. Users who don't expect this will think the CLI is hung.

## Where to put new code

- New corpus loader (PDF, DOCX) → `consistency_checker/corpus/loader.py`. The stub raises `NotImplementedError`; lift the file extension out of `STUB_EXTENSIONS` and route to a real implementation.
- New embedder → implement the `Embedder` Protocol in `consistency_checker/index/embedder.py`. The Protocol only requires `dim` and `embed_texts(texts) -> NDArray[np.float32]`.
- New judge provider → implement `JudgeProvider` in `consistency_checker/check/providers/<name>.py`. Add it to the `make_judge()` factory in `pipeline.py` and a fresh ADR in `docs/decisions/`.
- New schema fields → write a new migration `consistency_checker/index/migrations/NNNN_<name>.sql`. The migration loader picks it up by filename. Never edit existing migration files.
- New CLI command → `consistency_checker/cli/main.py`. Keep it a thin wrapper; orchestration belongs in `pipeline.py` so tests can call it without the CLI.

## Conventions

- **No comments unless they explain non-obvious _why_.** Identifiers should carry the _what_.
- **Default to dataclasses for cross-module entities; Pydantic for parsing external input.** `JudgePayload` is Pydantic because it validates LLM output; `JudgeVerdict` is a dataclass because it just moves data around.
- **One PR per build step.** Reference the step number in the PR title (`Step N: ...`). Squash-merge, then rebase the working branch on the new `main` before the next push.
- **Commit messages explain _why_, not _what_.** The diff covers the what.

## Plans

The 17-step build plan that produced v0.1.0 lives at `/root/.claude/plans/propose-build-steps-start-lovely-eagle.md` (session-local). Forward-looking work is captured in `futureplans.md`.

## Roadmap

`futureplans.md` is the live roadmap. If a session ends up doing work that's listed there, move that item to a "Completed" section in the same file rather than deleting it — provenance matters for future planning.
