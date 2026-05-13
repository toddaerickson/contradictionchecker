"""Pipeline orchestration.

Glues the per-step modules into two top-level operations:

- :func:`ingest` — load corpus, chunk, extract atomic facts, embed.
- :func:`check` — gate candidate pairs, run NLI, run the LLM judge, log findings.

The CLI in :mod:`consistency_checker.cli.main` is a thin wrapper around these
functions. Tests can call the same functions directly with fake providers, which
is how the end-to-end smoke tests (Step 15) stay hermetic.
"""

from __future__ import annotations

from dataclasses import dataclass

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.check.gate import AnnGate, CandidateGate, CandidatePair
from consistency_checker.check.llm_judge import Judge, LLMJudge
from consistency_checker.check.nli_checker import NliChecker, score_bidirectional
from consistency_checker.check.providers.anthropic import AnthropicProvider
from consistency_checker.check.providers.openai import OpenAIProvider
from consistency_checker.config import Config
from consistency_checker.corpus.chunker import chunk_document
from consistency_checker.corpus.loader import load_corpus
from consistency_checker.extract.atomic_facts import (
    AnthropicExtractor,
    Extractor,
    FixtureExtractor,
)
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.index.embedder import (
    Embedder,
    SentenceTransformerEmbedder,
    embed_pending,
    rebuild_index,
)
from consistency_checker.index.faiss_store import FaissStore
from consistency_checker.logging_setup import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class IngestResult:
    n_documents: int
    n_chunks: int
    n_assertions: int
    n_embedded: int


@dataclass(frozen=True, slots=True)
class CheckResult:
    run_id: str
    n_assertions: int
    n_pairs_gated: int
    n_pairs_judged: int
    n_findings: int


# --- factories --------------------------------------------------------------


def make_extractor(config: Config) -> Extractor:
    """Build an extractor from config; ``fixture`` provider returns an empty fixture."""
    if config.judge_provider == "fixture":
        return FixtureExtractor({})
    return AnthropicExtractor(model=config.judge_model)


def make_embedder(config: Config) -> Embedder:
    return SentenceTransformerEmbedder(model_name=config.embedder_model)


def make_judge(config: Config) -> Judge:
    if config.judge_provider == "anthropic":
        return LLMJudge(AnthropicProvider(model=config.judge_model))
    if config.judge_provider == "openai":
        return LLMJudge(OpenAIProvider(model=config.judge_model))
    raise ValueError(
        f"make_judge(): provider {config.judge_provider!r} has no factory; "
        "construct a FixtureJudge directly in tests."
    )


# --- ingest -----------------------------------------------------------------


def ingest(
    config: Config,
    *,
    store: AssertionStore,
    faiss_store: FaissStore,
    extractor: Extractor,
    embedder: Embedder,
) -> IngestResult:
    """Walk corpus_dir → chunks → assertions → embeddings."""
    n_docs = n_chunks = n_assertions = 0
    for loaded in load_corpus(config.corpus_dir):
        store.add_document(loaded.document)
        n_docs += 1
        chunks = chunk_document(
            loaded,
            max_chars=config.chunk_max_chars,
            overlap_chars=config.chunk_overlap_chars,
        )
        n_chunks += len(chunks)
        for chunk in chunks:
            assertions = extractor.extract(chunk)
            if assertions:
                store.add_assertions(assertions)
                n_assertions += len(assertions)

    n_embedded = embed_pending(store, faiss_store, embedder)
    _log.info(
        "Ingested %d docs / %d chunks / %d assertions (%d newly embedded)",
        n_docs,
        n_chunks,
        n_assertions,
        n_embedded,
    )
    return IngestResult(
        n_documents=n_docs,
        n_chunks=n_chunks,
        n_assertions=n_assertions,
        n_embedded=n_embedded,
    )


# --- check ------------------------------------------------------------------


def _iter_candidates(
    config: Config, store: AssertionStore, faiss_store: FaissStore, gate: CandidateGate | None
) -> list[CandidatePair]:
    if gate is None:
        gate = AnnGate(
            faiss_store,
            top_k=config.gate_top_k,
            similarity_threshold=config.gate_similarity_threshold,
        )
    return list(gate.candidates(store))


def check(
    config: Config,
    *,
    store: AssertionStore,
    faiss_store: FaissStore,
    nli_checker: NliChecker,
    judge: Judge,
    audit_logger: AuditLogger,
    gate: CandidateGate | None = None,
) -> CheckResult:
    """Stage A → Stage B → audit. Returns run summary."""
    run_id = audit_logger.begin_run(
        config={
            "embedder_model": config.embedder_model,
            "nli_model": config.nli_model,
            "judge_provider": config.judge_provider,
            "judge_model": config.judge_model,
            "nli_contradiction_threshold": config.nli_contradiction_threshold,
            "gate_top_k": config.gate_top_k,
            "gate_similarity_threshold": config.gate_similarity_threshold,
        }
    )

    pairs = _iter_candidates(config, store, faiss_store, gate)
    n_pairs_gated = len(pairs)
    n_pairs_judged = 0
    n_findings = 0
    n_assertions = sum(1 for _ in store.iter_assertions())

    for pair in pairs:
        nli = score_bidirectional(nli_checker, pair.a.assertion_text, pair.b.assertion_text)
        if nli.p_contradiction < config.nli_contradiction_threshold:
            continue
        verdict = judge.judge(pair.a, pair.b)
        audit_logger.record_finding(run_id, candidate=pair, nli=nli, verdict=verdict)
        n_pairs_judged += 1
        if verdict.verdict == "contradiction":
            n_findings += 1

    audit_logger.end_run(
        run_id,
        n_assertions=n_assertions,
        n_pairs_gated=n_pairs_gated,
        n_pairs_judged=n_pairs_judged,
        n_findings=n_findings,
    )
    _log.info(
        "Run %s — %d gated / %d judged / %d findings",
        run_id,
        n_pairs_gated,
        n_pairs_judged,
        n_findings,
    )
    return CheckResult(
        run_id=run_id,
        n_assertions=n_assertions,
        n_pairs_gated=n_pairs_gated,
        n_pairs_judged=n_pairs_judged,
        n_findings=n_findings,
    )


# --- store maintenance ------------------------------------------------------


def rebuild_faiss(
    *,
    store: AssertionStore,
    faiss_store: FaissStore,
    embedder: Embedder,
) -> int:
    """Wipe the in-memory FAISS index by rebuilding from SQLite. Caller saves."""
    return rebuild_index(store, faiss_store, embedder)
