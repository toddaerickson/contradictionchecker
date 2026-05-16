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
from consistency_checker.check.definition_checker import (
    DefinitionChecker,
    _group_by_canonical_term,
)
from consistency_checker.check.definition_judge import (
    DefinitionJudge,
    LLMDefinitionJudge,
)
from consistency_checker.check.gate import AnnGate, CandidateGate, CandidatePair
from consistency_checker.check.llm_judge import Judge, JudgeVerdict, LLMJudge
from consistency_checker.check.multi_party_judge import (
    LLMMultiPartyJudge,
    MultiPartyJudge,
)
from consistency_checker.check.nli_checker import NliChecker, score_bidirectional
from consistency_checker.check.providers.anthropic import (
    AnthropicDefinitionProvider,
    AnthropicMultiPartyProvider,
    AnthropicProvider,
)
from consistency_checker.check.providers.base import CONTRADICTION_VERDICTS
from consistency_checker.check.providers.definition_base import (
    DEFINITION_INCONSISTENCY_VERDICTS,
)
from consistency_checker.check.providers.openai import (
    OpenAIDefinitionProvider,
    OpenAIMultiPartyProvider,
    OpenAIProvider,
)
from consistency_checker.check.triangle import Triangle, find_triangles
from consistency_checker.config import Config
from consistency_checker.corpus.chunker import chunk_document
from consistency_checker.corpus.loader import load_corpus
from consistency_checker.extract.atomic_facts import (
    AnthropicExtractor,
    Extractor,
    FixtureExtractor,
)
from consistency_checker.extract.quantitative import (
    extract_quantities,
    find_value_disagreements,
    is_sign_flip,
)
from consistency_checker.extract.schema import Assertion
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
class CostEstimate:
    """Upper-bound estimate of API spend for a check run — the NLI gate
    typically filters 30-70% of gate-pass pairs before they reach the judge,
    so the dollar bounds here are a CEILING, not an actual-spend prediction.
    """

    n_assertions: int
    n_candidate_pairs: int
    n_definition_pairs: int
    judge_calls_ceiling: int
    est_cost_low: float
    est_cost_high: float
    per_call_low: float
    per_call_high: float


@dataclass(frozen=True, slots=True)
class CheckResult:
    run_id: str
    n_assertions: int
    n_pairs_gated: int
    n_pairs_judged: int
    n_findings: int
    n_triangles_judged: int = 0
    n_multi_party_findings: int = 0
    n_definition_pairs_judged: int = 0
    n_definition_findings: int = 0


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


def make_multi_party_judge(config: Config) -> MultiPartyJudge:
    """Build the triangle judge from config; ``fixture`` provider has no factory."""
    if config.judge_provider == "anthropic":
        return LLMMultiPartyJudge(AnthropicMultiPartyProvider(model=config.judge_model))
    if config.judge_provider == "openai":
        return LLMMultiPartyJudge(OpenAIMultiPartyProvider(model=config.judge_model))
    raise ValueError(
        f"make_multi_party_judge(): provider {config.judge_provider!r} has no factory; "
        "construct a FixtureMultiPartyJudge directly in tests."
    )


def make_definition_judge(config: Config) -> DefinitionJudge:
    """Build the definition judge from config.

    The ``fixture`` provider returns an empty :class:`FixtureDefinitionJudge`,
    mirroring how ``make_extractor`` handles fixture-mode for the web app's
    hermetic tests.
    """
    if config.judge_provider == "fixture":
        from consistency_checker.check.definition_judge import FixtureDefinitionJudge

        return FixtureDefinitionJudge({})
    if config.judge_provider == "anthropic":
        return LLMDefinitionJudge(AnthropicDefinitionProvider(model=config.judge_model))
    if config.judge_provider == "openai":
        return LLMDefinitionJudge(OpenAIDefinitionProvider(model=config.judge_model))
    raise ValueError(f"make_definition_judge(): provider {config.judge_provider!r} has no factory.")


def make_definition_checker(config: Config) -> DefinitionChecker:
    return DefinitionChecker(judge=make_definition_judge(config))


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


def _build_numeric_context(a: Assertion, b: Assertion, *, threshold: float) -> str | None:
    """Build a structured numeric-disagreement hint for the judge prompt (E3).

    Returns ``None`` when no value disagreement clears ``threshold`` — the judge
    sees an unchanged prompt in that case so golden tests stay stable.
    """
    disagreements = find_value_disagreements(
        a.assertion_text, b.assertion_text, threshold=threshold
    )
    if not disagreements:
        return None
    lines: list[str] = []
    for ta, tb in disagreements:
        scope_str = f" ({ta.scope})" if ta.scope else ""
        unit_str = ta.unit or ""
        lines.append(
            f"- {ta.metric}{scope_str}: A says {ta.value}{unit_str}, B says {tb.value}{unit_str}"
        )
    return "\n".join(lines)


def _try_numeric_short_circuit(a: Assertion, b: Assertion) -> JudgeVerdict | None:
    """Return a deterministic contradiction verdict iff ``a`` and ``b`` sign-flip on
    a shared (metric, scope, unit). Returns ``None`` otherwise — caller falls through
    to the LLM judge. See ADR-0005.
    """
    a_tuples = extract_quantities(a.assertion_text)
    if not a_tuples:
        return None
    b_tuples = extract_quantities(b.assertion_text)
    if not b_tuples:
        return None
    for ta in a_tuples:
        for tb in b_tuples:
            if not is_sign_flip(ta, tb):
                continue
            unit_suffix = ta.unit or ""
            rationale = (
                f"Numeric short-circuit: metric={ta.metric}, "
                f"A={ta.value}{unit_suffix}, B={tb.value}{unit_suffix}, "
                "polarity mismatch."
            )
            evidence = [
                f"{ta.value}{unit_suffix}",
                f"{tb.value}{unit_suffix}",
            ]
            return JudgeVerdict(
                assertion_a_id=min(a.assertion_id, b.assertion_id),
                assertion_b_id=max(a.assertion_id, b.assertion_id),
                verdict="numeric_short_circuit",
                confidence=1.0,
                rationale=rationale,
                evidence_spans=evidence,
            )
    return None


def _run_multi_party_pass(
    pairs: list[CandidatePair],
    *,
    multi_party_judge: MultiPartyJudge,
    audit_logger: AuditLogger,
    run_id: str,
    max_per_run: int,
) -> tuple[int, int]:
    """Enumerate triangles in the gate graph, judge each, log findings.

    Returns ``(n_triangles_judged, n_multi_party_findings)``. See ADR-0006.
    """
    n_triangles_judged = 0
    n_multi_party_findings = 0
    triangles: list[Triangle] = list(find_triangles(pairs, max_per_run=max_per_run))
    for triangle in triangles:
        verdict = multi_party_judge.judge(triangle)
        audit_logger.record_multi_party_finding(
            run_id,
            assertion_ids=list(triangle.assertion_ids),
            doc_ids=list(triangle.doc_ids),
            triangle_edge_scores=list(triangle.edge_scores),
            judge_verdict=verdict.verdict,
            judge_confidence=verdict.confidence,
            judge_rationale=verdict.rationale,
            evidence_spans=list(verdict.evidence_spans),
        )
        n_triangles_judged += 1
        if verdict.verdict == "multi_party_contradiction":
            n_multi_party_findings += 1
    return n_triangles_judged, n_multi_party_findings


def _run_definition_pass(
    *,
    store: AssertionStore,
    checker: DefinitionChecker,
    audit_logger: AuditLogger,
    run_id: str,
) -> tuple[int, int]:
    """Run the definition checker over all stored definitions and log findings.

    Returns ``(n_judged, n_findings)``. The NLI gate is bypassed for this stage
    by design — see DefinitionChecker for the rationale.
    """
    definitions = list(store.iter_definitions())
    n_judged = 0
    n_findings = 0
    for finding in checker.find_inconsistencies(definitions):
        audit_logger.record_definition_finding(run_id, finding=finding)
        n_judged += 1
        if finding.verdict.verdict in DEFINITION_INCONSISTENCY_VERDICTS:
            n_findings += 1
    return n_judged, n_findings


def check(
    config: Config,
    *,
    store: AssertionStore,
    faiss_store: FaissStore,
    nli_checker: NliChecker,
    judge: Judge,
    audit_logger: AuditLogger,
    gate: CandidateGate | None = None,
    multi_party_judge: MultiPartyJudge | None = None,
    definition_checker: DefinitionChecker | None = None,
    run_id: str,
) -> CheckResult:
    """Stage A → Stage B → optional multi-party + definition passes → audit."""
    audit_logger.update_run_status(run_id, "running")

    pairs = _iter_candidates(config, store, faiss_store, gate)
    n_pairs_gated = len(pairs)
    n_pairs_judged = 0
    n_findings = 0
    n_assertions = store.stats()["assertions"]

    for pair in pairs:
        nli = score_bidirectional(nli_checker, pair.a.assertion_text, pair.b.assertion_text)
        if nli.p_contradiction < config.nli_contradiction_threshold:
            continue
        # ADR-0005 step E2: deterministic sign-flip cases skip the LLM judge entirely.
        verdict = _try_numeric_short_circuit(pair.a, pair.b)
        if verdict is None:
            # Step E3: hand the LLM judge a structured hint when same-metric,
            # same-scope values disagree above the configured threshold without
            # flipping sign. Empty / None hint leaves the prompt unchanged.
            numeric_context = _build_numeric_context(
                pair.a, pair.b, threshold=config.numeric_disagreement_threshold
            )
            verdict = judge.judge(pair.a, pair.b, numeric_context=numeric_context)
        audit_logger.record_finding(run_id, candidate=pair, nli=nli, verdict=verdict)
        n_pairs_judged += 1
        if verdict.verdict in CONTRADICTION_VERDICTS:
            n_findings += 1

    # ADR-0006 F4: optional triangle pass over the same gate output.
    n_triangles_judged = 0
    n_multi_party_findings = 0
    if multi_party_judge is not None:
        n_triangles_judged, n_multi_party_findings = _run_multi_party_pass(
            pairs,
            multi_party_judge=multi_party_judge,
            audit_logger=audit_logger,
            run_id=run_id,
            max_per_run=config.max_triangles_per_run,
        )

    # ADR-0009: definition-inconsistency detector. Term-grouping replaces the
    # NLI gate; bypasses the contradiction pipeline entirely.
    n_definition_pairs_judged = 0
    n_definition_findings = 0
    if definition_checker is not None:
        n_definition_pairs_judged, n_definition_findings = _run_definition_pass(
            store=store,
            checker=definition_checker,
            audit_logger=audit_logger,
            run_id=run_id,
        )

    audit_logger.end_run(
        run_id,
        n_assertions=n_assertions,
        n_pairs_gated=n_pairs_gated,
        n_pairs_judged=n_pairs_judged,
        n_findings=n_findings + n_definition_findings,
    )
    _log.info(
        "Run %s — %d gated / %d judged / %d findings / %d triangles / "
        "%d multi-party / %d definition pairs / %d definition findings",
        run_id,
        n_pairs_gated,
        n_pairs_judged,
        n_findings,
        n_triangles_judged,
        n_multi_party_findings,
        n_definition_pairs_judged,
        n_definition_findings,
    )
    return CheckResult(
        run_id=run_id,
        n_assertions=n_assertions,
        n_pairs_gated=n_pairs_gated,
        n_pairs_judged=n_pairs_judged,
        n_findings=n_findings,
        n_triangles_judged=n_triangles_judged,
        n_multi_party_findings=n_multi_party_findings,
        n_definition_pairs_judged=n_definition_pairs_judged,
        n_definition_findings=n_definition_findings,
    )


# --- cost estimate ----------------------------------------------------------


def estimate_cost(
    config: Config,
    *,
    store: AssertionStore,
    faiss_store: FaissStore,
    per_call_low: float = 0.003,
    per_call_high: float = 0.010,
) -> CostEstimate:
    """Count judge calls a check run would make, without making any LLM calls.

    NLI is skipped because (a) it requires the ~800 MB DeBERTa model
    download and (b) the user wants a fast pre-flight estimate, not a
    real-cost figure. Real spend is usually 30-70% lower than this
    ceiling because NLI filters most gate-pass pairs.
    """
    candidates = _iter_candidates(config, store, faiss_store, gate=None)
    n_candidate_pairs = len(candidates)

    groups = _group_by_canonical_term(store.iter_definitions())
    n_definition_pairs = sum(len(v) * (len(v) - 1) // 2 for v in groups.values())

    judge_calls_ceiling = n_candidate_pairs + n_definition_pairs
    return CostEstimate(
        n_assertions=store.stats()["assertions"],
        n_candidate_pairs=n_candidate_pairs,
        n_definition_pairs=n_definition_pairs,
        judge_calls_ceiling=judge_calls_ceiling,
        est_cost_low=judge_calls_ceiling * per_call_low,
        est_cost_high=judge_calls_ceiling * per_call_high,
        per_call_low=per_call_low,
        per_call_high=per_call_high,
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
