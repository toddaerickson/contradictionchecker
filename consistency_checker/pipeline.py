"""Pipeline orchestration.

Glues the per-step modules into two top-level operations:

- :func:`ingest` — load corpus, chunk, extract atomic facts, embed.
- :func:`check` — gate candidate pairs, run NLI, run the LLM judge, log findings.

The CLI in :mod:`consistency_checker.cli.main` is a thin wrapper around these
functions. Tests can call the same functions directly with fake providers, which
is how the end-to-end smoke tests (Step 15) stay hermetic.
"""

from __future__ import annotations

import gc
import itertools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.check.definition_checker import (
    DefinitionChecker,
)
from consistency_checker.check.definition_judge import (
    DefinitionJudge,
    FixtureDefinitionJudge,
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
    DEFINITION_CONSISTENT_AUTO,
    DEFINITION_INCONSISTENCY_VERDICTS,
)
from consistency_checker.check.providers.moonshot import (
    MoonshotDefinitionProvider,
    MoonshotJudgeProvider,
    MoonshotMultiPartyJudgeProvider,
)
from consistency_checker.check.providers.openai import (
    OpenAIDefinitionProvider,
    OpenAIMultiPartyProvider,
    OpenAIProvider,
)
from consistency_checker.check.triangle import Triangle, find_triangles
from consistency_checker.config import Config
from consistency_checker.corpus.chunker import chunk_document
from consistency_checker.corpus.junk_filter import JunkAudit
from consistency_checker.corpus.loader import load_corpus
from consistency_checker.corpus.ocr import OcrAudit
from consistency_checker.extract.atomic_facts import (
    AnthropicExtractor,
    Extractor,
    FixtureExtractor,
    JunkFilteringExtractor,
    MoonshotExtractor,
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
    n_definition_short_circuited: int = 0
    n_definition_pairs_suppressed: int = 0
    n_cross_corpus_gate_drops: int = 0


# --- factories --------------------------------------------------------------


def make_extractor(config: Config) -> Extractor:
    """Build an extractor from config; ``fixture`` provider returns an empty fixture."""
    if config.judge_provider == "fixture":
        return FixtureExtractor({})
    inner: Extractor
    if config.judge_provider == "moonshot":
        inner = MoonshotExtractor(model="kimi-k2.6")
    else:
        inner = AnthropicExtractor(model=config.judge_model)
    if config.junk_filter_enabled:
        audit = JunkAudit(config.data_dir / "junk_drops.jsonl")
        return JunkFilteringExtractor(inner, audit=audit)
    return inner


def make_embedder(config: Config) -> Embedder:
    return SentenceTransformerEmbedder(model_name=config.embedder_model)


def make_judge(config: Config) -> Judge:
    if config.judge_provider == "anthropic":
        return LLMJudge(AnthropicProvider(model=config.judge_model))
    if config.judge_provider == "openai":
        return LLMJudge(OpenAIProvider(model=config.judge_model))
    if config.judge_provider == "moonshot":
        return LLMJudge(MoonshotJudgeProvider(model="kimi-k2.6"))
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
    if config.judge_provider == "moonshot":
        return LLMMultiPartyJudge(MoonshotMultiPartyJudgeProvider(model="kimi-k2.6"))
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
    if config.judge_provider == "moonshot":
        return LLMDefinitionJudge(MoonshotDefinitionProvider(model="kimi-k2.6"))
    raise ValueError(f"make_definition_judge(): provider {config.judge_provider!r} has no factory.")


def make_definition_checker(config: Config) -> DefinitionChecker:
    return DefinitionChecker(
        judge=make_definition_judge(config),
        org_scope_enabled=config.org_scope_enabled,
    )


# --- ingest -----------------------------------------------------------------


def ingest(
    config: Config,
    *,
    store: AssertionStore,
    faiss_store: FaissStore,
    extractor: Extractor,
    embedder: Embedder,
    corpus_id: str,
) -> IngestResult:
    """Walk corpus_dir → chunks → assertions → embeddings."""
    junk_audit = (
        JunkAudit(config.data_dir / "junk_drops.jsonl") if config.junk_filter_enabled else None
    )
    ocr_audit = OcrAudit(config.data_dir / "ocr_events.jsonl") if config.ocr_enabled else None
    n_docs = n_chunks = n_assertions = 0
    for loaded in load_corpus(
        config.corpus_dir,
        junk_filter_enabled=config.junk_filter_enabled,
        junk_audit=junk_audit,
        ocr_enabled=config.ocr_enabled,
        ocr_audit=ocr_audit,
    ):
        doc = loaded.document
        if config.org_grouping_enabled:
            res = extractor.identify_org(title=doc.title, text=loaded.text)
            doc = replace(doc, org_label=res.label, org_reason=res.reason)
        store.add_document(doc, corpus_id=corpus_id)
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
    if junk_audit is not None and junk_audit.counts:
        _log.info("Junk filter dropped (text stage): %s", junk_audit.counts)
    if ocr_audit is not None and ocr_audit.counts:
        _log.info("OCR fallback: %s", ocr_audit.counts)
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


def _filter_pairs_by_corpus(
    pairs: Iterable[tuple[str, str]], corpus_assertion_ids: set[str]
) -> tuple[list[tuple[str, str]], int]:
    """Drop pairs where either endpoint is outside the corpus's assertion-id set.

    Why: FAISS is a single shared index across all corpora (logical isolation
    per ADR-0013). The top-K gate can return cross-corpus neighbors; without
    this filter, a check on corpus A would leak findings into corpus B.
    """
    kept: list[tuple[str, str]] = []
    dropped = 0
    for a, b in pairs:
        if a in corpus_assertion_ids and b in corpus_assertion_ids:
            kept.append((a, b))
        else:
            dropped += 1
    return kept, dropped


def _resolve_gate(
    config: Config, faiss_store: FaissStore, gate: CandidateGate | None
) -> CandidateGate:
    if gate is None:
        return AnnGate(
            faiss_store,
            top_k=config.gate_top_k,
            similarity_threshold=config.gate_similarity_threshold,
        )
    return gate


def _iter_candidates(
    config: Config, store: AssertionStore, faiss_store: FaissStore, gate: CandidateGate | None
) -> Iterator[CandidatePair]:
    """Stream candidate pairs without materialising the full list.

    The check loop and estimate_cost both consume this once. Streaming caps
    peak memory at O(1) candidate pairs rather than O(N·top_k).
    """
    yield from _resolve_gate(config, faiss_store, gate).candidates(store)


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


def _strong_edge_count(t: Triangle, strong_keys: set[tuple[str, str]]) -> int:
    """Count how many of the triangle's three edges came from the strong gate.

    ``find_triangles`` constructs ``Triangle`` with ``a.assertion_id <
    b.assertion_id < c.assertion_id``, so the edge keys here are already in
    canonical (sorted) form — they match ``strong_keys`` directly.
    """
    edges = [
        (t.a.assertion_id, t.b.assertion_id),
        (t.b.assertion_id, t.c.assertion_id),
        (t.a.assertion_id, t.c.assertion_id),
    ]
    return sum(1 for e in edges if e in strong_keys)


def _run_multi_party_pass(
    pairs: Iterable[CandidatePair],
    *,
    strong_keys: set[tuple[str, str]],
    multi_party_judge: MultiPartyJudge,
    audit_logger: AuditLogger,
    run_id: str,
    max_per_run: int,
) -> tuple[int, int]:
    """Enumerate triangles in the gate graph, judge each, log findings.

    Returns ``(n_triangles_judged, n_multi_party_findings)``. See ADR-0006.

    ``pairs`` may include weak-edge pairs (below the strong gate threshold) so
    that triangles where two strong edges + one weak edge would otherwise be
    missed can still be enumerated. After enumeration, triangles are filtered
    to those with ≥2 strong edges so the LLM judge isn't called on triangles
    held together by only weak similarity signals.
    """
    n_triangles_judged = 0
    n_multi_party_findings = 0
    raw_triangles: list[Triangle] = list(find_triangles(pairs, max_per_run=max_per_run))
    triangles = [t for t in raw_triangles if _strong_edge_count(t, strong_keys) >= 2]
    _log.info(
        "_run_multi_party_pass: %d raw triangles → %d kept (≥2 strong edges)",
        len(raw_triangles),
        len(triangles),
    )
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
    corpus_id: str | None = None,
) -> tuple[int, int, int, int]:
    """Run the definition checker over all stored definitions and log findings.

    Returns ``(n_judged, n_findings, n_short_circuited, n_suppressed)``.
    ``n_judged`` counts every pair the checker emitted a verdict for (judged
    or short-circuited); ``n_short_circuited`` is the subset resolved
    deterministically without an LLM call; ``n_suppressed`` counts cross-org
    pairs dropped by the org-scope gate (when enabled) and persisted with
    ``suppressed=1`` for replay. The NLI gate is bypassed for this stage by
    design.
    """
    definitions = list(store.iter_definitions(corpus_id=corpus_id))
    result = checker.run(definitions)
    n_judged = 0
    n_findings = 0
    n_short_circuited = 0
    for finding in result.findings:
        audit_logger.record_definition_finding(run_id, finding=finding)
        n_judged += 1
        if finding.verdict.verdict == DEFINITION_CONSISTENT_AUTO:
            n_short_circuited += 1
        if finding.verdict.verdict in DEFINITION_INCONSISTENCY_VERDICTS:
            n_findings += 1
    for suppressed in result.suppressed_pairs:
        store.insert_suppressed_finding(
            run_id=run_id,
            assertion_a_id=suppressed.a.assertion_id,
            assertion_b_id=suppressed.b.assertion_id,
        )
    return n_judged, n_findings, n_short_circuited, len(result.suppressed_pairs)


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
    corpus_id: str,
) -> CheckResult:
    """Stage A → Stage B → optional multi-party + definition passes → audit."""
    audit_logger.update_run_status(run_id, "running")

    corpus_assertion_ids: set[str] = set(store.iter_assertion_ids(corpus_id=corpus_id))

    strong_gate = _resolve_gate(config, faiss_store, gate)
    n_pairs_gated = 0
    n_pairs_judged = 0
    n_findings = 0
    n_cross_corpus_drops = 0
    n_assertions = store.stats(corpus_id=corpus_id)["assertions"]
    track_strong_keys = multi_party_judge is not None
    strong_keys: set[tuple[str, str]] = set()

    for pair in strong_gate.candidates(store):
        if (
            pair.a.assertion_id not in corpus_assertion_ids
            or pair.b.assertion_id not in corpus_assertion_ids
        ):
            n_cross_corpus_drops += 1
            continue
        n_pairs_gated += 1
        if track_strong_keys:
            strong_keys.add((pair.a.assertion_id, pair.b.assertion_id))
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

    # The NLI checker isn't used by the multi-party LLM-judge pass or the
    # definition pass — release its weights now so the ~0.6-2 GB of model RSS
    # is reclaimed before those passes allocate their own request buffers.
    nli_checker.release()
    gc.collect()

    # ADR-0006 F4: optional triangle pass over the gate output.
    # Dual-gate: feed the triangle finder a union of strong pairs (used by the
    # pairwise judge above) plus weak pairs (triangle construction only).
    # After enumeration we filter to triangles with ≥2 strong edges so the LLM
    # judge isn't called on weak-only triangles. See v0.4.1 plan, Task 2.
    n_triangles_judged = 0
    n_multi_party_findings = 0
    if multi_party_judge is not None:
        weak_gate = AnnGate(
            faiss_store,
            top_k=config.triangle_weak_top_k,
            similarity_threshold=config.triangle_weak_threshold,
        )
        # Re-iterate the strong gate — calling .candidates(store) a second
        # time triggers a fresh FAISS top-k scan (the gate does not memoise);
        # cheap enough not to cache. itertools.chain keeps both streams lazy.
        # Both streams must be corpus-filtered for the same reason the strong
        # pairwise loop above filters: FAISS is shared across corpora, so the
        # gate can return cross-corpus neighbours that would otherwise leak
        # triangles between corpora. See ADR-0013 and PR #65 follow-up.
        strong_pairs_iter = (
            p
            for p in strong_gate.candidates(store)
            if p.a.assertion_id in corpus_assertion_ids and p.b.assertion_id in corpus_assertion_ids
        )
        weak_pairs_iter = (
            p
            for p in weak_gate.candidates(store)
            if p.a.assertion_id in corpus_assertion_ids
            and p.b.assertion_id in corpus_assertion_ids
            and (p.a.assertion_id, p.b.assertion_id) not in strong_keys
        )
        triangle_pairs = itertools.chain(strong_pairs_iter, weak_pairs_iter)
        n_triangles_judged, n_multi_party_findings = _run_multi_party_pass(
            triangle_pairs,
            strong_keys=strong_keys,
            multi_party_judge=multi_party_judge,
            audit_logger=audit_logger,
            run_id=run_id,
            max_per_run=config.max_triangles_per_run,
        )

    # ADR-0009: definition-inconsistency detector. Term-grouping replaces the
    # NLI gate; bypasses the contradiction pipeline entirely.
    n_definition_pairs_judged = 0
    n_definition_findings = 0
    n_definition_short_circuited = 0
    n_definition_pairs_suppressed = 0
    if definition_checker is not None:
        (
            n_definition_pairs_judged,
            n_definition_findings,
            n_definition_short_circuited,
            n_definition_pairs_suppressed,
        ) = _run_definition_pass(
            store=store,
            checker=definition_checker,
            audit_logger=audit_logger,
            run_id=run_id,
            corpus_id=corpus_id,
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
        "%d multi-party / %d definition pairs / %d definition findings / "
        "%d short-circuited",
        run_id,
        n_pairs_gated,
        n_pairs_judged,
        n_findings,
        n_triangles_judged,
        n_multi_party_findings,
        n_definition_pairs_judged,
        n_definition_findings,
        n_definition_short_circuited,
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
        n_definition_short_circuited=n_definition_short_circuited,
        n_definition_pairs_suppressed=n_definition_pairs_suppressed,
        n_cross_corpus_gate_drops=n_cross_corpus_drops,
    )


# --- cost estimate ----------------------------------------------------------


def estimate_cost(
    config: Config,
    *,
    store: AssertionStore,
    faiss_store: FaissStore,
    per_call_low: float = 0.003,
    per_call_high: float = 0.010,
    corpus_id: str | None = None,
) -> CostEstimate:
    """Count judge calls a check run would make, without making any LLM calls.

    NLI is skipped because (a) it requires the ~800 MB DeBERTa model
    download and (b) the user wants a fast pre-flight estimate, not a
    real-cost figure. Real spend is usually 30-70% lower than this
    ceiling because NLI filters most gate-pass pairs.

    When ``corpus_id`` is supplied the same FAISS post-filter applied by
    ``check()`` is used so the preview matches the actual run.
    """
    corpus_assertion_ids: set[str] | None = (
        set(store.iter_assertion_ids(corpus_id=corpus_id)) if corpus_id is not None else None
    )

    # Stream the candidates: PR #58 made _iter_candidates an iterator for
    # OOM protection on large corpora; materialising it here would regress
    # that. Apply the corpus filter inline as we count.
    n_candidate_pairs = 0
    for pair in _iter_candidates(config, store, faiss_store, gate=None):
        if corpus_assertion_ids is not None and (
            pair.a.assertion_id not in corpus_assertion_ids
            or pair.b.assertion_id not in corpus_assertion_ids
        ):
            continue
        n_candidate_pairs += 1

    # Route through DefinitionChecker so org-scope suppression matches the run.
    # The judge is a no-op stand-in — count_pairs never invokes it, and we
    # don't want to require API keys for a cost preview.
    counter = DefinitionChecker(
        judge=FixtureDefinitionJudge(fixtures={}),
        org_scope_enabled=config.org_scope_enabled,
    )
    n_definition_pairs = counter.count_pairs(list(store.iter_definitions(corpus_id=corpus_id)))

    judge_calls_ceiling = n_candidate_pairs + n_definition_pairs
    return CostEstimate(
        n_assertions=store.stats(corpus_id=corpus_id)["assertions"],
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
