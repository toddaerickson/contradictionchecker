"""Pipeline tests for the v0.4.1 dual-gate weak-edge triangle recovery.

The triangle pass takes a *union* of strong (pairwise-judge) pairs and weak
pairs (above a lower threshold), enumerates triangles, then filters to those
with ≥2 strong edges so the LLM judge isn't called on triangles held together
by weak similarity alone.

These tests verify:

1. The dual-gate code path runs end-to-end without crashing.
2. The degenerate case (``triangle_weak_threshold`` ≥ strong threshold) yields
   the same triangle count as if the weak gate were disabled — weak_pairs is
   empty and the strong-edge filter is a no-op.
3. The ≥2-strong-edges filter actually fires when a triangle has only weak
   support (verified via the ``_strong_edge_count`` helper directly).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.check.gate import AllPairsGate
from consistency_checker.check.llm_judge import FixtureJudge
from consistency_checker.check.multi_party_judge import (
    FixtureMultiPartyJudge,
    MultiPartyJudgeVerdict,
)
from consistency_checker.check.nli_checker import FixtureNliChecker
from consistency_checker.check.triangle import Triangle
from consistency_checker.config import Config
from consistency_checker.extract.schema import Assertion, Document
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.index.embedder import embed_pending
from consistency_checker.index.faiss_store import FaissStore
from consistency_checker.pipeline import _strong_edge_count
from consistency_checker.pipeline import check as run_check
from tests.conftest import HashEmbedder


def _build_three_doc_store(
    tmp_path: Path,
    *,
    triangle_weak_top_k: int = 50,
    triangle_weak_threshold: float = 0.5,
) -> tuple[Config, AssertionStore, FaissStore, list[Assertion]]:
    cfg = Config(
        corpus_dir=tmp_path / "corpus",
        judge_provider="fixture",
        judge_model="test",
        data_dir=tmp_path / "store",
        log_dir=tmp_path / "logs",
        embedder_model="hash",
        nli_model="fixture",
        gate_top_k=10,
        gate_similarity_threshold=-1.0,
        nli_contradiction_threshold=0.0,
        triangle_weak_top_k=triangle_weak_top_k,
        triangle_weak_threshold=triangle_weak_threshold,
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    store = AssertionStore(cfg.db_path)
    store.migrate()
    docs = [
        Document.from_content("Policy A.", source_path="policy_a.md", title="Policy A"),
        Document.from_content("Policy B.", source_path="policy_b.md", title="Policy B"),
        Document.from_content("Policy C.", source_path="policy_c.md", title="Policy C"),
    ]
    for d in docs:
        store.add_document(d)
    assertions = [
        Assertion.build(docs[0].doc_id, "All employees get four weeks vacation."),
        Assertion.build(docs[1].doc_id, "Engineers are employees."),
        Assertion.build(docs[2].doc_id, "Engineers get two weeks vacation."),
    ]
    store.add_assertions(assertions)
    embedder = HashEmbedder(dim=64)
    fs = FaissStore.open_or_create(
        index_path=cfg.faiss_path,
        id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
        dim=embedder.dim,
    )
    embed_pending(store, fs, embedder)
    return cfg, store, fs, assertions


def test_dual_gate_codepath_runs_without_crash(tmp_path: Path) -> None:
    """End-to-end smoke: check() with a multi-party judge exercises the dual gate."""
    cfg, store, fs, assertions = _build_three_doc_store(tmp_path)
    audit_logger = AuditLogger(store)

    triangle_ids = tuple(sorted(a.assertion_id for a in assertions))
    multi_party_judge = FixtureMultiPartyJudge(
        {
            triangle_ids: MultiPartyJudgeVerdict(
                assertion_ids=triangle_ids,
                verdict="multi_party_contradiction",
                confidence=0.9,
                rationale="A ∧ B ⇒ ¬C",
                contradicting_subset=("A", "B", "C"),
                evidence_spans=["four weeks", "two weeks"],
            )
        }
    )

    run_id = audit_logger.begin_run()
    result = run_check(
        cfg,
        store=store,
        faiss_store=fs,
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
        audit_logger=audit_logger,
        gate=AllPairsGate(),
        multi_party_judge=multi_party_judge,
        run_id=run_id,
    )
    # AllPairsGate marks every cross-doc pair as strong (score=1.0), so all
    # three edges of the only triangle are strong → it survives the ≥2 filter.
    assert result.n_triangles_judged == 1
    assert result.n_multi_party_findings == 1
    store.close()


def test_weak_threshold_at_or_above_strong_is_degenerate(tmp_path: Path) -> None:
    """When the weak threshold is ≥ the strong gate's max score, weak_pairs is empty.

    Result should match a baseline run where no weak gate was active — i.e.
    triangle count is identical to the v0.4.0 behaviour.
    """
    # AllPairsGate emits score=1.0 for every cross-doc pair; setting the weak
    # threshold to 1.0 means no weak pair clears it. The dual-gate union
    # collapses back to just the strong pairs.
    cfg, store, fs, assertions = _build_three_doc_store(
        tmp_path, triangle_weak_top_k=5, triangle_weak_threshold=1.0
    )
    audit_logger = AuditLogger(store)

    triangle_ids = tuple(sorted(a.assertion_id for a in assertions))
    multi_party_judge = FixtureMultiPartyJudge(
        {
            triangle_ids: MultiPartyJudgeVerdict(
                assertion_ids=triangle_ids,
                verdict="multi_party_contradiction",
                confidence=0.9,
                rationale="A ∧ B ⇒ ¬C",
                contradicting_subset=("A", "B", "C"),
                evidence_spans=["four weeks", "two weeks"],
            )
        }
    )

    run_id = audit_logger.begin_run()
    result = run_check(
        cfg,
        store=store,
        faiss_store=fs,
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
        audit_logger=audit_logger,
        gate=AllPairsGate(),
        multi_party_judge=multi_party_judge,
        run_id=run_id,
    )
    # Same triangle as the baseline test — all edges are strong, all survive.
    assert result.n_triangles_judged == 1
    assert result.n_multi_party_findings == 1
    store.close()


def test_strong_edge_count_filters_weak_only_triangles() -> None:
    """The ≥2-strong-edges filter rejects triangles supported only by weak edges."""
    a = Assertion.build("doc1", "alpha")
    b = Assertion.build("doc2", "beta")
    c = Assertion.build("doc3", "gamma")
    # Canonicalise so triangle field order matches find_triangles' invariant.
    sorted_a, sorted_b, sorted_c = sorted([a, b, c], key=lambda x: x.assertion_id)
    triangle = Triangle(
        a=sorted_a,
        b=sorted_b,
        c=sorted_c,
        edge_scores=(
            (sorted_a.assertion_id, sorted_b.assertion_id, 0.6),
            (sorted_a.assertion_id, sorted_c.assertion_id, 0.6),
            (sorted_b.assertion_id, sorted_c.assertion_id, 0.6),
        ),
    )

    # No strong edges → 0 → would be filtered out.
    assert _strong_edge_count(triangle, set()) == 0

    # Only one strong edge → still filtered out by the ≥2 rule.
    one_strong = {(sorted_a.assertion_id, sorted_b.assertion_id)}
    assert _strong_edge_count(triangle, one_strong) == 1

    # Two strong edges → survives the filter.
    two_strong = {
        (sorted_a.assertion_id, sorted_b.assertion_id),
        (sorted_b.assertion_id, sorted_c.assertion_id),
    }
    assert _strong_edge_count(triangle, two_strong) == 2

    # All three strong.
    all_strong = {
        (sorted_a.assertion_id, sorted_b.assertion_id),
        (sorted_a.assertion_id, sorted_c.assertion_id),
        (sorted_b.assertion_id, sorted_c.assertion_id),
    }
    assert _strong_edge_count(triangle, all_strong) == 3


def test_dual_gate_with_uncertain_judge(tmp_path: Path) -> None:
    """``uncertain`` triangle verdicts are still logged but not counted as findings."""
    cfg, store, fs, _ = _build_three_doc_store(tmp_path)
    audit_logger = AuditLogger(store)

    run_id = audit_logger.begin_run()
    result = run_check(
        cfg,
        store=store,
        faiss_store=fs,
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
        audit_logger=audit_logger,
        gate=AllPairsGate(),
        multi_party_judge=FixtureMultiPartyJudge({}),  # defaults to uncertain
        run_id=run_id,
    )
    assert result.n_triangles_judged == 1
    assert result.n_multi_party_findings == 0
    store.close()


@pytest.mark.parametrize("weak_top_k,weak_threshold", [(5, 0.9), (50, 0.5), (1, -1.0)])
def test_dual_gate_config_variants_dont_crash(
    tmp_path: Path, weak_top_k: int, weak_threshold: float
) -> None:
    """A small grid of weak-gate configs should all run cleanly."""
    cfg, store, fs, _ = _build_three_doc_store(
        tmp_path, triangle_weak_top_k=weak_top_k, triangle_weak_threshold=weak_threshold
    )
    audit_logger = AuditLogger(store)

    run_id = audit_logger.begin_run()
    result = run_check(
        cfg,
        store=store,
        faiss_store=fs,
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
        audit_logger=audit_logger,
        gate=AllPairsGate(),
        multi_party_judge=FixtureMultiPartyJudge({}),
        run_id=run_id,
    )
    # The fixture corpus has exactly one cross-doc triangle. With AllPairsGate
    # as the strong gate, all three edges are strong regardless of how the
    # weak gate is configured, so the ≥2-strong filter never trips here.
    assert result.n_triangles_judged == 1
    store.close()
