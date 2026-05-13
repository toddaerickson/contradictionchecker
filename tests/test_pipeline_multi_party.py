"""Pipeline integration tests for the multi-party (triangle) pass — F4.

Exercises the wiring between ``find_triangles`` (F2), ``MultiPartyJudge``
(F3), and ``record_multi_party_finding`` (F1) through ``pipeline.check``.
All hermetic — uses FixtureMultiPartyJudge so no network calls.
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
from consistency_checker.config import Config
from consistency_checker.extract.schema import Assertion, Document
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.index.embedder import embed_pending
from consistency_checker.index.faiss_store import FaissStore
from consistency_checker.pipeline import check as run_check
from tests.conftest import HashEmbedder


@pytest.fixture
def three_doc_store(tmp_path: Path) -> tuple[Config, AssertionStore, FaissStore, list[Assertion]]:
    """A 3-document corpus modelling the canonical vacation-policy triangle."""
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


# --- baseline: pass disabled ---------------------------------------------


def test_check_without_multi_party_judge_skips_triangle_pass(
    three_doc_store: tuple[Config, AssertionStore, FaissStore, list[Assertion]],
) -> None:
    cfg, store, fs, _ = three_doc_store
    audit_logger = AuditLogger(store)
    result = run_check(
        cfg,
        store=store,
        faiss_store=fs,
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
        audit_logger=audit_logger,
        gate=AllPairsGate(),
    )
    assert result.n_triangles_judged == 0
    assert result.n_multi_party_findings == 0
    assert list(audit_logger.iter_multi_party_findings(run_id=result.run_id)) == []
    store.close()


# --- enabled: detects + records the canonical triangle -------------------


def test_check_with_multi_party_judge_records_finding(
    three_doc_store: tuple[Config, AssertionStore, FaissStore, list[Assertion]],
) -> None:
    cfg, store, fs, assertions = three_doc_store
    audit_logger = AuditLogger(store)

    triangle_ids = tuple(sorted(a.assertion_id for a in assertions))
    multi_party_judge = FixtureMultiPartyJudge(
        {
            triangle_ids: MultiPartyJudgeVerdict(
                assertion_ids=triangle_ids,
                verdict="multi_party_contradiction",
                confidence=0.88,
                rationale="A ∧ B ⇒ ¬C — engineers should get both 4w and 2w.",
                contradicting_subset=("A", "B", "C"),
                evidence_spans=["four weeks", "two weeks"],
            )
        }
    )

    result = run_check(
        cfg,
        store=store,
        faiss_store=fs,
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
        audit_logger=audit_logger,
        gate=AllPairsGate(),
        multi_party_judge=multi_party_judge,
    )

    assert result.n_triangles_judged == 1
    assert result.n_multi_party_findings == 1
    findings = list(audit_logger.iter_multi_party_findings(run_id=result.run_id))
    assert len(findings) == 1
    f = findings[0]
    assert f.judge_verdict == "multi_party_contradiction"
    assert f.judge_confidence == pytest.approx(0.88)
    assert sorted(f.assertion_ids) == list(triangle_ids)
    assert len({d for d in f.doc_ids}) == 3
    assert len(f.triangle_edge_scores) == 3
    store.close()


def test_check_with_multi_party_judge_returns_uncertain_does_not_count_finding(
    three_doc_store: tuple[Config, AssertionStore, FaissStore, list[Assertion]],
) -> None:
    """An ``uncertain`` triangle verdict is logged but not counted as a finding."""
    cfg, store, fs, _ = three_doc_store
    audit_logger = AuditLogger(store)

    # Fixture judge has no fixtures → defaults to uncertain for the only triangle.
    multi_party_judge = FixtureMultiPartyJudge({})

    result = run_check(
        cfg,
        store=store,
        faiss_store=fs,
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
        audit_logger=audit_logger,
        gate=AllPairsGate(),
        multi_party_judge=multi_party_judge,
    )
    assert result.n_triangles_judged == 1
    assert result.n_multi_party_findings == 0
    all_findings = list(audit_logger.iter_multi_party_findings(run_id=result.run_id))
    assert len(all_findings) == 1
    assert all_findings[0].judge_verdict == "uncertain"
    store.close()


# --- max_triangles_per_run respected -------------------------------------


def test_check_respects_max_triangles_per_run(tmp_path: Path) -> None:
    """max_triangles_per_run=0 silences the multi-party pass even when judge is given."""
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
        max_triangles_per_run=0,
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    store = AssertionStore(cfg.db_path)
    store.migrate()
    docs = [Document.from_content(f"Body {i}.", source_path=f"d{i}.md") for i in range(3)]
    for d in docs:
        store.add_document(d)
    assertions = [Assertion.build(d.doc_id, f"text {i}") for i, d in enumerate(docs)]
    store.add_assertions(assertions)
    embedder = HashEmbedder(dim=64)
    fs = FaissStore.open_or_create(
        index_path=cfg.faiss_path,
        id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
        dim=embedder.dim,
    )
    embed_pending(store, fs, embedder)
    audit_logger = AuditLogger(store)

    result = run_check(
        cfg,
        store=store,
        faiss_store=fs,
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
        audit_logger=audit_logger,
        gate=AllPairsGate(),
        multi_party_judge=FixtureMultiPartyJudge({}),
    )
    assert result.n_triangles_judged == 0
    assert result.n_multi_party_findings == 0
    store.close()


# --- CheckResult defaults -----------------------------------------------


def test_check_result_defaults_to_zero_triangle_counters() -> None:
    """v0.1 callers that don't pass multi_party_judge see zeroed counters."""
    from consistency_checker.pipeline import CheckResult

    r = CheckResult(run_id="r", n_assertions=0, n_pairs_gated=0, n_pairs_judged=0, n_findings=0)
    assert r.n_triangles_judged == 0
    assert r.n_multi_party_findings == 0
