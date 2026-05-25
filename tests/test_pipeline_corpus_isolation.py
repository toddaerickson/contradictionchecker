"""Integration test: pipeline.check enforces corpus isolation at the FAISS gate.

Two corpora each contain one assertion with identical text.  FAISS top-K
will pair them.  With corpus_id=a the cross-corpus pair must be dropped
before reaching the judge; only intra-corpus pairs may produce findings.
"""

from __future__ import annotations

from pathlib import Path

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.check.gate import AllPairsGate
from consistency_checker.check.llm_judge import FixtureJudge
from consistency_checker.check.nli_checker import FixtureNliChecker
from consistency_checker.config import Config
from consistency_checker.extract.schema import Assertion, Document
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.index.embedder import embed_pending
from consistency_checker.index.faiss_store import FaissStore
from consistency_checker.pipeline import check
from tests.conftest import HashEmbedder


def _setup(tmp_path: Path) -> tuple[Config, AssertionStore, FaissStore, str, str]:
    """Return (cfg, store, faiss, corpus_id_a, corpus_id_b) with two corpora."""
    cfg = Config(
        corpus_dir=tmp_path / "corpus",
        judge_provider="fixture",
        judge_model="test",
        data_dir=tmp_path / "store",
        log_dir=tmp_path / "logs",
        embedder_model="hash",
        nli_model="fixture",
        gate_similarity_threshold=-1.0,
        nli_contradiction_threshold=0.0,
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    store = AssertionStore(cfg.db_path)
    store.migrate()

    cid_a = store.get_or_create_corpus("corpus_a", "/a", "moonshot")
    cid_b = store.get_or_create_corpus("corpus_b", "/b", "moonshot")

    doc_a = Document(doc_id="dA", source_path="/a/doc.txt")
    doc_b = Document(doc_id="dB", source_path="/b/doc.txt")
    store.add_document(doc_a, corpus_id=cid_a)
    store.add_document(doc_b, corpus_id=cid_b)

    # Identical text → FAISS will pair them as nearest neighbours.
    text = "Revenue grew 12% in 2023."
    aa = Assertion.build("dA", text)
    ab = Assertion.build("dB", text)
    store.add_assertion(aa)
    store.add_assertion(ab)

    embedder = HashEmbedder(dim=64)
    faiss = FaissStore.open_or_create(
        index_path=cfg.faiss_path,
        id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
        dim=embedder.dim,
    )
    embed_pending(store, faiss, embedder)
    return cfg, store, faiss, cid_a, cid_b


def test_check_does_not_judge_cross_corpus_pairs(tmp_path: Path) -> None:
    """The cross-corpus pair must be dropped before reaching the judge."""
    cfg, store, faiss, cid_a, _ = _setup(tmp_path)

    audit = AuditLogger(store)
    run_id = audit.begin_run()
    result = check(
        cfg,
        store=store,
        faiss_store=faiss,
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
        audit_logger=audit,
        gate=AllPairsGate(allow_same_document=False),
        run_id=run_id,
        corpus_id=cid_a,
    )

    assert result.n_cross_corpus_gate_drops >= 1, (
        "the cross-corpus pair was NOT reported as dropped"
    )
    # No intra-corpus pairs exist for corpus_a (only one doc with one assertion).
    assert result.n_pairs_gated == 0
    store.close()


def test_check_result_exposes_drop_count(tmp_path: Path) -> None:
    """n_cross_corpus_gate_drops is present on CheckResult and increments."""
    cfg, store, faiss, cid_a, _ = _setup(tmp_path)

    audit = AuditLogger(store)
    run_id = audit.begin_run()
    result = check(
        cfg,
        store=store,
        faiss_store=faiss,
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
        audit_logger=audit,
        gate=AllPairsGate(allow_same_document=False),
        run_id=run_id,
        corpus_id=cid_a,
    )

    assert isinstance(result.n_cross_corpus_gate_drops, int)
    assert result.n_cross_corpus_gate_drops >= 1
    store.close()


def test_check_corpus_b_sees_only_its_own_assertions(tmp_path: Path) -> None:
    """The same cross-corpus drop must fire when corpus_b is checked."""
    cfg, store, faiss, _, cid_b = _setup(tmp_path)

    audit = AuditLogger(store)
    run_id = audit.begin_run()
    result = check(
        cfg,
        store=store,
        faiss_store=faiss,
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
        audit_logger=audit,
        gate=AllPairsGate(allow_same_document=False),
        run_id=run_id,
        corpus_id=cid_b,
    )

    assert result.n_cross_corpus_gate_drops >= 1
    assert result.n_pairs_gated == 0
    store.close()
