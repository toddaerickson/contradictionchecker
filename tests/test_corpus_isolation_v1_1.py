"""Regression tests for the three PR #65 follow-up bugs.

Each scored 75 (real, verified) but landed below the 80 auto-post threshold of
the original cross-task review:

1. ``pipeline.check`` triangle / multi-party pass bypassed the FAISS corpus
   filter under ``--deep``; with two corpora and one cross-corpus FAISS
   neighbour the strong pairwise loop dropped it but the triangle stream
   still saw it.
2. ``AssertionStore.add_document`` used ``INSERT OR IGNORE``; doc_id is a
   content hash so ingesting the same file into corpus B after corpus A
   silently left the doc under corpus A and reported success.
3. ``pipeline.estimate_cost`` did ``raw_pairs = list(_iter_candidates(...))``;
   PR #58 made ``_iter_candidates`` a lazy iterator for OOM protection and
   PR #65 silently regressed that for the estimate-cost path.
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path

import pytest

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.check.gate import AllPairsGate, CandidatePair
from consistency_checker.check.llm_judge import FixtureJudge
from consistency_checker.check.multi_party_judge import (
    FixtureMultiPartyJudge,
    MultiPartyJudgeVerdict,
)
from consistency_checker.check.nli_checker import FixtureNliChecker
from consistency_checker.config import Config
from consistency_checker.extract.schema import Assertion, Document
from consistency_checker.index.assertion_store import (
    AssertionStore,
    CrossCorpusDocumentError,
)
from consistency_checker.index.embedder import embed_pending
from consistency_checker.index.faiss_store import FaissStore
from consistency_checker.pipeline import check as run_check
from consistency_checker.pipeline import estimate_cost
from tests.conftest import HashEmbedder

# --- Bug 1: triangle pass corpus filter -------------------------------------


def _two_corpus_triangle_setup(
    tmp_path: Path,
) -> tuple[Config, AssertionStore, FaissStore, str, str, list[Assertion]]:
    """Two corpora: A has 3 assertions (can form one triangle); B has 1 assertion
    that AllPairsGate will pair with every A assertion (would form 3 cross-corpus
    triangles if the corpus filter were skipped)."""
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
        triangle_weak_top_k=10,
        triangle_weak_threshold=-1.0,
        pairwise_enabled=True,
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    store = AssertionStore(cfg.db_path)
    store.migrate()
    cid_a = store.get_or_create_corpus("a", "/a", "moonshot")
    cid_b = store.get_or_create_corpus("b", "/b", "moonshot")
    docs_a = [
        Document.from_content("A1.", source_path="a1.md", title="A1"),
        Document.from_content("A2.", source_path="a2.md", title="A2"),
        Document.from_content("A3.", source_path="a3.md", title="A3"),
    ]
    doc_b = Document.from_content("B1.", source_path="b1.md", title="B1")
    for d in docs_a:
        store.add_document(d, corpus_id=cid_a)
    store.add_document(doc_b, corpus_id=cid_b)
    assertions = [
        Assertion.build(docs_a[0].doc_id, "All employees get four weeks vacation."),
        Assertion.build(docs_a[1].doc_id, "Engineers are employees."),
        Assertion.build(docs_a[2].doc_id, "Engineers get two weeks vacation."),
        Assertion.build(doc_b.doc_id, "Contractors get one week vacation."),
    ]
    store.add_assertions(assertions)
    embedder = HashEmbedder(dim=64)
    fs = FaissStore.open_or_create(
        index_path=cfg.faiss_path,
        id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
        dim=embedder.dim,
    )
    embed_pending(store, fs, embedder)
    return cfg, store, fs, cid_a, cid_b, assertions


def test_triangle_pass_drops_cross_corpus_assertions(tmp_path: Path) -> None:
    """Triangle / multi-party pass must apply the same FAISS post-filter as the
    pairwise loop.  AllPairsGate makes every pair a candidate; without the
    filter ``check(corpus_id=A)`` would enumerate triangles that include B's
    assertion."""
    cfg, store, fs, cid_a, _cid_b, assertions = _two_corpus_triangle_setup(tmp_path)

    intra_corpus_triangle = tuple(sorted(a.assertion_id for a in assertions[:3]))
    cross_corpus_triangles_judged: list[tuple[str, ...]] = []

    class RecordingJudge(FixtureMultiPartyJudge):
        def judge(self, triangle):  # type: ignore[override]
            ids = tuple(sorted(triangle.assertion_ids))
            cross_corpus_triangles_judged.append(ids)
            return MultiPartyJudgeVerdict(
                assertion_ids=ids,
                verdict="uncertain",
                confidence=0.0,
                rationale="",
                contradicting_subset=(),
                evidence_spans=[],
            )

    audit = AuditLogger(store)
    run_id = audit.begin_run()
    result = run_check(
        cfg,
        store=store,
        faiss_store=fs,
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
        audit_logger=audit,
        gate=AllPairsGate(),
        multi_party_judge=RecordingJudge({}),
        run_id=run_id,
        corpus_id=cid_a,
    )

    b_id = assertions[3].assertion_id
    leaked = [t for t in cross_corpus_triangles_judged if b_id in t]
    assert leaked == [], f"triangle pass leaked cross-corpus triangles: {leaked}"
    assert cross_corpus_triangles_judged == [intra_corpus_triangle]
    assert result.n_triangles_judged == 1
    store.close()


# --- Bug 2: add_document INSERT OR IGNORE -----------------------------------


def _two_corpus_store(tmp_path: Path) -> tuple[AssertionStore, str, str]:
    cfg = Config(
        corpus_dir=tmp_path / "corpus",
        judge_provider="fixture",
        judge_model="test",
        data_dir=tmp_path / "store",
        log_dir=tmp_path / "logs",
        embedder_model="hash",
        nli_model="fixture",
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    store = AssertionStore(cfg.db_path)
    store.migrate()
    cid_a = store.get_or_create_corpus("a", "/a", "moonshot")
    cid_b = store.get_or_create_corpus("b", "/b", "moonshot")
    return store, cid_a, cid_b


def test_add_document_raises_on_cross_corpus_duplicate(tmp_path: Path) -> None:
    """Same doc_id under two different corpora must raise, not silently misroute."""
    store, cid_a, cid_b = _two_corpus_store(tmp_path)
    doc = Document.from_content("shared content", source_path="shared.md", title="Shared")
    store.add_document(doc, corpus_id=cid_a)

    with pytest.raises(CrossCorpusDocumentError) as excinfo:
        store.add_document(doc, corpus_id=cid_b)

    assert excinfo.value.doc_id == doc.doc_id
    assert excinfo.value.existing_corpus_id == cid_a
    assert excinfo.value.requested_corpus_id == cid_b

    row = store._conn.execute(
        "SELECT corpus_id FROM documents WHERE doc_id = ?", (doc.doc_id,)
    ).fetchone()
    assert row["corpus_id"] == cid_a, "doc must remain under the original corpus"
    store.close()


def test_add_document_is_idempotent_within_same_corpus(tmp_path: Path) -> None:
    """Re-ingesting the same file into the same corpus stays a no-op (no raise)."""
    store, cid_a, _cid_b = _two_corpus_store(tmp_path)
    doc = Document.from_content("shared content", source_path="shared.md", title="Shared")
    store.add_document(doc, corpus_id=cid_a)
    store.add_document(doc, corpus_id=cid_a)  # must not raise
    row_count = store._conn.execute(
        "SELECT COUNT(*) AS n FROM documents WHERE doc_id = ?", (doc.doc_id,)
    ).fetchone()
    assert row_count["n"] == 1
    store.close()


# --- Bug 3: estimate_cost streaming -----------------------------------------


def test_estimate_cost_streams_candidates_without_materialising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: PR #58 made ``_iter_candidates`` lazy for OOM protection on
    large corpora; PR #65 silently regressed that with ``list(...)``. Verify the
    streaming shape via peak memory under a long synthetic stream."""
    cfg = Config(
        corpus_dir=tmp_path / "corpus",
        judge_provider="fixture",
        judge_model="test",
        data_dir=tmp_path / "store",
        log_dir=tmp_path / "logs",
        embedder_model="hash",
        nli_model="fixture",
        gate_similarity_threshold=-1.0,
        pairwise_enabled=True,
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    store = AssertionStore(cfg.db_path)
    store.migrate()
    cid_a = store.get_or_create_corpus("a", "/a", "moonshot")
    doc = Document.from_content("body", source_path="doc.md", title="Doc")
    store.add_document(doc, corpus_id=cid_a)
    assertions = [
        Assertion.build(doc.doc_id, "first claim."),
        Assertion.build(doc.doc_id, "second claim."),
    ]
    store.add_assertions(assertions)
    embedder = HashEmbedder(dim=64)
    fs = FaissStore.open_or_create(
        index_path=cfg.faiss_path,
        id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
        dim=embedder.dim,
    )
    embed_pending(store, fs, embedder)

    one_pair = CandidatePair(a=assertions[0], b=assertions[1], score=1.0)
    n = 50_000

    def huge_stream(*_args: object, **_kw: object):
        for _ in range(n):
            yield one_pair

    monkeypatch.setattr("consistency_checker.pipeline._iter_candidates", huge_stream)

    tracemalloc.start()
    estimate = estimate_cost(cfg, store=store, faiss_store=fs, corpus_id=cid_a)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert estimate.n_candidate_pairs == n
    # 50k CandidatePair objects in a list would be several MB; streaming holds
    # at most a constant number of references. 1 MB is a generous ceiling that
    # catches a list() regression without being flaky on Python version drift.
    assert peak < 1_000_000, f"peak memory {peak} bytes suggests list() materialisation"
    store.close()
