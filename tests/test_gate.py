"""Tests for the candidate-pair gate (Stage A entry)."""

from __future__ import annotations

from math import comb
from pathlib import Path

import pytest

from consistency_checker.check.gate import AllPairsGate, AnnGate
from consistency_checker.extract.schema import Assertion, Document
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.index.embedder import embed_pending
from consistency_checker.index.faiss_store import FaissStore
from tests.conftest import HashEmbedder


def _add_doc(store: AssertionStore, source_path: str, *texts: str) -> Document:
    body = " ".join(texts) or source_path
    doc = Document.from_content(body, source_path=source_path)
    store.add_document(doc)
    for text in texts:
        store.add_assertion(Assertion.build(doc.doc_id, text))
    return doc


def _build_indexed_store(tmp_path: Path) -> tuple[AssertionStore, FaissStore]:
    store = AssertionStore(tmp_path / "store.db")
    store.migrate()
    _add_doc(
        store,
        "alpha.md",
        "Revenue from Alpha grew 12% in fiscal 2025.",
        "The Alpha team shipped in Q1 2025.",
        "Alpha customers are enterprises.",
    )
    _add_doc(
        store,
        "beta.txt",
        "Revenue from Alpha declined 5% in fiscal 2025.",
        "The Beta initiative began in 2024.",
        "Customer satisfaction held at 4.7.",
    )
    embedder = HashEmbedder(dim=64)
    fs = FaissStore.open_or_create(
        index_path=tmp_path / "i.faiss",
        id_map_path=tmp_path / "m.json",
        dim=embedder.dim,
    )
    embed_pending(store, fs, embedder)
    return store, fs


# --- AllPairsGate -----------------------------------------------------------


def test_all_pairs_gate_returns_exact_count_across_documents(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "s.db")
    store.migrate()
    _add_doc(store, "a.txt", "Claim a1.", "Claim a2.")
    _add_doc(store, "b.txt", "Claim b1.", "Claim b2.")
    pairs = list(AllPairsGate().candidates(store))
    # 4 assertions, 2 per doc — same-document pairs excluded by default → 2*2 = 4 pairs.
    assert len(pairs) == 4
    for p in pairs:
        assert p.a.doc_id != p.b.doc_id
        assert p.a.assertion_id < p.b.assertion_id  # canonical order


def test_all_pairs_gate_includes_same_doc_when_allowed(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "s.db")
    store.migrate()
    _add_doc(store, "a.txt", "Claim a1.", "Claim a2.", "Claim a3.")
    _add_doc(store, "b.txt", "Claim b1.", "Claim b2.")
    pairs = list(AllPairsGate(allow_same_document=True).candidates(store))
    # 5 assertions total → C(5, 2) = 10 pairs when same-doc is allowed.
    assert len(pairs) == comb(5, 2)


def test_all_pairs_gate_canonical_order(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "s.db")
    store.migrate()
    _add_doc(store, "a.txt", "x.")
    _add_doc(store, "b.txt", "y.")
    [pair] = AllPairsGate().candidates(store)
    assert pair.a.assertion_id < pair.b.assertion_id


def test_all_pairs_gate_score_constant(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "s.db")
    store.migrate()
    _add_doc(store, "a.txt", "x.")
    _add_doc(store, "b.txt", "y.")
    [pair] = AllPairsGate().candidates(store)
    assert pair.score == 1.0


def test_all_pairs_gate_empty_store(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "s.db")
    store.migrate()
    assert list(AllPairsGate().candidates(store)) == []


# --- AnnGate ----------------------------------------------------------------


def test_ann_gate_validates_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        AnnGate(faiss_store=None, top_k=0)  # type: ignore[arg-type]


def test_ann_gate_validates_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        AnnGate(faiss_store=None, similarity_threshold=1.5)  # type: ignore[arg-type]


def test_ann_gate_dedupes_pairs(tmp_path: Path) -> None:
    """Each unordered pair must be emitted at most once even when both endpoints surface it."""
    store, fs = _build_indexed_store(tmp_path)
    pairs = list(AnnGate(fs, top_k=10, similarity_threshold=0.0).candidates(store))
    seen: set[tuple[str, str]] = set()
    for p in pairs:
        key = (p.a.assertion_id, p.b.assertion_id)
        assert key not in seen
        seen.add(key)


def test_ann_gate_excludes_same_document_by_default(tmp_path: Path) -> None:
    store, fs = _build_indexed_store(tmp_path)
    pairs = list(AnnGate(fs, top_k=10, similarity_threshold=0.0).candidates(store))
    assert all(p.a.doc_id != p.b.doc_id for p in pairs)


def test_ann_gate_can_include_same_document(tmp_path: Path) -> None:
    store, fs = _build_indexed_store(tmp_path)
    pairs = list(
        AnnGate(fs, top_k=10, similarity_threshold=0.0, allow_same_document=True).candidates(store)
    )
    same_doc = [p for p in pairs if p.a.doc_id == p.b.doc_id]
    assert same_doc, "expected at least one intra-document pair when allow_same_document=True"


def test_ann_gate_respects_threshold(tmp_path: Path) -> None:
    """A threshold near 1.0 must exclude pairs of distinct (non-identical) hashed vectors."""
    store, fs = _build_indexed_store(tmp_path)
    pairs = list(AnnGate(fs, top_k=10, similarity_threshold=0.99).candidates(store))
    assert pairs == []


def test_ann_gate_score_propagates_similarity(tmp_path: Path) -> None:
    store, fs = _build_indexed_store(tmp_path)
    pairs = list(AnnGate(fs, top_k=10, similarity_threshold=0.0).candidates(store))
    assert pairs
    for p in pairs:
        assert -1.0 <= p.score <= 1.0


def test_ann_gate_bounded_by_n_times_k(tmp_path: Path) -> None:
    """Across N assertions, AnnGate must emit at most N * top_k pairs."""
    store, fs = _build_indexed_store(tmp_path)
    n = sum(1 for _ in store.iter_assertions())
    top_k = 3
    pairs = list(AnnGate(fs, top_k=top_k, similarity_threshold=0.0).candidates(store))
    assert len(pairs) <= n * top_k


def test_ann_gate_skips_unembedded_assertions(tmp_path: Path) -> None:
    """Assertions without faiss_row must be skipped, not crash."""
    store = AssertionStore(tmp_path / "s.db")
    store.migrate()
    _add_doc(store, "a.txt", "Claim a1.")
    _add_doc(store, "b.txt", "Claim b1.")
    # No embedding pass — faiss_row is None for everything.
    embedder = HashEmbedder(dim=64)
    fs = FaissStore.open_or_create(
        index_path=tmp_path / "i.faiss",
        id_map_path=tmp_path / "m.json",
        dim=embedder.dim,
    )
    pairs = list(AnnGate(fs, top_k=5, similarity_threshold=0.0).candidates(store))
    assert pairs == []


def test_ann_gate_canonical_order(tmp_path: Path) -> None:
    store, fs = _build_indexed_store(tmp_path)
    for p in AnnGate(fs, top_k=10, similarity_threshold=0.0).candidates(store):
        assert p.a.assertion_id < p.b.assertion_id
