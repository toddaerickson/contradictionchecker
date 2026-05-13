"""Tests for the embedder and FAISS sidecar."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from consistency_checker.extract.schema import Assertion, Document
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.index.embedder import (
    SentenceTransformerEmbedder,
    embed_pending,
    rebuild_index,
)
from consistency_checker.index.faiss_store import FaissStore
from tests.conftest import HashEmbedder

# --- FaissStore (no embedder dependency) ------------------------------------


def test_faiss_store_creates_empty(tmp_path: Path) -> None:
    fs = FaissStore.open_or_create(
        index_path=tmp_path / "x.faiss",
        id_map_path=tmp_path / "x.json",
        dim=8,
    )
    assert len(fs) == 0
    assert fs.dim == 8


def test_faiss_store_add_returns_assigned_rows(tmp_path: Path) -> None:
    fs = FaissStore.open_or_create(
        index_path=tmp_path / "x.faiss",
        id_map_path=tmp_path / "x.json",
        dim=4,
    )
    vecs = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], dtype=np.float32)
    rows = fs.add(["a", "b"], vecs)
    assert rows == [0, 1]
    assert len(fs) == 2


def test_faiss_store_dim_mismatch_raises(tmp_path: Path) -> None:
    fs = FaissStore.open_or_create(
        index_path=tmp_path / "x.faiss",
        id_map_path=tmp_path / "x.json",
        dim=4,
    )
    bad = np.zeros((1, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="vector dim"):
        fs.add(["a"], bad)


def test_faiss_store_size_mismatch_raises(tmp_path: Path) -> None:
    fs = FaissStore.open_or_create(
        index_path=tmp_path / "x.faiss",
        id_map_path=tmp_path / "x.json",
        dim=4,
    )
    vecs = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="batch size"):
        fs.add(["only_one"], vecs)


def test_faiss_store_save_and_reload(tmp_path: Path) -> None:
    index_path = tmp_path / "i.faiss"
    id_map_path = tmp_path / "m.json"
    fs = FaissStore.open_or_create(index_path=index_path, id_map_path=id_map_path, dim=4)
    vecs = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], dtype=np.float32)
    fs.add(["a", "b"], vecs)
    fs.save()

    reopened = FaissStore.open_or_create(index_path=index_path, id_map_path=id_map_path, dim=4)
    assert len(reopened) == 2
    results = reopened.search(np.array([1.0, 0, 0, 0], dtype=np.float32), k=1)
    assert results[0][0][0] == "a"


def test_faiss_store_search_returns_self_top1(tmp_path: Path) -> None:
    """A vector identical to one in the index must rank itself first."""
    fs = FaissStore.open_or_create(
        index_path=tmp_path / "x.faiss",
        id_map_path=tmp_path / "x.json",
        dim=4,
    )
    vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    fs.add(["a", "b", "c"], vecs)
    results = fs.search(np.array([1, 0, 0, 0], dtype=np.float32), k=3)
    assert results[0][0][0] == "a"
    assert results[0][0][1] == pytest.approx(1.0, abs=1e-5)


def test_faiss_store_search_empty_index_returns_empty(tmp_path: Path) -> None:
    fs = FaissStore.open_or_create(
        index_path=tmp_path / "x.faiss",
        id_map_path=tmp_path / "x.json",
        dim=4,
    )
    results = fs.search(np.array([1, 0, 0, 0], dtype=np.float32), k=5)
    assert results == [[]]


def test_faiss_store_dim_check_on_reopen(tmp_path: Path) -> None:
    """Re-opening with a different dim than was persisted must fail loudly."""
    index_path = tmp_path / "i.faiss"
    id_map_path = tmp_path / "m.json"
    fs = FaissStore.open_or_create(index_path=index_path, id_map_path=id_map_path, dim=4)
    fs.add(["a"], np.array([[1, 0, 0, 0]], dtype=np.float32))
    fs.save()
    with pytest.raises(ValueError, match="dim"):
        FaissStore.open_or_create(index_path=index_path, id_map_path=id_map_path, dim=8)


# --- embed_pending integration ---------------------------------------------


def _seed_store(tmp_path: Path, n: int) -> AssertionStore:
    store = AssertionStore(tmp_path / "store.db")
    store.migrate()
    doc = Document.from_content("Body text.", source_path="d.txt")
    store.add_document(doc)
    for i in range(n):
        text = f"Assertion number {i} about widgets."
        store.add_assertion(Assertion.build(doc.doc_id, text))
    return store


def test_embed_pending_attaches_faiss_rows(tmp_path: Path) -> None:
    store = _seed_store(tmp_path, n=5)
    embedder = HashEmbedder(dim=64)
    faiss_store = FaissStore.open_or_create(
        index_path=tmp_path / "i.faiss",
        id_map_path=tmp_path / "m.json",
        dim=embedder.dim,
    )
    count = embed_pending(store, faiss_store, embedder)
    assert count == 5
    embedded = [a for a in store.iter_assertions() if a.faiss_row is not None]
    assert len(embedded) == 5
    assert {a.faiss_row for a in embedded} == {0, 1, 2, 3, 4}


def test_embed_pending_is_resumable(tmp_path: Path) -> None:
    """Running embed_pending twice must not re-embed already-embedded assertions."""
    store = _seed_store(tmp_path, n=3)
    embedder = HashEmbedder(dim=64)
    faiss_store = FaissStore.open_or_create(
        index_path=tmp_path / "i.faiss",
        id_map_path=tmp_path / "m.json",
        dim=embedder.dim,
    )
    first = embed_pending(store, faiss_store, embedder)
    second = embed_pending(store, faiss_store, embedder)
    assert first == 3
    assert second == 0
    assert len(faiss_store) == 3


def test_embed_pending_empty_store_is_noop(tmp_path: Path) -> None:
    store = AssertionStore(tmp_path / "store.db")
    store.migrate()
    embedder = HashEmbedder(dim=64)
    faiss_store = FaissStore.open_or_create(
        index_path=tmp_path / "i.faiss",
        id_map_path=tmp_path / "m.json",
        dim=embedder.dim,
    )
    assert embed_pending(store, faiss_store, embedder) == 0


def test_embed_pending_dim_mismatch_raises(tmp_path: Path) -> None:
    store = _seed_store(tmp_path, n=1)
    embedder = HashEmbedder(dim=64)
    faiss_store = FaissStore.open_or_create(
        index_path=tmp_path / "i.faiss",
        id_map_path=tmp_path / "m.json",
        dim=32,
    )
    with pytest.raises(ValueError, match="dim"):
        embed_pending(store, faiss_store, embedder)


def test_embed_pending_round_trip_via_search(tmp_path: Path) -> None:
    """An assertion's own text must retrieve its own assertion id as top-1."""
    store = _seed_store(tmp_path, n=4)
    embedder = HashEmbedder(dim=64)
    faiss_store = FaissStore.open_or_create(
        index_path=tmp_path / "i.faiss",
        id_map_path=tmp_path / "m.json",
        dim=embedder.dim,
    )
    embed_pending(store, faiss_store, embedder)

    target = next(iter(store.iter_assertions()))
    query_vec = embedder.embed_texts([target.assertion_text])[0]
    [results] = faiss_store.search(query_vec, k=1)
    assert results[0][0] == target.assertion_id


def test_rebuild_index_from_sqlite_matches_original(tmp_path: Path) -> None:
    """Rebuilding the FAISS index from SQLite produces identical search results."""
    store = _seed_store(tmp_path, n=5)
    embedder = HashEmbedder(dim=64)

    first_index = tmp_path / "a.faiss"
    first_map = tmp_path / "a.json"
    fs1 = FaissStore.open_or_create(index_path=first_index, id_map_path=first_map, dim=embedder.dim)
    embed_pending(store, fs1, embedder)

    # Wipe rebuild on a fresh index path.
    second_index = tmp_path / "b.faiss"
    second_map = tmp_path / "b.json"
    fs2 = FaissStore.open_or_create(
        index_path=second_index, id_map_path=second_map, dim=embedder.dim
    )
    n = rebuild_index(store, fs2, embedder)
    assert n == 5
    assert len(fs1) == len(fs2)

    sample = next(iter(store.iter_assertions()))
    qv = embedder.embed_texts([sample.assertion_text])[0]
    r1 = fs1.search(qv, k=1)
    r2 = fs2.search(qv, k=1)
    assert r1[0][0][0] == r2[0][0][0]


# --- live model test (gated) ------------------------------------------------


@pytest.mark.slow
def test_sentence_transformer_paraphrase_top_1(tmp_path: Path) -> None:
    """With a real model, a paraphrase of an assertion should rank the source top-1."""
    if os.environ.get("CC_SKIP_HF_DOWNLOAD") == "1":
        pytest.skip("HF download skipped by env flag")

    embedder = SentenceTransformerEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = AssertionStore(tmp_path / "store.db")
    store.migrate()
    doc = Document.from_content("body", source_path="d.txt")
    store.add_document(doc)
    claims = [
        "Revenue grew twelve percent in fiscal 2025.",
        "The cat sat on the mat.",
        "Customer satisfaction scores held at 4.7 out of 5.",
    ]
    for text in claims:
        store.add_assertion(Assertion.build(doc.doc_id, text))

    fs = FaissStore.open_or_create(
        index_path=tmp_path / "i.faiss",
        id_map_path=tmp_path / "m.json",
        dim=embedder.dim,
    )
    embed_pending(store, fs, embedder)

    qv = embedder.embed_texts(["Fiscal 2025 revenue rose 12 percent."])[0]
    [results] = fs.search(qv, k=1)
    top_id = results[0][0]
    matched = store.get_assertion(top_id)
    assert matched is not None
    assert "Revenue grew" in matched.assertion_text
