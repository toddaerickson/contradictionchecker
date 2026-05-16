"""Tests for the pre-flight cost-estimate command."""

from __future__ import annotations

from pathlib import Path

import pytest

from consistency_checker.config import Config
from consistency_checker.extract.schema import Assertion, Document
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.index.embedder import embed_pending
from consistency_checker.index.faiss_store import FaissStore
from consistency_checker.pipeline import estimate_cost
from tests.conftest import HashEmbedder


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        corpus_dir=tmp_path / "corpus",
        judge_provider="fixture",
        judge_model="test",
        data_dir=tmp_path / "store",
        log_dir=tmp_path / "logs",
        embedder_model="hash",
        nli_model="fixture",
        gate_similarity_threshold=-1.0,
    )


def _seed_store(cfg: Config) -> tuple[AssertionStore, FaissStore]:
    store = AssertionStore(cfg.db_path)
    store.migrate()
    embedder = HashEmbedder(dim=64)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    faiss = FaissStore.open_or_create(
        index_path=cfg.faiss_path,
        id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
        dim=embedder.dim,
    )
    doc_a = Document.from_content("A.", source_path="a.md")
    doc_b = Document.from_content("B.", source_path="b.md")
    store.add_document(doc_a)
    store.add_document(doc_b)
    a1 = Assertion.build(doc_a.doc_id, "Revenue grew 12%.")
    a2 = Assertion.build(doc_b.doc_id, "Revenue declined 5%.")
    d1 = Assertion.build(
        doc_a.doc_id, '"MAE" means A.', kind="definition", term="MAE", definition_text="A"
    )
    d2 = Assertion.build(
        doc_b.doc_id, '"MAE" means B.', kind="definition", term="MAE", definition_text="B"
    )
    singleton = Assertion.build(
        doc_a.doc_id,
        '"Lender" means First Bank.',
        kind="definition",
        term="Lender",
        definition_text="First Bank",
    )
    store.add_assertions([a1, a2, d1, d2, singleton])
    embed_pending(store, faiss, embedder)
    return store, faiss


def test_estimate_cost_counts_gate_pairs_and_definitions(cfg: Config) -> None:
    store, faiss = _seed_store(cfg)
    est = estimate_cost(cfg, store=store, faiss_store=faiss)
    store.close()
    # Five assertions seeded.
    assert est.n_assertions == 5
    # Gate at threshold -1.0 + top_k default permits some candidate pairs.
    assert est.n_candidate_pairs > 0
    # MAE has two definitions → 1 pair; Lender singleton → 0. Total: 1.
    assert est.n_definition_pairs == 1
    assert est.judge_calls_ceiling == est.n_candidate_pairs + 1


def test_estimate_cost_singleton_definitions_contribute_zero(cfg: Config) -> None:
    store = AssertionStore(cfg.db_path)
    store.migrate()
    embedder = HashEmbedder(dim=64)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    faiss = FaissStore.open_or_create(
        index_path=cfg.faiss_path,
        id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
        dim=embedder.dim,
    )
    doc = Document.from_content("A.", source_path="a.md")
    store.add_document(doc)
    store.add_assertion(
        Assertion.build(
            doc.doc_id,
            '"Lender" means First Bank.',
            kind="definition",
            term="Lender",
            definition_text="First Bank",
        )
    )
    embed_pending(store, faiss, embedder)
    est = estimate_cost(cfg, store=store, faiss_store=faiss)
    store.close()
    assert est.n_definition_pairs == 0


def test_estimate_cost_dollars_use_supplied_bounds(cfg: Config) -> None:
    store, faiss = _seed_store(cfg)
    est = estimate_cost(
        cfg, store=store, faiss_store=faiss, per_call_low=0.001, per_call_high=0.020
    )
    store.close()
    assert est.per_call_low == 0.001
    assert est.per_call_high == 0.020
    assert est.est_cost_low == pytest.approx(est.judge_calls_ceiling * 0.001)
    assert est.est_cost_high == pytest.approx(est.judge_calls_ceiling * 0.020)


def test_estimate_cost_empty_store_returns_zero(cfg: Config) -> None:
    store = AssertionStore(cfg.db_path)
    store.migrate()
    embedder = HashEmbedder(dim=64)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    faiss = FaissStore.open_or_create(
        index_path=cfg.faiss_path,
        id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
        dim=embedder.dim,
    )
    est = estimate_cost(cfg, store=store, faiss_store=faiss)
    store.close()
    assert est.n_assertions == 0
    assert est.n_candidate_pairs == 0
    assert est.n_definition_pairs == 0
    assert est.judge_calls_ceiling == 0
    assert est.est_cost_low == 0.0
    assert est.est_cost_high == 0.0


def test_estimate_cost_multiple_term_groups(cfg: Config) -> None:
    """Two terms with 2 and 3 definitions respectively => 1 + 3 = 4 def pairs."""
    store = AssertionStore(cfg.db_path)
    store.migrate()
    embedder = HashEmbedder(dim=64)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    faiss = FaissStore.open_or_create(
        index_path=cfg.faiss_path,
        id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
        dim=embedder.dim,
    )
    doc = Document.from_content("Body.", source_path="doc.md")
    store.add_document(doc)
    store.add_assertions(
        [
            Assertion.build(
                doc.doc_id, '"MAE" means A.', kind="definition", term="MAE", definition_text="A"
            ),
            Assertion.build(
                doc.doc_id, '"MAE" means B.', kind="definition", term="MAE", definition_text="B"
            ),
            Assertion.build(
                doc.doc_id,
                '"Borrower" means X.',
                kind="definition",
                term="Borrower",
                definition_text="X",
            ),
            Assertion.build(
                doc.doc_id,
                '"Borrower" means Y.',
                kind="definition",
                term="Borrower",
                definition_text="Y",
            ),
            Assertion.build(
                doc.doc_id,
                '"Borrower" means Z.',
                kind="definition",
                term="Borrower",
                definition_text="Z",
            ),
        ]
    )
    embed_pending(store, faiss, embedder)
    est = estimate_cost(cfg, store=store, faiss_store=faiss)
    store.close()
    # C(2,2) + C(3,2) = 1 + 3 = 4
    assert est.n_definition_pairs == 4
