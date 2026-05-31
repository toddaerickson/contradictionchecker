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
from tests.conftest import HashEmbedder, strip_ansi


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    # pairwise_enabled=True so the existing gate-counting tests continue to
    # exercise the FAISS candidate path; ADR-0015 made False the global default.
    return Config(
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
    _cid = store.get_or_create_corpus("test", "/test", "moonshot")
    doc_a = Document.from_content("A.", source_path="a.md")
    doc_b = Document.from_content("B.", source_path="b.md")
    store.add_document(doc_a, corpus_id=_cid)
    store.add_document(doc_b, corpus_id=_cid)
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
    _cid = store.get_or_create_corpus("test", "/test", "moonshot")
    doc = Document.from_content("A.", source_path="a.md")
    store.add_document(doc, corpus_id=_cid)
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


def test_estimate_cost_uses_moonshot_defaults_when_provider_moonshot(cfg: Config) -> None:
    store, faiss = _seed_store(cfg)
    cfg_m = cfg.model_copy(update={"judge_provider": "moonshot"})
    est = estimate_cost(cfg_m, store=store, faiss_store=faiss)
    store.close()
    assert est.per_call_low == 0.0001
    assert est.per_call_high == 0.001


def test_estimate_cost_explicit_overrides_win(cfg: Config) -> None:
    store, faiss = _seed_store(cfg)
    cfg_m = cfg.model_copy(update={"judge_provider": "moonshot"})
    est = estimate_cost(cfg_m, store=store, faiss_store=faiss, per_call_low=0.5, per_call_high=0.6)
    store.close()
    assert est.per_call_low == 0.5
    assert est.per_call_high == 0.6


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
    _cid = store.get_or_create_corpus("test", "/test", "moonshot")
    doc = Document.from_content("Body.", source_path="doc.md")
    store.add_document(doc, corpus_id=_cid)
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


def test_estimate_cost_does_not_count_cross_corpus_pairs(tmp_path: Path) -> None:
    """estimate_cost with corpus_id=a excludes the cross-corpus FAISS pair."""
    store = AssertionStore(tmp_path / "store.db")
    store.migrate()
    embedder = HashEmbedder(dim=64)
    (tmp_path / "store").mkdir(parents=True, exist_ok=True)
    faiss = FaissStore.open_or_create(
        index_path=tmp_path / "store" / "faiss.idx",
        id_map_path=tmp_path / "store" / "faiss.idmap.json",
        dim=embedder.dim,
    )
    cid_a = store.get_or_create_corpus("corp_a", "/a", "moonshot")
    cid_b = store.get_or_create_corpus("corp_b", "/b", "moonshot")

    doc_a = Document(doc_id="dA", source_path="/a/doc.txt")
    doc_b = Document(doc_id="dB", source_path="/b/doc.txt")
    store.add_document(doc_a, corpus_id=cid_a)
    store.add_document(doc_b, corpus_id=cid_b)

    text = "Revenue grew 12% in 2023."
    aa = Assertion.build("dA", text)
    ab = Assertion.build("dB", text)
    store.add_assertion(aa)
    store.add_assertion(ab)
    embed_pending(store, faiss, embedder)

    cfg_obj = Config(
        corpus_dir=tmp_path,
        judge_provider="fixture",
        judge_model="test",
        data_dir=tmp_path / "store",
        log_dir=tmp_path / "logs",
        embedder_model="hash",
        nli_model="fixture",
        gate_similarity_threshold=-1.0,
        pairwise_enabled=True,
    )

    est_no_filter = estimate_cost(cfg_obj, store=store, faiss_store=faiss)
    est_corpus_a = estimate_cost(cfg_obj, store=store, faiss_store=faiss, corpus_id=cid_a)

    # Without corpus filter the cross-corpus pair is counted.
    assert est_no_filter.n_candidate_pairs >= 1
    # With corpus_a filter the cross-corpus pair is excluded (corpus_a has
    # only one assertion, so no intra-corpus pairs).
    assert est_corpus_a.n_candidate_pairs == 0
    store.close()


def test_estimate_cost_zero_candidate_pairs_when_pairwise_disabled(cfg: Config) -> None:
    """ADR-0015: with pairwise_enabled=False the cost preview reports zero
    candidate pairs (no FAISS scan), but still counts definition pairs."""
    store, faiss = _seed_store(cfg)

    cfg_off = cfg.model_copy(update={"pairwise_enabled": False})
    cfg_on = cfg.model_copy(update={"pairwise_enabled": True})

    est_off = estimate_cost(cfg_off, store=store, faiss_store=faiss)
    est_on = estimate_cost(cfg_on, store=store, faiss_store=faiss)
    store.close()

    # Off: no candidate pairs, definition pairs unchanged, ceiling = def pairs.
    assert est_off.n_candidate_pairs == 0
    assert est_off.n_definition_pairs == 1
    assert est_off.judge_calls_ceiling == est_off.n_definition_pairs
    # On: candidate pairs come back; sanity-check against the off run.
    assert est_on.n_candidate_pairs > 0
    assert est_on.n_definition_pairs == est_off.n_definition_pairs


def test_estimate_cost_excludes_cross_org_pairs_when_scope_enabled(cfg: Config) -> None:
    store = AssertionStore(cfg.db_path)
    store.migrate()
    embedder = HashEmbedder(dim=64)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    faiss = FaissStore.open_or_create(
        index_path=cfg.faiss_path,
        id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
        dim=embedder.dim,
    )
    _cid = store.get_or_create_corpus("test", "/test", "moonshot")
    doc_a = Document.from_content("A.", source_path="a.md", org_label="Acme")
    doc_b = Document.from_content("B.", source_path="b.md", org_label="Beta")
    store.add_document(doc_a, corpus_id=_cid)
    store.add_document(doc_b, corpus_id=_cid)
    store.add_assertions(
        [
            Assertion.build(
                doc_a.doc_id,
                '"Director" means a member.',
                kind="definition",
                term="Director",
                definition_text="a member",
            ),
            Assertion.build(
                doc_b.doc_id,
                '"Director" means a manager.',
                kind="definition",
                term="Director",
                definition_text="a manager",
            ),
        ]
    )
    embed_pending(store, faiss, embedder)

    cfg_off = cfg.model_copy(update={"org_scope_enabled": False})
    cfg_on = cfg.model_copy(update={"org_scope_enabled": True})

    off = estimate_cost(cfg_off, store=store, faiss_store=faiss)
    on = estimate_cost(cfg_on, store=store, faiss_store=faiss)
    store.close()

    assert off.n_definition_pairs == 1
    assert on.n_definition_pairs == 0


# --- Task 8: estimate-cost CLI --corpus required ----------------------------


def test_estimate_cost_cli_without_corpus_errors_in_non_tty(tmp_path: Path) -> None:
    """estimate-cost without --corpus fails with '--corpus is required' when not a TTY."""
    from typer.testing import CliRunner

    from consistency_checker.cli.main import app
    from consistency_checker.index.assertion_store import AssertionStore
    from consistency_checker.index.faiss_store import FaissStore

    cfg = tmp_path / "config.yml"
    cfg.write_text(f"corpus_dir: {tmp_path}\ndata_dir: {tmp_path}\n", encoding="utf-8")

    # Seed a minimal FAISS index so the command doesn't fail with the missing-index error.
    # With data_dir=tmp_path, faiss_path is tmp_path/assertions.faiss.
    store = AssertionStore(tmp_path / "assertions.db")
    store.migrate()
    store.close()
    fs = FaissStore.open_or_create(
        index_path=tmp_path / "assertions.faiss",
        id_map_path=tmp_path / "assertions.idmap.json",
        dim=64,
    )
    fs.save()

    runner = CliRunner()
    res = runner.invoke(app, ["estimate-cost", "--config", str(cfg)])
    assert res.exit_code != 0
    out = strip_ansi((res.output or "") + str(res.exception or ""))
    assert "--corpus is required" in out


def test_estimate_cost_cli_with_corpus_passes_corpus_id(tmp_path: Path) -> None:
    """estimate-cost --corpus <name> passes corpus_id to the pipeline function."""
    from typing import Any

    from typer.testing import CliRunner

    from consistency_checker.cli.main import app
    from consistency_checker.config import Config

    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(
        f"corpus_dir: {tmp_path}\ndata_dir: {tmp_path}\n"
        "embedder_model: hash\nnli_model: fixture\ngate_similarity_threshold: -1.0\n",
        encoding="utf-8",
    )
    cfg_obj = Config.from_yaml(cfg_path)

    # Seed store + FAISS.

    store_s, _faiss_s = _seed_store(cfg_obj)
    atkins_id = store_s.get_or_create_corpus("atkins", str(tmp_path), "moonshot")
    store_s.close()

    captured: dict[str, Any] = {}

    def fake_estimate_cost(_cfg: Any, **kwargs: Any) -> Any:
        captured["corpus_id"] = kwargs.get("corpus_id")
        from consistency_checker.pipeline import CostEstimate

        return CostEstimate(
            n_assertions=0,
            n_candidate_pairs=0,
            n_definition_pairs=0,
            judge_calls_ceiling=0,
            est_cost_low=0.0,
            est_cost_high=0.0,
            per_call_low=0.003,
            per_call_high=0.010,
        )

    import consistency_checker.cli.main as cli_main

    runner = CliRunner()
    from unittest.mock import patch

    with patch.object(cli_main, "run_estimate_cost", fake_estimate_cost):
        res = runner.invoke(app, ["estimate-cost", "--config", str(cfg_path), "--corpus", "atkins"])
    assert res.exit_code == 0, res.output
    assert captured["corpus_id"] == atkins_id
