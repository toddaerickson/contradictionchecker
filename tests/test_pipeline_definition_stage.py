"""Test the definition-inconsistency stage routing in pipeline.check()."""

from __future__ import annotations

from pathlib import Path

import pytest

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.check.definition_checker import DefinitionChecker
from consistency_checker.check.definition_judge import (
    DefinitionJudgeVerdict,
    FixtureDefinitionJudge,
)
from consistency_checker.check.llm_judge import FixtureJudge
from consistency_checker.check.nli_checker import FixtureNliChecker
from consistency_checker.config import Config
from consistency_checker.extract.schema import Assertion, Document
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.index.faiss_store import FaissStore
from consistency_checker.pipeline import check


@pytest.fixture
def stocked_store(tmp_path: Path) -> AssertionStore:
    store = AssertionStore(tmp_path / "store.db")
    store.migrate()
    _cid = store.get_or_create_corpus("test", "/test", "moonshot")
    store.add_document(Document(doc_id="docA", source_path="/A.txt"), corpus_id=_cid)
    store.add_document(Document(doc_id="docB", source_path="/B.txt"), corpus_id=_cid)
    a = Assertion.build(
        "docA", '"MAE" means A.', kind="definition", term="MAE", definition_text="A"
    )
    b = Assertion.build(
        "docB", '"MAE" means B.', kind="definition", term="MAE", definition_text="B"
    )
    store.add_assertions([a, b])
    return store


def _config(tmp_path: Path) -> Config:
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


def test_check_skips_definition_stage_when_checker_is_none(
    tmp_path: Path, stocked_store: AssertionStore
) -> None:
    config = _config(tmp_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    faiss = FaissStore.open_or_create(
        index_path=config.data_dir / "faiss.idx",
        id_map_path=config.data_dir / "faiss.idmap.json",
        dim=64,
    )
    logger = AuditLogger(stocked_store)
    run_id = logger.begin_run()
    result = check(
        config,
        store=stocked_store,
        faiss_store=faiss,
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
        audit_logger=logger,
        run_id=run_id,
    )
    assert result.n_definition_pairs_judged == 0
    assert result.n_definition_findings == 0


def test_check_runs_definition_stage_and_logs_findings(
    tmp_path: Path, stocked_store: AssertionStore
) -> None:
    config = _config(tmp_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    faiss = FaissStore.open_or_create(
        index_path=config.data_dir / "faiss.idx",
        id_map_path=config.data_dir / "faiss.idmap.json",
        dim=64,
    )
    logger = AuditLogger(stocked_store)
    run_id = logger.begin_run()

    definitions = list(stocked_store.iter_definitions())
    a, b = definitions[0][0], definitions[1][0]
    key = (min(a.assertion_id, b.assertion_id), max(a.assertion_id, b.assertion_id))
    fixture_verdict = DefinitionJudgeVerdict(
        assertion_a_id=key[0],
        assertion_b_id=key[1],
        verdict="definition_divergent",
        confidence=0.9,
        rationale="A vs B",
        evidence_spans=["A", "B"],
    )
    definition_checker = DefinitionChecker(judge=FixtureDefinitionJudge({key: fixture_verdict}))

    result = check(
        config,
        store=stocked_store,
        faiss_store=faiss,
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
        audit_logger=logger,
        run_id=run_id,
        definition_checker=definition_checker,
    )

    assert result.n_definition_pairs_judged == 1
    assert result.n_definition_findings == 1

    rows = stocked_store._conn.execute(
        "SELECT detector_type, judge_verdict FROM findings WHERE run_id = ? "
        "AND detector_type = 'definition_inconsistency'",
        (run_id,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["judge_verdict"] == "definition_divergent"


def test_check_uncertain_definition_does_not_increment_findings(
    tmp_path: Path, stocked_store: AssertionStore
) -> None:
    config = _config(tmp_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    faiss = FaissStore.open_or_create(
        index_path=config.data_dir / "faiss.idx",
        id_map_path=config.data_dir / "faiss.idmap.json",
        dim=64,
    )
    logger = AuditLogger(stocked_store)
    run_id = logger.begin_run()
    definition_checker = DefinitionChecker(judge=FixtureDefinitionJudge({}))

    result = check(
        config,
        store=stocked_store,
        faiss_store=faiss,
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
        audit_logger=logger,
        run_id=run_id,
        definition_checker=definition_checker,
    )

    assert result.n_definition_pairs_judged == 1
    assert result.n_definition_findings == 0  # uncertain does not count
    rows = stocked_store._conn.execute(
        "SELECT judge_verdict FROM findings WHERE run_id = ? "
        "AND detector_type = 'definition_inconsistency'",
        (run_id,),
    ).fetchall()
    assert len(rows) == 1  # still persisted (audit replay)
    assert rows[0]["judge_verdict"] == "uncertain"


def test_check_counts_definition_short_circuits(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    store = AssertionStore(tmp_path / "store.db")
    store.migrate()
    _cid = store.get_or_create_corpus("test", "/test", "moonshot")
    store.add_document(Document(doc_id="docA", source_path="/A.txt"), corpus_id=_cid)
    store.add_document(Document(doc_id="docB", source_path="/B.txt"), corpus_id=_cid)
    text = "the board of directors of the Corporation"
    a = Assertion.build(
        "docA", f'"Board" means {text}.', kind="definition", term="Board", definition_text=text
    )
    b = Assertion.build(
        "docB", f'"Board" means {text}.', kind="definition", term="Board", definition_text=text
    )
    store.add_assertions([a, b])

    faiss = FaissStore.open_or_create(
        index_path=config.data_dir / "faiss.idx",
        id_map_path=config.data_dir / "faiss.idmap.json",
        dim=64,
    )
    logger = AuditLogger(store)
    run_id = logger.begin_run()

    class _RaisingJudge:
        def judge(self, a, b):  # type: ignore[no-untyped-def]
            raise AssertionError("identical definitions must short-circuit, not reach the judge")

    result = check(
        config,
        store=store,
        faiss_store=faiss,
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
        audit_logger=logger,
        run_id=run_id,
        definition_checker=DefinitionChecker(judge=_RaisingJudge()),
    )

    assert result.n_definition_short_circuited == 1
    assert result.n_definition_findings == 0  # consistent_auto is not a finding
    assert result.n_definition_pairs_judged == 1  # the pair was still processed
    rows = store._conn.execute(
        "SELECT judge_verdict FROM findings WHERE run_id = ? "
        "AND detector_type = 'definition_inconsistency'",
        (run_id,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["judge_verdict"] == "definition_consistent_auto"


def _ingest_config(tmp_path: Path) -> Config:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    data = tmp_path / "store"
    data.mkdir()
    return Config(
        corpus_dir=corpus,
        judge_provider="fixture",
        judge_model="test",
        data_dir=data,
        log_dir=tmp_path / "logs",
        embedder_model="hash",
        nli_model="fixture",
        gate_similarity_threshold=-1.0,
        junk_filter_enabled=False,
    )


def test_ingest_populates_org_label_via_fixture_extractor(tmp_path: Path) -> None:
    from consistency_checker.extract.atomic_facts import (
        FixtureExtractor,
        OrgIdentification,
    )
    from consistency_checker.pipeline import ingest
    from tests.conftest import HashEmbedder

    cfg = _ingest_config(tmp_path).model_copy(update={"org_grouping_enabled": True})
    doc_path = cfg.corpus_dir / "acme_bylaws.txt"
    doc_path.write_text("Bylaws of Acme Foundation, Inc.\n\nArticle I. ...", encoding="utf-8")

    extractor = FixtureExtractor(
        fixtures={},
        org_fixtures={
            ("acme_bylaws", "Bylaws of Acme"): OrgIdentification(
                label="Acme Foundation, Inc.", reason="org_found"
            ),
        },
    )
    store = AssertionStore(cfg.data_dir / "ingest.db")
    store.migrate()
    embedder = HashEmbedder(dim=64)
    faiss = FaissStore.open_or_create(
        index_path=cfg.data_dir / "faiss.idx",
        id_map_path=cfg.data_dir / "faiss.idmap.json",
        dim=embedder.dim,
    )

    ingest(cfg, store=store, faiss_store=faiss, extractor=extractor, embedder=embedder)

    docs = list(store.iter_documents())
    assert len(docs) == 1
    assert docs[0].org_label == "Acme Foundation, Inc."
    assert docs[0].org_reason == "org_found"
    store.close()


def test_ingest_skips_org_identification_when_disabled(tmp_path: Path) -> None:
    from consistency_checker.extract.atomic_facts import FixtureExtractor
    from consistency_checker.pipeline import ingest
    from tests.conftest import HashEmbedder

    cfg = _ingest_config(tmp_path).model_copy(update={"org_grouping_enabled": False})
    (cfg.corpus_dir / "x.txt").write_text("anything", encoding="utf-8")

    store = AssertionStore(cfg.data_dir / "ingest.db")
    store.migrate()
    embedder = HashEmbedder(dim=64)
    faiss = FaissStore.open_or_create(
        index_path=cfg.data_dir / "faiss.idx",
        id_map_path=cfg.data_dir / "faiss.idmap.json",
        dim=embedder.dim,
    )

    ingest(cfg, store=store, faiss_store=faiss, extractor=FixtureExtractor({}), embedder=embedder)

    docs = list(store.iter_documents())
    assert len(docs) == 1
    assert docs[0].org_label is None
    assert docs[0].org_reason is None
    store.close()
