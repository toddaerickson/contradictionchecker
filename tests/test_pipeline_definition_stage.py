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
    store.add_document(Document(doc_id="docA", source_path="/A.txt"))
    store.add_document(Document(doc_id="docB", source_path="/B.txt"))
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
    a, b = definitions[0], definitions[1]
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
