"""Tests for the pipeline orchestration module.

Verifies that the pipeline can be configured with different judge providers
and that the factory functions wire up the correct provider instances.
"""

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
from consistency_checker.check.multi_party_judge import FixtureMultiPartyJudge
from consistency_checker.check.providers.anthropic import AnthropicProvider
from consistency_checker.check.providers.moonshot import MoonshotJudgeProvider
from consistency_checker.check.providers.openai import OpenAIProvider
from consistency_checker.config import Config
from consistency_checker.extract.schema import Assertion, Document
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.index.embedder import embed_pending
from consistency_checker.index.faiss_store import FaissStore
from consistency_checker.pipeline import (
    CostCeilingExceeded,
    check,
    default_per_call_costs,
    make_judge,
)
from tests.conftest import HashEmbedder


@pytest.mark.e2e_fixture
def test_pipeline_with_moonshot_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that pipeline factories work with moonshot provider configured.

    Verifies that:
    1. Config accepts "moonshot" as a judge_provider value
    2. make_judge() returns a MoonshotJudgeProvider instance
    3. The pipeline provider wiring doesn't crash on instantiation
    """
    # Patch MOONSHOT_API_KEY so the provider can initialize
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key-for-testing")

    # Create minimal config with moonshot provider
    config = Config(
        corpus_dir=tmp_path,
        judge_provider="moonshot",
    )

    # Verify that config accepted the moonshot provider
    assert config.judge_provider == "moonshot"

    # Verify that make_judge() returns a MoonshotJudgeProvider instance
    judge = make_judge(config)
    assert judge is not None
    # The judge is an LLMJudge that wraps the provider
    from consistency_checker.check.llm_judge import LLMJudge

    assert isinstance(judge, LLMJudge)
    # The underlying provider should be MoonshotJudgeProvider
    assert isinstance(judge._provider, MoonshotJudgeProvider)


@pytest.mark.e2e_fixture
def test_pipeline_with_anthropic_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that the anthropic provider still works with the factory."""
    # Mock API key (not needed for this test but good practice)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-for-testing")

    config = Config(
        corpus_dir=tmp_path,
        judge_provider="anthropic",
    )

    assert config.judge_provider == "anthropic"

    judge = make_judge(config)
    from consistency_checker.check.llm_judge import LLMJudge

    assert isinstance(judge, LLMJudge)
    assert isinstance(judge._provider, AnthropicProvider)


@pytest.mark.e2e_fixture
def test_pipeline_with_openai_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that the openai provider still works with the factory."""
    # Mock API key for OpenAI provider initialization
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-testing")

    config = Config(
        corpus_dir=tmp_path,
        judge_provider="openai",
    )

    assert config.judge_provider == "openai"

    judge = make_judge(config)
    from consistency_checker.check.llm_judge import LLMJudge

    assert isinstance(judge, LLMJudge)
    assert isinstance(judge._provider, OpenAIProvider)


# --- Task 2: pairwise opt-in gate (ADR-0015) --------------------------------


def _pairwise_config(tmp_path: Path) -> Config:
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


def _seed_pairwise_store(tmp_path: Path) -> tuple[AssertionStore, FaissStore, str, tuple[str, str]]:
    """Seed two definitions of MAE so the definition pass produces ≥1 pair,
    plus two contradiction-style assertions for the pairwise pass.
    Returns (store, faiss, corpus_id, (def_a_id, def_b_id)).
    """
    store = AssertionStore(tmp_path / "store.db")
    store.migrate()
    cid = store.get_or_create_corpus("test", "/test", "moonshot")
    doc_a = Document(doc_id="docA", source_path="/A.txt")
    doc_b = Document(doc_id="docB", source_path="/B.txt")
    store.add_document(doc_a, corpus_id=cid)
    store.add_document(doc_b, corpus_id=cid)
    # Two non-definition assertions to give the pairwise gate something to find.
    a1 = Assertion.build("docA", "Revenue grew 12%.")
    a2 = Assertion.build("docB", "Revenue declined 5%.")
    d1 = Assertion.build(
        "docA", '"MAE" means A.', kind="definition", term="MAE", definition_text="A"
    )
    d2 = Assertion.build(
        "docB", '"MAE" means B.', kind="definition", term="MAE", definition_text="B"
    )
    store.add_assertions([a1, a2, d1, d2])

    embedder = HashEmbedder(dim=64)
    (tmp_path / "store").mkdir(parents=True, exist_ok=True)
    faiss = FaissStore.open_or_create(
        index_path=tmp_path / "store" / "faiss.idx",
        id_map_path=tmp_path / "store" / "faiss.idmap.json",
        dim=embedder.dim,
    )
    embed_pending(store, faiss, embedder)
    return store, faiss, cid, (d1.assertion_id, d2.assertion_id)


def test_check_skips_pairwise_when_disabled(tmp_path: Path) -> None:
    """pairwise_enabled=False + nli_checker=None: skip the pairwise pass
    entirely, but still run the definition pass."""
    cfg = _pairwise_config(tmp_path).model_copy(update={"pairwise_enabled": False})
    store, faiss, cid, (d_a, d_b) = _seed_pairwise_store(tmp_path)
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    key = (min(d_a, d_b), max(d_a, d_b))
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
        cfg,
        store=store,
        faiss_store=faiss,
        nli_checker=None,
        judge=FixtureJudge({}),
        audit_logger=logger,
        definition_checker=definition_checker,
        run_id=run_id,
        corpus_id=cid,
    )
    store.close()

    assert result.n_pairs_gated == 0
    assert result.n_pairs_judged == 0
    assert result.n_findings == 0
    assert result.n_cross_corpus_gate_drops == 0
    # Definition pass still runs.
    assert result.n_definition_pairs_judged >= 1
    assert result.n_definition_findings >= 1


def test_check_pairwise_enabled_requires_nli_checker(tmp_path: Path) -> None:
    """pairwise_enabled=True but nli_checker=None must raise ValueError."""
    cfg = _pairwise_config(tmp_path).model_copy(update={"pairwise_enabled": True})
    store, faiss, cid, _ = _seed_pairwise_store(tmp_path)
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    with pytest.raises(ValueError, match="pairwise_enabled=True but nli_checker is None"):
        check(
            cfg,
            store=store,
            faiss_store=faiss,
            nli_checker=None,
            judge=FixtureJudge({}),
            audit_logger=logger,
            run_id=run_id,
            corpus_id=cid,
        )
    store.close()


def test_check_deep_without_pairwise_rejected(tmp_path: Path) -> None:
    """pairwise_enabled=False with multi_party_judge supplied must raise ValueError —
    the deep pass relies on the strong-pair gate output."""
    cfg = _pairwise_config(tmp_path).model_copy(update={"pairwise_enabled": False})
    store, faiss, cid, _ = _seed_pairwise_store(tmp_path)
    logger = AuditLogger(store)
    run_id = logger.begin_run()
    with pytest.raises(ValueError, match="--deep requires --pairwise"):
        check(
            cfg,
            store=store,
            faiss_store=faiss,
            nli_checker=None,
            judge=FixtureJudge({}),
            audit_logger=logger,
            multi_party_judge=FixtureMultiPartyJudge({}),
            run_id=run_id,
            corpus_id=cid,
        )
    store.close()


# --- Task 2: provider-aware default per-call costs --------------------------


def test_default_per_call_costs_anthropic() -> None:
    assert default_per_call_costs("anthropic") == (0.003, 0.010)


def test_default_per_call_costs_openai() -> None:
    assert default_per_call_costs("openai") == (0.003, 0.010)


def test_default_per_call_costs_moonshot() -> None:
    assert default_per_call_costs("moonshot") == (0.0001, 0.001)


def test_default_per_call_costs_fixture() -> None:
    assert default_per_call_costs("fixture") == (0.0, 0.0)


# --- Task 3: pre-flight cost ceiling (ADR-0016) ----------------------------


def _seed_def_pair_store(
    tmp_path: Path,
) -> tuple[AssertionStore, FaissStore, str, tuple[str, str]]:
    """Seed two MAE definitions so the definition pass yields ≥1 pair.

    Pairwise is disabled in these tests so we don't need NLI-friendly
    assertions; the definition pair alone drives the cost projection.
    """
    store = AssertionStore(tmp_path / "store.db")
    store.migrate()
    cid = store.get_or_create_corpus("test", "/test", "anthropic")
    doc_a = Document(doc_id="docA", source_path="/A.txt")
    doc_b = Document(doc_id="docB", source_path="/B.txt")
    store.add_document(doc_a, corpus_id=cid)
    store.add_document(doc_b, corpus_id=cid)
    d1 = Assertion.build(
        "docA", '"MAE" means A.', kind="definition", term="MAE", definition_text="A"
    )
    d2 = Assertion.build(
        "docB", '"MAE" means B.', kind="definition", term="MAE", definition_text="B"
    )
    store.add_assertions([d1, d2])

    embedder = HashEmbedder(dim=64)
    (tmp_path / "store").mkdir(parents=True, exist_ok=True)
    faiss = FaissStore.open_or_create(
        index_path=tmp_path / "store" / "faiss.idx",
        id_map_path=tmp_path / "store" / "faiss.idmap.json",
        dim=embedder.dim,
    )
    embed_pending(store, faiss, embedder)
    return store, faiss, cid, (d1.assertion_id, d2.assertion_id)


def _cost_ceiling_config(tmp_path: Path) -> Config:
    """Base config for cost-ceiling tests — pairwise disabled to keep the
    projection driven solely by the definition pair count."""
    return Config(
        corpus_dir=tmp_path / "corpus",
        judge_provider="fixture",
        judge_model="test",
        data_dir=tmp_path / "store",
        log_dir=tmp_path / "logs",
        embedder_model="hash",
        nli_model="fixture",
        gate_similarity_threshold=-1.0,
        pairwise_enabled=False,
    )


def _def_fixture_checker(key: tuple[str, str]) -> DefinitionChecker:
    verdict = DefinitionJudgeVerdict(
        assertion_a_id=key[0],
        assertion_b_id=key[1],
        verdict="definition_divergent",
        confidence=0.9,
        rationale="A vs B",
        evidence_spans=["A", "B"],
    )
    return DefinitionChecker(judge=FixtureDefinitionJudge({key: verdict}))


def test_check_aborts_when_estimate_exceeds_max_cost(tmp_path: Path) -> None:
    """max_cost_usd lower than the high-end projection aborts the run,
    and the audit row stays in 'pending' (never marked 'running')."""
    store, faiss, cid, (d_a, d_b) = _seed_def_pair_store(tmp_path)
    cfg = _cost_ceiling_config(tmp_path).model_copy(
        update={"max_cost_usd": 0.001, "judge_provider": "anthropic"}
    )
    logger = AuditLogger(store)
    run_id = logger.begin_run(run_status="pending")
    key = (min(d_a, d_b), max(d_a, d_b))
    def_checker = _def_fixture_checker(key)

    with pytest.raises(CostCeilingExceeded) as excinfo:
        check(
            cfg,
            store=store,
            faiss_store=faiss,
            nli_checker=None,
            judge=FixtureJudge({}),
            audit_logger=logger,
            definition_checker=def_checker,
            run_id=run_id,
            corpus_id=cid,
        )

    assert excinfo.value.ceiling == 0.001
    # Anthropic per-call high = 0.010, one definition pair → est_cost_high = 0.010.
    assert excinfo.value.estimated_high >= 0.010
    # Critical: no run-status update fired — the row is still 'pending'.
    run = logger.get_run(run_id)
    assert run is not None
    assert run.run_status == "pending"
    store.close()


def test_check_runs_when_estimate_under_max_cost(tmp_path: Path) -> None:
    """max_cost_usd above the projection lets the run complete normally."""
    store, faiss, cid, (d_a, d_b) = _seed_def_pair_store(tmp_path)
    cfg = _cost_ceiling_config(tmp_path).model_copy(
        update={"max_cost_usd": 1000.0, "judge_provider": "anthropic"}
    )
    logger = AuditLogger(store)
    run_id = logger.begin_run(run_status="pending")
    key = (min(d_a, d_b), max(d_a, d_b))
    def_checker = _def_fixture_checker(key)

    result = check(
        cfg,
        store=store,
        faiss_store=faiss,
        nli_checker=None,
        judge=FixtureJudge({}),
        audit_logger=logger,
        definition_checker=def_checker,
        run_id=run_id,
        corpus_id=cid,
    )

    assert result.n_definition_pairs_judged >= 1
    run = logger.get_run(run_id)
    assert run is not None
    assert run.run_status == "done"
    store.close()


def test_check_raises_when_max_cost_zero_and_pairs_exist(tmp_path: Path) -> None:
    """max_cost_usd=0.0 with a non-fixture provider + ≥ 1 pair aborts.

    'Spend exactly nothing' is a valid signal — the high-end projection is
    0.010 (one anthropic pair) which strictly exceeds 0.0.
    """
    store, faiss, cid, (d_a, d_b) = _seed_def_pair_store(tmp_path)
    cfg = _cost_ceiling_config(tmp_path).model_copy(
        update={"max_cost_usd": 0.0, "judge_provider": "anthropic"}
    )
    logger = AuditLogger(store)
    run_id = logger.begin_run(run_status="pending")
    key = (min(d_a, d_b), max(d_a, d_b))
    def_checker = _def_fixture_checker(key)

    with pytest.raises(CostCeilingExceeded):
        check(
            cfg,
            store=store,
            faiss_store=faiss,
            nli_checker=None,
            judge=FixtureJudge({}),
            audit_logger=logger,
            definition_checker=def_checker,
            run_id=run_id,
            corpus_id=cid,
        )
    run = logger.get_run(run_id)
    assert run is not None
    assert run.run_status == "pending"
    store.close()


def test_check_no_ceiling_when_max_cost_usd_none(tmp_path: Path) -> None:
    """max_cost_usd=None (default): no pre-flight, no exception even if the
    projection would otherwise blow past any sane budget."""
    store, faiss, cid, (d_a, d_b) = _seed_def_pair_store(tmp_path)
    # Default config (max_cost_usd unset) with the expensive anthropic provider
    # plus a definition pair: would project $0.010 — well above $0.001 — but
    # with no ceiling set, the run should still complete.
    cfg = _cost_ceiling_config(tmp_path).model_copy(update={"judge_provider": "anthropic"})
    assert cfg.max_cost_usd is None
    logger = AuditLogger(store)
    run_id = logger.begin_run(run_status="pending")
    key = (min(d_a, d_b), max(d_a, d_b))
    def_checker = _def_fixture_checker(key)

    result = check(
        cfg,
        store=store,
        faiss_store=faiss,
        nli_checker=None,
        judge=FixtureJudge({}),
        audit_logger=logger,
        definition_checker=def_checker,
        run_id=run_id,
        corpus_id=cid,
    )

    assert result.n_definition_pairs_judged >= 1
    run = logger.get_run(run_id)
    assert run is not None
    assert run.run_status == "done"
    store.close()
