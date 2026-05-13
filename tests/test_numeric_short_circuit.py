"""Tests for the numeric short-circuit path in pipeline.check (E2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.check.gate import AllPairsGate
from consistency_checker.check.llm_judge import FixtureJudge, JudgeVerdict
from consistency_checker.check.nli_checker import FixtureNliChecker
from consistency_checker.config import Config
from consistency_checker.extract.schema import Assertion, Document
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.index.embedder import embed_pending
from consistency_checker.index.faiss_store import FaissStore
from consistency_checker.pipeline import (
    CONTRADICTION_VERDICTS,
    _try_numeric_short_circuit,
)
from consistency_checker.pipeline import (
    check as run_check,
)
from tests.conftest import HashEmbedder


def _write_config(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(
        f"""
corpus_dir: {tmp_path / "corpus"}
judge_provider: fixture
judge_model: test-model
data_dir: {tmp_path / "store"}
log_dir: {tmp_path / "logs"}
embedder_model: hash
nli_model: fixture
gate_top_k: 10
gate_similarity_threshold: -1.0
nli_contradiction_threshold: 0.0
""".strip()
    )
    return cfg_path


# --- _try_numeric_short_circuit (unit) --------------------------------------


def test_short_circuit_fires_on_sign_flip_pair() -> None:
    a = Assertion.build("doc_a", "Revenue grew 12% in fiscal 2025.")
    b = Assertion.build("doc_b", "Revenue declined 5% in fiscal 2025.")
    verdict = _try_numeric_short_circuit(a, b)
    assert verdict is not None
    assert verdict.verdict == "numeric_short_circuit"
    assert verdict.confidence == 1.0
    assert "Numeric short-circuit" in verdict.rationale
    # Canonical id ordering even when caller passed (a, b) the other way.
    assert verdict.assertion_a_id == min(a.assertion_id, b.assertion_id)
    assert verdict.assertion_b_id == max(a.assertion_id, b.assertion_id)


def test_short_circuit_returns_none_when_scope_differs() -> None:
    a = Assertion.build("doc_a", "Revenue grew 12% in fiscal 2025.")
    b = Assertion.build("doc_b", "Revenue declined 5% in fiscal 2024.")
    assert _try_numeric_short_circuit(a, b) is None


def test_short_circuit_returns_none_when_unit_differs() -> None:
    a = Assertion.build("doc_a", "Revenue grew 12% in fiscal 2025.")
    b = Assertion.build("doc_b", "Revenue fell 5 million dollars in fiscal 2025.")
    assert _try_numeric_short_circuit(a, b) is None


def test_short_circuit_returns_none_when_no_numbers() -> None:
    a = Assertion.build("doc_a", "Revenue performance was strong.")
    b = Assertion.build("doc_b", "Revenue performance was weak.")
    assert _try_numeric_short_circuit(a, b) is None


def test_contradiction_verdicts_set_includes_both_labels() -> None:
    assert "contradiction" in CONTRADICTION_VERDICTS
    assert "numeric_short_circuit" in CONTRADICTION_VERDICTS
    assert "not_contradiction" not in CONTRADICTION_VERDICTS
    assert "uncertain" not in CONTRADICTION_VERDICTS


# --- run_check integration --------------------------------------------------


class _TrackingFixtureJudge:
    """Wraps FixtureJudge but counts calls so we can assert the short-circuit
    bypasses the LLM judge."""

    def __init__(self, inner: FixtureJudge) -> None:
        self._inner = inner
        self.call_count = 0

    def judge(self, a: Assertion, b: Assertion) -> JudgeVerdict:
        self.call_count += 1
        return self._inner.judge(a, b)


def _seed_revenue_flip(tmp_path: Path, cfg: Config) -> tuple[Assertion, Assertion]:
    store = AssertionStore(cfg.db_path)
    store.migrate()
    doc_a = Document.from_content("Body A.", source_path="alpha.md", title="Alpha")
    doc_b = Document.from_content("Body B.", source_path="beta.md", title="Beta")
    store.add_document(doc_a)
    store.add_document(doc_b)
    a = Assertion.build(doc_a.doc_id, "Revenue grew 12% in fiscal 2025.")
    b = Assertion.build(doc_b.doc_id, "Revenue declined 5% in fiscal 2025.")
    store.add_assertions([a, b])
    embedder = HashEmbedder(dim=64)
    fs = FaissStore.open_or_create(
        index_path=cfg.faiss_path,
        id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
        dim=embedder.dim,
    )
    embed_pending(store, fs, embedder)
    store.close()
    return a, b


def test_pipeline_short_circuits_revenue_flip_pair(tmp_path: Path) -> None:
    """The Alpha-revenue flip surfaces as a finding without invoking the LLM judge."""
    cfg_path = _write_config(tmp_path)
    cfg = Config.from_yaml(cfg_path)
    _seed_revenue_flip(tmp_path, cfg)

    store = AssertionStore(cfg.db_path)
    fs = FaissStore.open_or_create(
        index_path=cfg.faiss_path,
        id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
        dim=64,
    )
    audit_logger = AuditLogger(store)

    nli = FixtureNliChecker({})  # default neutral → passes the 0.0 threshold
    judge = _TrackingFixtureJudge(FixtureJudge({}))

    result = run_check(
        cfg,
        store=store,
        faiss_store=fs,
        nli_checker=nli,
        judge=judge,  # type: ignore[arg-type]
        audit_logger=audit_logger,
        gate=AllPairsGate(),
    )

    assert result.n_findings == 1, (
        f"expected the sign-flip pair to be the only finding; got {result.n_findings}"
    )
    assert judge.call_count == 0, "LLM judge must NOT be called when the short-circuit fires"

    # The finding's verdict in the audit DB must be numeric_short_circuit.
    [finding] = list(
        audit_logger.iter_findings(run_id=result.run_id, verdict="numeric_short_circuit")
    )
    assert finding.judge_confidence == 1.0
    assert finding.judge_rationale is not None
    assert "Numeric short-circuit" in finding.judge_rationale
    store.close()


def test_pipeline_falls_through_to_judge_when_no_short_circuit(tmp_path: Path) -> None:
    """No sign-flip → judge gets called once per pair, short-circuit verdict absent."""
    cfg_path = _write_config(tmp_path)
    cfg = Config.from_yaml(cfg_path)

    store = AssertionStore(cfg.db_path)
    store.migrate()
    doc_a = Document.from_content("Body A.", source_path="a.md", title="A")
    doc_b = Document.from_content("Body B.", source_path="b.md", title="B")
    store.add_document(doc_a)
    store.add_document(doc_b)
    # Same-polarity pair — no sign-flip available.
    a = Assertion.build(doc_a.doc_id, "Revenue grew 12% in fiscal 2025.")
    b = Assertion.build(doc_b.doc_id, "Revenue grew 8% in fiscal 2025.")
    store.add_assertions([a, b])

    embedder = HashEmbedder(dim=64)
    fs = FaissStore.open_or_create(
        index_path=cfg.faiss_path,
        id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
        dim=embedder.dim,
    )
    embed_pending(store, fs, embedder)

    judge = _TrackingFixtureJudge(FixtureJudge({}))
    audit_logger = AuditLogger(store)
    result = run_check(
        cfg,
        store=store,
        faiss_store=fs,
        nli_checker=FixtureNliChecker({}),
        judge=judge,  # type: ignore[arg-type]
        audit_logger=audit_logger,
        gate=AllPairsGate(),
    )

    assert judge.call_count >= 1, "Judge must be called when no short-circuit applies"
    assert result.n_findings == 0  # FixtureJudge default is "uncertain"
    short_circuit_count = len(
        list(audit_logger.iter_findings(run_id=result.run_id, verdict="numeric_short_circuit"))
    )
    assert short_circuit_count == 0
    store.close()


def test_short_circuit_finding_appears_in_report(tmp_path: Path) -> None:
    """audit/report.py must render numeric_short_circuit findings alongside LLM contradictions."""
    from consistency_checker.audit.report import render_report

    cfg_path = _write_config(tmp_path)
    cfg = Config.from_yaml(cfg_path)
    _seed_revenue_flip(tmp_path, cfg)

    store = AssertionStore(cfg.db_path)
    fs = FaissStore.open_or_create(
        index_path=cfg.faiss_path,
        id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
        dim=64,
    )
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

    report = render_report(store, audit_logger, run_id=result.run_id)
    assert "Numeric short-circuit" in report
    assert "Contradictions found: 1" in report
    store.close()


@pytest.mark.parametrize(
    "rationale_substring,evidence_token",
    [
        ("metric=revenue", "12.0percent"),
        ("polarity mismatch", "5.0percent"),
    ],
)
def test_short_circuit_verdict_has_deterministic_rationale(
    rationale_substring: str, evidence_token: str
) -> None:
    a = Assertion.build("doc_a", "Revenue grew 12% in fiscal 2025.")
    b = Assertion.build("doc_b", "Revenue declined 5% in fiscal 2025.")
    verdict = _try_numeric_short_circuit(a, b)
    assert verdict is not None
    assert rationale_substring in verdict.rationale
    assert evidence_token in verdict.evidence_spans
