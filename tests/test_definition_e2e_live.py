"""End-to-end live test for the definition-inconsistency detector.

Runs only when ``ANTHROPIC_API_KEY`` is set (the ``live`` mark gates the CI
runner). Ingests two short loan-fixture documents that intentionally diverge
on ``"Material Adverse Effect"`` and asserts the detector surfaces at least
one ``definition_divergent`` finding for that term.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.check.definition_terms import canonicalize_term
from consistency_checker.config import Config
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.index.faiss_store import FaissStore
from consistency_checker.pipeline import (
    check as run_check,
)
from consistency_checker.pipeline import (
    ingest as run_ingest,
)
from consistency_checker.pipeline import (
    make_definition_checker,
    make_embedder,
    make_extractor,
    make_judge,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "definition_e2e"


@pytest.mark.live
@pytest.mark.e2e_fixture
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_mae_divergence_detected_by_real_judge(tmp_path: Path) -> None:
    cfg = Config(
        corpus_dir=FIXTURE_DIR,
        judge_provider="anthropic",
        judge_model="claude-sonnet-4-6",
        data_dir=tmp_path / "store",
        log_dir=tmp_path / "logs",
        embedder_model="all-MiniLM-L6-v2",
        nli_model="cross-encoder/nli-deberta-v3-base",
    )
    store = AssertionStore(cfg.db_path)
    store.migrate()
    embedder = make_embedder(cfg)
    faiss_store = FaissStore.open_or_create(
        index_path=cfg.faiss_path,
        id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
        dim=embedder.dim,
    )
    extractor = make_extractor(cfg)
    run_ingest(cfg, store=store, faiss_store=faiss_store, extractor=extractor, embedder=embedder)

    # Sanity: at least the two MAE definitions must have been extracted.
    definitions = list(store.iter_definitions())
    mae_defs = [
        d for d in definitions if canonicalize_term(d.term or "") == "material adverse effect"
    ]
    assert len(mae_defs) >= 2, f"expected ≥2 MAE definitions extracted; got {len(mae_defs)}"

    from consistency_checker.check.nli_checker import TransformerNliChecker

    audit_logger = AuditLogger(store)
    run_id = audit_logger.begin_run()
    result = run_check(
        cfg,
        store=store,
        faiss_store=faiss_store,
        nli_checker=TransformerNliChecker(model_name=cfg.nli_model),
        judge=make_judge(cfg),
        audit_logger=audit_logger,
        definition_checker=make_definition_checker(cfg),
        run_id=run_id,
    )
    store.close()

    assert result.n_definition_findings >= 1, (
        "expected the MAE divergence to surface as a definition_divergent finding"
    )

    rows = store._conn.execute(  # type: ignore[unreachable]
        "SELECT judge_verdict, judge_rationale FROM findings "
        "WHERE run_id = ? AND detector_type = 'definition_inconsistency'",
        (run_id,),
    ).fetchall()
    assert any(row["judge_verdict"] == "definition_divergent" for row in rows)
