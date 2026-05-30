"""Tests for the OOM-prevention guardrails added in v0.3.x.

Covers:
- ``Config.max_memory_mb`` validation.
- ``cli/main._preflight_memory`` abort/warn/no-op paths.
- ``pipeline._iter_candidates`` streams (regression for the materialised-list
  memory hotspot fixed by switching to ``Iterator[CandidatePair]``).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import typer
from pydantic import ValidationError

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.check.llm_judge import FixtureJudge
from consistency_checker.check.nli_checker import FixtureNliChecker
from consistency_checker.cli.main import _preflight_memory
from consistency_checker.config import Config
from consistency_checker.extract.schema import Assertion, Document
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.index.embedder import embed_pending
from consistency_checker.index.faiss_store import FaissStore
from consistency_checker.pipeline import _iter_candidates
from consistency_checker.pipeline import check as run_check
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


# --- Config field validation -----------------------------------------------


def test_max_memory_mb_defaults_to_none(cfg: Config) -> None:
    assert cfg.max_memory_mb is None


def test_max_memory_mb_rejects_below_floor(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        Config(
            corpus_dir=tmp_path / "corpus",
            judge_provider="fixture",
            judge_model="test",
            embedder_model="hash",
            nli_model="fixture",
            max_memory_mb=128,
        )


# --- pre-flight memory check -----------------------------------------------


def test_preflight_noop_when_threshold_unset(cfg: Config) -> None:
    """No exception, no warning — guardrail is opt-in via max_memory_mb."""
    _preflight_memory(cfg)


def test_preflight_aborts_when_available_below_threshold(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("consistency_checker.cli.main._available_memory_mb", lambda: 1000)
    tight = cfg.model_copy(update={"max_memory_mb": 4000})
    with pytest.raises(typer.BadParameter, match="below max_memory_mb"):
        _preflight_memory(tight)


def test_preflight_passes_when_available_meets_threshold(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("consistency_checker.cli.main._available_memory_mb", lambda: 8000)
    relaxed = cfg.model_copy(update={"max_memory_mb": 2000})
    _preflight_memory(relaxed)


def test_preflight_skips_when_psutil_unavailable(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Best-effort: no psutil → silently skip, do not raise."""
    monkeypatch.setattr("consistency_checker.cli.main._available_memory_mb", lambda: None)
    tight = cfg.model_copy(update={"max_memory_mb": 999_999})
    _preflight_memory(tight)


# --- streaming candidate iterator ------------------------------------------


def test_iter_candidates_returns_iterator_not_list(cfg: Config, tmp_path: Path) -> None:
    """Regression: switching to streaming caps peak memory at O(1) pairs."""
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
    store.add_assertions(
        [
            Assertion.build(doc_a.doc_id, "Revenue grew 12%."),
            Assertion.build(doc_b.doc_id, "Revenue declined 5%."),
        ]
    )
    embed_pending(store, faiss, embedder)

    result = _iter_candidates(cfg, store, faiss, gate=None)
    assert isinstance(result, Iterator)
    # Iterator is single-pass; second consumption yields nothing.
    first_pass = list(result)
    second_pass = list(result)
    assert len(first_pass) >= 1
    assert second_pass == []
    store.close()


# --- NLI release after pair loop -------------------------------------------


class _SpyNliChecker(FixtureNliChecker):
    """FixtureNliChecker that records release() calls."""

    def __init__(self) -> None:
        super().__init__({})
        self.release_count = 0

    def release(self) -> None:
        self.release_count += 1
        super().release()


def test_check_calls_nli_release_after_pair_loop(cfg: Config, tmp_path: Path) -> None:
    """Pipeline must drop NLI weights before the LLM-judge passes allocate."""
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
    store.add_assertions(
        [
            Assertion.build(doc_a.doc_id, "Revenue grew 12%."),
            Assertion.build(doc_b.doc_id, "Revenue declined 5%."),
        ]
    )
    embed_pending(store, faiss, embedder)
    audit_logger = AuditLogger(store)
    spy = _SpyNliChecker()
    run_id = audit_logger.begin_run()
    run_check(
        cfg,
        store=store,
        faiss_store=faiss,
        nli_checker=spy,
        judge=FixtureJudge({}),
        audit_logger=audit_logger,
        run_id=run_id,
        corpus_id=_cid,
    )
    assert spy.release_count == 1
    store.close()
