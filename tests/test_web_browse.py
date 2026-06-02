"""Tests for the assertion-detail route.

ADR-0017 Phase 6 deleted the legacy ``/tabs/documents``, ``/documents/{id}``,
and ``/tabs/assertions`` browse routes (their content now lives in the
single-page shell's per-corpus drawers). ``GET /assertions/{id}`` survives —
the new UI links to it — so its coverage stays here.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from consistency_checker.config import Config
from consistency_checker.extract.atomic_facts import FixtureExtractor
from consistency_checker.extract.schema import Assertion, Document
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.web.app import create_app
from tests.conftest import HashEmbedder


def _config(tmp_path: Path) -> Config:
    return Config(
        corpus_dir=tmp_path / "corpus",
        judge_provider="fixture",
        judge_model="test",
        data_dir=tmp_path / "store",
        log_dir=tmp_path / "logs",
        embedder_model="hash",
        nli_model="fixture",
    )


def _client(cfg: Config) -> TestClient:
    return TestClient(
        create_app(cfg, extractor=FixtureExtractor({}), embedder=HashEmbedder(dim=64))
    )


def _seed(tmp_path: Path, *, n_docs: int, assertions_per_doc: int) -> Config:
    cfg = _config(tmp_path)
    store = AssertionStore(cfg.db_path)
    store.migrate()
    _cid = store.get_or_create_corpus("test", "/test", "moonshot")
    for i in range(n_docs):
        doc = Document.from_content(
            f"Body of doc {i}.", source_path=f"doc_{i:03d}.md", title=f"Doc {i:03d}"
        )
        store.add_document(doc, corpus_id=_cid)
        for j in range(assertions_per_doc):
            store.add_assertion(
                Assertion.build(
                    doc.doc_id,
                    f"Assertion {j} from doc {i}: some content.",
                    chunk_id=f"chunk_{i}_{j}",
                    char_start=0,
                    char_end=10,
                )
            )
    store.close()
    return cfg


# --- GET /assertions/{id} --------------------------------------------------


def test_assertion_detail_partial_renders(tmp_path: Path) -> None:
    cfg = _seed(tmp_path, n_docs=1, assertions_per_doc=1)
    store = AssertionStore(cfg.db_path)
    assertion = next(iter(store.iter_assertions()))
    store.close()
    client = _client(cfg)
    response = client.get(f"/assertions/{assertion.assertion_id}")
    assert response.status_code == 200
    body = response.text
    assert assertion.assertion_text in body
    assert "Doc 000" in body
    assert assertion.assertion_id in body
    assert "chunk_0_0" in body
    assert "0:10" in body  # char span


def test_assertion_detail_404_for_unknown(tmp_path: Path) -> None:
    client = _client(_config(tmp_path))
    response = client.get("/assertions/no_such_assertion")
    assert response.status_code == 404
