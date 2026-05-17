"""Tests for the Documents + Assertions tabs — step G3."""

from __future__ import annotations

from pathlib import Path

import pytest
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
    for i in range(n_docs):
        doc = Document.from_content(
            f"Body of doc {i}.", source_path=f"doc_{i:03d}.md", title=f"Doc {i:03d}"
        )
        store.add_document(doc)
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


# --- /tabs/documents -------------------------------------------------------


def test_documents_tab_empty_state(tmp_path: Path) -> None:
    client = _client(_config(tmp_path))
    response = client.get("/tabs/documents")
    assert response.status_code == 200
    body = response.text
    assert "Documents" in body
    assert "No documents yet" in body


def test_documents_tab_lists_seeded_documents(tmp_path: Path) -> None:
    cfg = _seed(tmp_path, n_docs=3, assertions_per_doc=2)
    client = _client(cfg)
    response = client.get("/tabs/documents")
    assert response.status_code == 200
    body = response.text
    assert "Doc 000" in body
    assert "Doc 001" in body
    assert "Doc 002" in body
    assert "3 ingested document(s)" in body


def test_documents_tab_pagination(tmp_path: Path) -> None:
    """Page 1 shows the first 25, page 2 shows the rest."""
    cfg = _seed(tmp_path, n_docs=27, assertions_per_doc=1)
    client = _client(cfg)
    page1 = client.get("/tabs/documents?page=1").text
    page2 = client.get("/tabs/documents?page=2").text
    assert "Page 1 of 2" in page1
    assert "Page 2 of 2" in page2
    # Page 1 has the Next button; page 2 does not.
    assert "Next →" in page1
    assert "Next →" not in page2
    assert "← Prev" not in page1
    assert "← Prev" in page2


def test_documents_tab_htmx_partial_omits_chrome(tmp_path: Path) -> None:
    cfg = _seed(tmp_path, n_docs=1, assertions_per_doc=1)
    client = _client(cfg)
    response = client.get("/tabs/documents", headers={"HX-Request": "true"})
    assert response.status_code == 200
    body = response.text
    assert 'class="cc-tabs"' not in body
    assert "Documents" in body


# --- GET /documents/{id} ---------------------------------------------------


def test_document_detail_partial_renders(tmp_path: Path) -> None:
    cfg = _seed(tmp_path, n_docs=1, assertions_per_doc=3)
    store = AssertionStore(cfg.db_path)
    doc_id = next(iter(store.iter_documents())).doc_id
    store.close()
    client = _client(cfg)
    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 200
    body = response.text
    assert "Doc 000" in body
    assert doc_id in body
    # Preview shows the first few assertions.
    assert "Assertion 0 from doc 0" in body
    assert "3" in body  # n_assertions


def test_document_detail_404_for_unknown(tmp_path: Path) -> None:
    client = _client(_config(tmp_path))
    response = client.get("/documents/no_such_doc")
    assert response.status_code == 404


# --- /tabs/assertions ------------------------------------------------------


def test_assertions_tab_empty_state(tmp_path: Path) -> None:
    client = _client(_config(tmp_path))
    response = client.get("/tabs/assertions")
    assert response.status_code == 200
    body = response.text
    assert "Assertions" in body
    assert "No assertions extracted yet" in body


def test_assertions_tab_lists_with_doc_labels(tmp_path: Path) -> None:
    cfg = _seed(tmp_path, n_docs=2, assertions_per_doc=2)
    client = _client(cfg)
    response = client.get("/tabs/assertions")
    assert response.status_code == 200
    body = response.text
    assert "4 assertion(s)" in body
    assert "Doc 000" in body
    assert "Doc 001" in body


def test_assertions_tab_pagination(tmp_path: Path) -> None:
    cfg = _seed(tmp_path, n_docs=2, assertions_per_doc=20)  # 40 total
    client = _client(cfg)
    page1 = client.get("/tabs/assertions?page=1").text
    page2 = client.get("/tabs/assertions?page=2").text
    assert "Page 1 of 2" in page1
    assert "Page 2 of 2" in page2


def test_assertions_tab_truncates_long_text(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    store = AssertionStore(cfg.db_path)
    store.migrate()
    doc = Document.from_content("Body.", source_path="d.md", title="D")
    store.add_document(doc)
    long_text = "A" * 250
    store.add_assertion(Assertion.build(doc.doc_id, long_text))
    store.close()
    client = _client(cfg)
    response = client.get("/tabs/assertions")
    body = response.text
    assert "…" in body
    assert "A" * 250 not in body  # full text isn't dumped


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


# --- pagination helper edge cases ------------------------------------------


@pytest.mark.parametrize("page", [-3, 0, 1, 99])
def test_documents_tab_handles_out_of_range_page(tmp_path: Path, page: int) -> None:
    """Negative / out-of-range page numbers clamp into [1, n_pages]."""
    cfg = _seed(tmp_path, n_docs=2, assertions_per_doc=1)
    client = _client(cfg)
    response = client.get(f"/tabs/documents?page={page}")
    assert response.status_code == 200
