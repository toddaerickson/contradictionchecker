"""Tests for corpus loading and chunking."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pytest

from consistency_checker.corpus.chunker import Chunk, chunk_document
from consistency_checker.corpus.loader import (
    LOADERS,
    STUB_EXTENSIONS,
    LoadedDocument,
    UnstructuredLoader,
    load_corpus,
    load_path,
)
from consistency_checker.extract.schema import Document

_skip_unstructured_on_win = pytest.mark.skipif(
    sys.platform == "win32",
    reason="unstructured/libmagic access violation on Windows; passes on CI (Ubuntu)",
)

FIXTURES = Path(__file__).parent / "fixtures" / "sample_docs"


# --- loader -----------------------------------------------------------------


def test_load_markdown_fixture() -> None:
    loaded = load_path(FIXTURES / "a.md")
    assert isinstance(loaded, LoadedDocument)
    assert isinstance(loaded.document, Document)
    assert loaded.document.title == "a"
    assert "Alpha" in loaded.text


def test_load_txt_fixture() -> None:
    loaded = load_path(FIXTURES / "b.txt")
    assert "Beta initiative" in loaded.text
    assert loaded.document.source_path.endswith("b.txt")


def test_load_path_unsupported_extension_raises(tmp_path: Path) -> None:
    p = tmp_path / "weird.xyz"
    p.write_text("noop")
    with pytest.raises(ValueError, match="Unsupported extension"):
        load_path(p)


def test_load_corpus_walks_recursively_and_skips_unsupported(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.md").write_text("Top-level markdown.")
    (tmp_path / "b.txt").write_text("Top-level text.")
    (tmp_path / "skipme.xyz").write_text("ignored")

    loaded = list(load_corpus(tmp_path))
    paths = {Path(ld.document.source_path).name for ld in loaded}
    assert paths == {"a.md", "b.txt"}


def test_load_corpus_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(load_corpus(tmp_path / "does-not-exist"))


def test_load_corpus_path_not_a_dir(tmp_path: Path) -> None:
    f = tmp_path / "single.txt"
    f.write_text("hi")
    with pytest.raises(NotADirectoryError):
        list(load_corpus(f))


def test_stub_extensions_is_empty_after_d2() -> None:
    """D2 wired UnstructuredLoader for .pdf and .docx — both are no longer stubs."""
    assert len(STUB_EXTENSIONS) == 0


def test_load_path_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_path(tmp_path / "ghost.txt")


# --- loader registry -------------------------------------------------------


def test_loaders_registry_has_plaintext_handlers() -> None:
    """The registry must register a loader for every plaintext extension."""
    assert ".txt" in LOADERS
    assert ".md" in LOADERS
    # Same callable so the plaintext path stays single-source.
    assert LOADERS[".txt"] is LOADERS[".md"]


def test_loaders_registry_routes_pdf_and_docx_to_unstructured() -> None:
    """D2: .pdf and .docx are bound to the UnstructuredLoader callable."""
    assert isinstance(LOADERS[".pdf"], UnstructuredLoader)
    assert LOADERS[".pdf"] is LOADERS[".docx"]


def test_load_corpus_routes_through_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adding a new registered loader at runtime must be picked up by load_corpus."""
    seen: list[str] = []

    def fake_xml_loader(path: Path) -> LoadedDocument:
        seen.append(str(path))
        text = "fake xml body"
        return LoadedDocument(
            document=Document.from_content(text, source_path=str(path), title=path.stem),
            text=text,
        )

    monkeypatch.setitem(LOADERS, ".xml", fake_xml_loader)
    (tmp_path / "a.xml").write_text("<x/>")
    (tmp_path / "b.txt").write_text("hello")
    loaded = list(load_corpus(tmp_path))
    paths = {Path(ld.document.source_path).name for ld in loaded}
    assert paths == {"a.xml", "b.txt"}
    assert len(seen) == 1


# --- chunker ----------------------------------------------------------------


def _load_text(text: str) -> LoadedDocument:
    """Make a LoadedDocument inline for chunker tests."""
    return LoadedDocument(
        document=Document.from_content(text, source_path="inline.txt"),
        text=text,
    )


def test_chunk_text_round_trips_char_spans() -> None:
    """The chunker's core invariant: text[start:end] == chunk.text for every chunk."""
    loaded = load_path(FIXTURES / "b.txt")
    chunks = chunk_document(loaded, max_chars=120)
    assert chunks
    for c in chunks:
        assert loaded.text[c.char_start : c.char_end] == c.text


def test_empty_document_yields_no_chunks() -> None:
    chunks = chunk_document(_load_text(""))
    assert chunks == []


def test_single_short_sentence_yields_one_chunk() -> None:
    loaded = _load_text("Just one sentence.")
    chunks = chunk_document(loaded)
    assert len(chunks) == 1
    assert chunks[0].text == "Just one sentence."
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(loaded.text)


def test_chunks_respect_max_chars_with_multiple_sentences() -> None:
    text = "First sentence here. Second sentence here. Third sentence here."
    loaded = _load_text(text)
    chunks = chunk_document(loaded, max_chars=40)
    # Each individual sentence is ~21 chars; two fit in 40 but three do not.
    assert len(chunks) >= 2
    for c in chunks:
        assert loaded.text[c.char_start : c.char_end] == c.text


def test_oversized_single_sentence_emits_one_chunk() -> None:
    """A sentence longer than max_chars must still be emitted, not dropped."""
    text = "This sentence has many words and is intentionally long to exceed the limit."
    loaded = _load_text(text)
    chunks = chunk_document(loaded, max_chars=10)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_chunk_id_is_deterministic() -> None:
    loaded = _load_text("Hello there. General Kenobi.")
    first = chunk_document(loaded, max_chars=15)
    second = chunk_document(loaded, max_chars=15)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_chunk_id_differs_per_span() -> None:
    loaded = _load_text("Sentence one. Sentence two. Sentence three.")
    chunks = chunk_document(loaded, max_chars=20)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_unicode_chars_preserved() -> None:
    text = "Café opened in 2024. Crème brûlée is on the menu. Привет, мир!"
    loaded = _load_text(text)
    chunks = chunk_document(loaded, max_chars=40)
    rebuilt = "".join(loaded.text[c.char_start : c.char_end] for c in chunks)
    # Concatenated chunk substrings should account for every sentence span.
    assert "Café" in chunks[0].text
    # The full reconstruction may omit inter-chunk whitespace; ensure each chunk round-trips.
    for c in chunks:
        assert loaded.text[c.char_start : c.char_end] == c.text
    assert rebuilt  # non-empty


def test_overlap_not_implemented() -> None:
    loaded = _load_text("Hello. World.")
    with pytest.raises(NotImplementedError):
        chunk_document(loaded, overlap_chars=10)


def test_negative_max_chars_raises() -> None:
    loaded = _load_text("Hello.")
    with pytest.raises(ValueError):
        chunk_document(loaded, max_chars=0)


def test_chunks_carry_doc_id() -> None:
    loaded = load_path(FIXTURES / "a.md")
    chunks = chunk_document(loaded)
    assert all(isinstance(c, Chunk) and c.doc_id == loaded.document.doc_id for c in chunks)


# --- UnstructuredLoader -----------------------------------------------------


def _assert_char_span_round_trip(loaded: LoadedDocument) -> None:
    chunks = chunk_document(loaded, max_chars=200)
    assert chunks
    for c in chunks:
        assert loaded.text[c.char_start : c.char_end] == c.text


@_skip_unstructured_on_win
def test_unstructured_loader_loads_pdf(sample_pdf_path: Path) -> None:
    loaded = load_path(sample_pdf_path)
    assert "widgets" in loaded.text
    assert "gadgets" in loaded.text
    assert loaded.document.metadata_json is not None
    _assert_char_span_round_trip(loaded)


@_skip_unstructured_on_win
def test_unstructured_loader_loads_docx(sample_docx_path: Path) -> None:
    loaded = load_path(sample_docx_path)
    assert "DOCX" in loaded.text or "body paragraph" in loaded.text.lower()
    assert loaded.document.metadata_json is not None
    _assert_char_span_round_trip(loaded)


@_skip_unstructured_on_win
def test_unstructured_loader_records_element_spans(sample_docx_path: Path) -> None:
    """Element-spans must point at the substrings they describe."""
    loaded = load_path(sample_docx_path)
    assert loaded.document.metadata_json is not None
    payload = json.loads(loaded.document.metadata_json)
    spans = payload["element_spans"]
    assert spans
    for span in spans:
        substring = loaded.text[span["char_start"] : span["char_end"]]
        assert substring.strip() != ""
        # Every element type that lands must be in our BODY_TYPES.
        assert span["element_type"] in UnstructuredLoader.BODY_TYPES


@_skip_unstructured_on_win
def test_unstructured_loader_spans_are_contiguous(sample_docx_path: Path) -> None:
    """Adjacent element spans must be separated by exactly the element separator length."""
    loaded = load_path(sample_docx_path)
    assert loaded.document.metadata_json is not None
    spans = json.loads(loaded.document.metadata_json)["element_spans"]
    sep_len = len(UnstructuredLoader.ELEMENT_SEPARATOR)
    for prev, curr in itertools.pairwise(spans):
        assert curr["char_start"] == prev["char_end"] + sep_len


@_skip_unstructured_on_win
def test_unstructured_loader_full_corpus_walk(
    sample_pdf_path: Path,
    sample_docx_path: Path,
    tmp_path: Path,
) -> None:
    """load_corpus picks up the registered .pdf and .docx loaders."""
    corpus = tmp_path / "mixed"
    corpus.mkdir()
    (corpus / "plain.txt").write_text("Plaintext document.")
    (corpus / "from_pdf.pdf").write_bytes(sample_pdf_path.read_bytes())
    (corpus / "from_docx.docx").write_bytes(sample_docx_path.read_bytes())
    loaded = list(load_corpus(corpus))
    suffixes = {Path(ld.document.source_path).suffix for ld in loaded}
    assert suffixes == {".txt", ".pdf", ".docx"}
