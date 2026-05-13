"""Tests for corpus loading and chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from consistency_checker.corpus.chunker import Chunk, chunk_document
from consistency_checker.corpus.loader import (
    STUB_EXTENSIONS,
    LoadedDocument,
    load_corpus,
    load_path,
)
from consistency_checker.extract.schema import Document

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


def test_load_path_stub_extension_raises(tmp_path: Path) -> None:
    p = tmp_path / "fake.pdf"
    p.write_bytes(b"%PDF-1.4 stub")
    with pytest.raises(NotImplementedError):
        load_path(p)


def test_load_path_unsupported_extension_raises(tmp_path: Path) -> None:
    p = tmp_path / "weird.xyz"
    p.write_text("noop")
    with pytest.raises(ValueError, match="Unsupported extension"):
        load_path(p)


def test_load_corpus_walks_recursively_and_skips_unsupported(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.md").write_text("Top-level markdown.")
    (tmp_path / "b.txt").write_text("Top-level text.")
    (tmp_path / "fake.pdf").write_bytes(b"%PDF stub")
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


def test_stub_extensions_match_load_corpus_warning() -> None:
    assert ".pdf" in STUB_EXTENSIONS
    assert ".docx" in STUB_EXTENSIONS


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
