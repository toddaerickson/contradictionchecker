from consistency_checker.corpus.chunker import chunk_document
from consistency_checker.corpus.loader import LoadedDocument
from consistency_checker.extract.schema import Document

TEXT = " ".join(f"Sentence number {i} ends here." for i in range(40))


def _doc(text: str = TEXT) -> LoadedDocument:
    return LoadedDocument(
        document=Document.from_content(text, source_path="t.txt"),
        text=text,
    )


def test_zero_overlap_char_span_invariant():
    doc = _doc()
    for c in chunk_document(doc, max_chars=200, overlap_chars=0):
        assert doc.text[c.char_start : c.char_end] == c.text


def test_overlap_produces_more_chunks():
    doc = _doc()
    n0 = len(chunk_document(doc, max_chars=200, overlap_chars=0))
    n_overlap = len(chunk_document(doc, max_chars=200, overlap_chars=60))
    assert n_overlap >= n0


def test_overlap_char_span_invariant():
    doc = _doc()
    for c in chunk_document(doc, max_chars=200, overlap_chars=60):
        assert doc.text[c.char_start : c.char_end] == c.text


def test_overlap_chunk_ids_unique():
    doc = _doc()
    ids = [c.chunk_id for c in chunk_document(doc, max_chars=200, overlap_chars=60)]
    assert len(ids) == len(set(ids))


def test_overlap_larger_than_chunk_clamps_gracefully():
    # overlap_chars larger than max_chars shouldn't loop forever
    doc = _doc()
    chunks = chunk_document(doc, max_chars=200, overlap_chars=300)
    assert len(chunks) > 0
