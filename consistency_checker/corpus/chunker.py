"""Sentence-window chunking with char-offset preservation.

The chunker groups consecutive sentences into chunks bounded by ``max_chars``.
Each chunk records the character span in the original document so that
``original_text[chunk.char_start:chunk.char_end] == chunk.text`` round-trips
exactly — required for audit trail integrity.

Sentence segmentation uses ``pysbd``; sentences are treated as atomic and are
never split, so a single sentence longer than ``max_chars`` yields one
oversized chunk (logged at WARNING).
"""

from __future__ import annotations

from dataclasses import dataclass

from pysbd import Segmenter

from consistency_checker.corpus.loader import LoadedDocument
from consistency_checker.extract.schema import hash_id
from consistency_checker.logging_setup import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A character-span of consecutive sentences from one document."""

    chunk_id: str
    doc_id: str
    text: str
    char_start: int
    char_end: int


def _segmenter() -> Segmenter:
    return Segmenter(language="en", char_span=True)


def chunk_document(
    loaded: LoadedDocument,
    *,
    max_chars: int = 1000,
    overlap_chars: int = 0,
) -> list[Chunk]:
    """Split a loaded document into sentence-window chunks.

    Args:
        loaded: The loaded document to chunk.
        max_chars: Target maximum size of a chunk, measured in source characters.
        overlap_chars: Reserved for future use. Non-zero raises ``NotImplementedError``;
            the MVP supports only non-overlapping chunks.
    """
    if overlap_chars != 0:
        raise NotImplementedError(
            "overlap_chars > 0 is not implemented yet; set chunk_overlap_chars: 0 in config."
        )
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if not loaded.text:
        return []

    segmenter = _segmenter()
    sentences = segmenter.segment(loaded.text)
    if not sentences:
        return []

    doc_id = loaded.document.doc_id
    chunks: list[Chunk] = []

    current_start: int | None = None
    current_end: int = 0

    def emit(start: int, end: int) -> None:
        text = loaded.text[start:end]
        chunks.append(
            Chunk(
                chunk_id=hash_id(doc_id, str(start), str(end)),
                doc_id=doc_id,
                text=text,
                char_start=start,
                char_end=end,
            )
        )

    for sent in sentences:
        sent_start = int(sent.start)
        sent_end = int(sent.end)
        sent_len = sent_end - sent_start

        if current_start is None:
            current_start = sent_start
            current_end = sent_end
            if sent_len > max_chars:
                _log.warning(
                    "Sentence at chars %d-%d in %s exceeds max_chars=%d (len=%d); "
                    "emitting oversized chunk.",
                    sent_start,
                    sent_end,
                    doc_id,
                    max_chars,
                    sent_len,
                )
            continue

        prospective_len = sent_end - current_start
        if prospective_len > max_chars:
            emit(current_start, current_end)
            current_start = sent_start
            current_end = sent_end
        else:
            current_end = sent_end

    if current_start is not None:
        emit(current_start, current_end)

    return chunks
