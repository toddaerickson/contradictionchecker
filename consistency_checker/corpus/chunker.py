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
        overlap_chars: When > 0, successive chunks overlap by approximately this many
            characters, snapped to sentence boundaries (no partial sentences). Clamped
            to ``max_chars - 1`` to guarantee forward progress.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be non-negative")
    if not loaded.text:
        return []

    if overlap_chars >= max_chars:
        overlap_chars = max_chars - 1

    segmenter = _segmenter()
    sentences = segmenter.segment(loaded.text)
    if not sentences:
        return []

    doc_id = loaded.document.doc_id
    chunks: list[Chunk] = []

    # window_sents tracks sentences currently buffered as (char_start, char_end) tuples.
    window_sents: list[tuple[int, int]] = []

    def emit_window() -> None:
        if not window_sents:
            return
        start = window_sents[0][0]
        end = window_sents[-1][1]
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

    def rewind_for_overlap() -> None:
        """Drop leading sentences from the window once it has been emitted.

        With ``overlap_chars == 0`` the window is fully cleared. Otherwise we keep
        the tail-most sentences whose ``char_start`` lies at or after the overlap
        boundary, snapping to sentence edges (no partial sentences).
        """
        nonlocal window_sents
        if not window_sents:
            return
        if overlap_chars <= 0:
            window_sents = []
            return
        last_end = window_sents[-1][1]
        boundary = last_end - overlap_chars
        retained = [(s, e) for s, e in window_sents if s >= boundary]
        # Forward-progress guard: if we somehow retained the whole window
        # (e.g. a single huge sentence longer than the overlap), drop everything
        # so the next sentence opens a fresh chunk.
        if len(retained) == len(window_sents):
            retained = []
        window_sents = retained

    for sent in sentences:
        sent_start = int(sent.start)
        sent_end = int(sent.end)
        sent_len = sent_end - sent_start

        if not window_sents:
            window_sents.append((sent_start, sent_end))
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

        prospective_len = sent_end - window_sents[0][0]
        if prospective_len > max_chars:
            emit_window()
            rewind_for_overlap()
            window_sents.append((sent_start, sent_end))
        else:
            window_sents.append((sent_start, sent_end))

    if window_sents:
        emit_window()

    return chunks
