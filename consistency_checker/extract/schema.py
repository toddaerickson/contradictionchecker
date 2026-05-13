"""Dataclasses for documents and extracted assertions.

`doc_id` and `assertion_id` are short content hashes so re-ingesting the same
content is idempotent — same input always produces the same id, and SQLite
``INSERT OR IGNORE`` makes round-trips safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256

ID_LENGTH = 16


def hash_id(*parts: str) -> str:
    """Stable 16-char hex id from the SHA-256 of the joined parts."""
    payload = "".join(parts).encode("utf-8")
    return sha256(payload).hexdigest()[:ID_LENGTH]


@dataclass(frozen=True, slots=True)
class Document:
    """A source document. Identified by a content hash of its full text."""

    doc_id: str
    source_path: str
    title: str | None = None
    doc_date: str | None = None
    doc_type: str | None = None
    metadata_json: str | None = None
    ingested_at: datetime | None = None

    @classmethod
    def from_content(
        cls,
        content: str,
        source_path: str,
        *,
        title: str | None = None,
        doc_date: str | None = None,
        doc_type: str | None = None,
        metadata_json: str | None = None,
    ) -> Document:
        return cls(
            doc_id=hash_id(content),
            source_path=source_path,
            title=title,
            doc_date=doc_date,
            doc_type=doc_type,
            metadata_json=metadata_json,
        )


@dataclass(frozen=True, slots=True)
class Assertion:
    """An atomic, decontextualised claim extracted from a document."""

    assertion_id: str
    doc_id: str
    assertion_text: str
    chunk_id: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    faiss_row: int | None = None
    embedded_at: datetime | None = None
    created_at: datetime | None = None

    @classmethod
    def build(
        cls,
        doc_id: str,
        assertion_text: str,
        *,
        chunk_id: str | None = None,
        char_start: int | None = None,
        char_end: int | None = None,
    ) -> Assertion:
        return cls(
            assertion_id=hash_id(doc_id, assertion_text),
            doc_id=doc_id,
            assertion_text=assertion_text,
            chunk_id=chunk_id,
            char_start=char_start,
            char_end=char_end,
        )
