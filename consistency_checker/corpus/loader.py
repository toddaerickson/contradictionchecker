"""Load source documents from disk into :class:`LoadedDocument` pairs.

A loaded document keeps the persistent :class:`Document` (metadata + content
hash; suitable for the assertion store) alongside the transient full text used
by the chunker. Raw text is never persisted to the documents table — the
corpus is the source of truth on disk.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from consistency_checker.extract.schema import Document
from consistency_checker.logging_setup import get_logger

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md"})
STUB_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx"})

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """A document with its content paired for chunking."""

    document: Document
    text: str


def load_path(path: Path | str) -> LoadedDocument:
    """Load a single document by path. Title defaults to the file stem."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in STUB_EXTENSIONS:
        raise NotImplementedError(
            f"{ext} support is not implemented in the MVP — see Step 6 of the build plan."
        )
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension: {ext!r}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}; stubbed: {sorted(STUB_EXTENSIONS)}."
        )
    text = p.read_text(encoding="utf-8")
    document = Document.from_content(text, source_path=str(p), title=p.stem)
    return LoadedDocument(document=document, text=text)


def load_corpus(corpus_dir: Path | str) -> Iterator[LoadedDocument]:
    """Walk ``corpus_dir`` recursively, yielding loaded documents.

    Files with unsupported extensions are skipped with a warning. Stub
    extensions (``.pdf``, ``.docx``) emit an explicit log line so users see
    them rather than wondering why their corpus shrank.
    """
    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus path is not a directory: {root}")

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            yield load_path(path)
        elif ext in STUB_EXTENSIONS:
            _log.warning("Skipping %s — %s loader not implemented", path, ext)
        else:
            _log.debug("Skipping %s — extension %s not recognised", path, ext)
