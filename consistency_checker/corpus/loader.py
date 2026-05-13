"""Load source documents from disk into :class:`LoadedDocument` pairs.

A loaded document keeps the persistent :class:`Document` (metadata + content
hash; suitable for the assertion store) alongside the transient full text used
by the chunker. Raw text is never persisted to the documents table — the
corpus is the source of truth on disk.

Loaders are registered in :data:`LOADERS` keyed by file extension. v0.2's
:class:`PlaintextLoader` handles ``.txt`` / ``.md``; the ``unstructured``-backed
loader for ``.pdf`` / ``.docx`` lands in Step D2. Unknown extensions are
silently skipped during corpus walks (with a DEBUG log) and raise on direct
:func:`load_path` calls.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from consistency_checker.extract.schema import Document
from consistency_checker.logging_setup import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """A document with its content paired for chunking."""

    document: Document
    text: str


class FileLoader(Protocol):
    """A callable that turns a path into a :class:`LoadedDocument`.

    Implementations are responsible for any format-specific bookkeeping
    (e.g. element-span sidecars stored in ``documents.metadata_json``) and for
    preserving the char-span round-trip invariant
    ``text[chunk.char_start:chunk.char_end] == chunk.text`` once the loaded
    text reaches the chunker.
    """

    def __call__(self, path: Path) -> LoadedDocument: ...


def _plaintext_loader(path: Path) -> LoadedDocument:
    """Loader for ``.txt`` and ``.md``. Reads UTF-8 and stores no extra metadata."""
    text = path.read_text(encoding="utf-8")
    document = Document.from_content(text, source_path=str(path), title=path.stem)
    return LoadedDocument(document=document, text=text)


def _stub_loader(extension: str) -> FileLoader:
    """Builds a loader that raises ``NotImplementedError`` for a stubbed extension."""

    def _stub(path: Path) -> LoadedDocument:
        raise NotImplementedError(
            f"{extension} loader is registered as a stub. "
            "Replace it via consistency_checker.corpus.loader.LOADERS to enable."
        )

    return _stub


#: Registry of file extension → loader. Mutate to add or override loaders.
LOADERS: dict[str, FileLoader] = {
    ".txt": _plaintext_loader,
    ".md": _plaintext_loader,
    ".pdf": _stub_loader(".pdf"),
    ".docx": _stub_loader(".docx"),
}

#: Extensions whose registered loader raises ``NotImplementedError`` —
#: surfaced by :func:`load_corpus` as a WARNING rather than silently skipped.
STUB_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx"})


def _is_stub(loader: FileLoader) -> bool:
    return getattr(loader, "__name__", "") == "_stub"


def load_path(path: Path | str) -> LoadedDocument:
    """Load a single document by path. Dispatches via :data:`LOADERS`.

    Raises ``NotImplementedError`` for stubbed extensions, ``ValueError`` for
    extensions with no registered loader, and ``FileNotFoundError`` for missing
    paths.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Document path does not exist: {p}")
    ext = p.suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported extension: {ext!r}. "
            f"Registered: {sorted(LOADERS)}; stubbed: {sorted(STUB_EXTENSIONS)}."
        )
    return loader(p)


def load_corpus(corpus_dir: Path | str) -> Iterator[LoadedDocument]:
    """Walk ``corpus_dir`` recursively, yielding loaded documents.

    Files with unregistered extensions are skipped silently (DEBUG log). Stub
    extensions (``.pdf``, ``.docx`` until D2 lands) emit an explicit WARNING so
    users see them rather than wondering where their files went.
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
        loader = LOADERS.get(ext)
        if loader is None:
            _log.debug("Skipping %s — extension %s not registered", path, ext)
            continue
        if ext in STUB_EXTENSIONS and _is_stub(loader):
            _log.warning("Skipping %s — %s loader not yet implemented", path, ext)
            continue
        yield loader(path)


def make_metadata_json(payload: dict[str, object]) -> str:
    """Helper for loaders that need to store structured sidecars in ``Document.metadata_json``."""
    return json.dumps(payload, ensure_ascii=False)
