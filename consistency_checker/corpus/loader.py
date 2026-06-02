"""Load source documents from disk into :class:`LoadedDocument` pairs.

A loaded document keeps the persistent :class:`Document` (metadata + content
hash; suitable for the assertion store) alongside the transient full text used
by the chunker. Raw text is never persisted to the documents table — the
corpus is the source of truth on disk.

Loaders are registered in :data:`LOADERS` keyed by file extension. The
plaintext loader handles ``.txt`` / ``.md``. The :class:`UnstructuredLoader`
handles ``.pdf`` and ``.docx`` (and could be wired to any extension
``unstructured`` supports). Unknown extensions are silently skipped during
corpus walks (with a DEBUG log) and raise on direct :func:`load_path` calls.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from consistency_checker.corpus.junk_filter import JunkAudit, is_junk_line
from consistency_checker.corpus.ocr import OcrAudit, looks_empty, needs_ocr
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


def _pdf_page_count(path: Path) -> int:
    """Page count for a PDF; returns 0 on any error (treated as "skip OCR")."""
    if path.suffix.lower() != ".pdf":
        return 0
    try:
        from pypdf import PdfReader  # transitive dep of unstructured[pdf]

        return len(PdfReader(str(path)).pages)
    except Exception:  # pypdf raises a zoo of errors; treat all as "unknown"
        return 0


class UnstructuredLoader:
    """Loader backed by :mod:`unstructured` ``partition.auto``.

    Concatenates body-content element text in document order separated by
    ``"\\n\\n"``, builds a sidecar ``element_spans`` mapping, and stores it as
    JSON in ``Document.metadata_json``. Char-span invariant survives because
    the text the chunker sees is exactly the concatenation we build.

    ``strategy`` defaults to ``"fast"`` (rule-based, no model inference) per
    ADR-0004 so the loader stays hermetic. Switch to ``"hi_res"`` for
    layout-aware / OCR parsing — that path is slow and not in default CI.
    """

    # Element types we treat as body content. unstructured emits a different
    # mix depending on format and source (DOCX paragraphs → Text; PDF lines
    # → NarrativeText or Title; markdown lists → ListItem). Keep the set
    # permissive in v0.2; the ADR records Header/Footer/PageBreak/Footnote
    # as the explicit exclusions.
    BODY_TYPES: frozenset[str] = frozenset(
        {"NarrativeText", "Title", "Text", "ListItem", "Table", "UncategorizedText"}
    )
    ELEMENT_SEPARATOR = "\n\n"

    def __init__(
        self,
        *,
        strategy: str = "fast",
        drop_junk_lines: bool = True,
        audit: JunkAudit | None = None,
        ocr_enabled: bool = True,
        ocr_audit: OcrAudit | None = None,
    ) -> None:
        self._strategy = strategy
        self._drop_junk_lines = drop_junk_lines
        self._audit = audit
        self._ocr_enabled = ocr_enabled
        self._ocr_audit = ocr_audit

    def with_options(
        self,
        *,
        drop_junk_lines: bool,
        audit: JunkAudit | None,
        ocr_enabled: bool,
        ocr_audit: OcrAudit | None,
    ) -> UnstructuredLoader:
        """Return a copy with junk-filter and OCR settings overridden (strategy preserved)."""
        return UnstructuredLoader(
            strategy=self._strategy,
            drop_junk_lines=drop_junk_lines,
            audit=audit,
            ocr_enabled=ocr_enabled,
            ocr_audit=ocr_audit,
        )

    def __call__(self, path: Path) -> LoadedDocument:
        from unstructured.partition.auto import partition

        # First pass: configured strategy (default "fast").
        elements = partition(filename=str(path), strategy=self._strategy)
        full_text, element_spans = self._build_text_and_spans(elements, path)

        # OCR fallback: re-partition with hi_res when fast looks empty. Cheap
        # alpha-count check first so text-native PDFs don't pay for pypdf +
        # stat probes they'll never need.
        if (
            self._ocr_enabled
            and self._strategy == "fast"
            and path.suffix.lower() == ".pdf"
            and looks_empty(full_text)
        ):
            page_count = _pdf_page_count(path)
            try:
                file_size = path.stat().st_size
            except OSError:
                file_size = 0
            if needs_ocr(text=full_text, page_count=page_count, file_size=file_size):
                if self._ocr_audit is not None:
                    self._ocr_audit.record(
                        event="escalated",
                        path=str(path),
                        page_count=page_count,
                    )
                _log.warning(
                    "Fast extraction returned near-empty text on %s — "
                    "retrying with hi_res (OCR). First use downloads ~500 MB.",
                    path,
                )
                # hi_res can raise: missing system Tesseract, model-download
                # failure, OOM mid-OCR, Pillow decode error. Catch broadly so
                # one bad PDF doesn't abort the rest of the corpus walk.
                try:
                    elements = partition(filename=str(path), strategy="hi_res")
                    full_text, element_spans = self._build_text_and_spans(elements, path)
                except Exception as exc:
                    if self._ocr_audit is not None:
                        self._ocr_audit.record(
                            event="ocr_error",
                            path=str(path),
                            page_count=page_count,
                        )
                    _log.warning(
                        "hi_res OCR failed on %s (%s: %s) — continuing with "
                        "fast-pass empty text. Verify system Tesseract is "
                        "installed if this is unexpected.",
                        path,
                        type(exc).__name__,
                        exc,
                    )
                else:
                    if needs_ocr(text=full_text, page_count=page_count, file_size=file_size):
                        if self._ocr_audit is not None:
                            self._ocr_audit.record(
                                event="ocr_failed",
                                path=str(path),
                                page_count=page_count,
                            )
                        _log.warning(
                            "hi_res extraction also returned near-empty text on %s — "
                            "document will be ingested with whatever text was recovered.",
                            path,
                        )

        metadata = make_metadata_json({"element_spans": element_spans})
        document = Document.from_content(
            full_text,
            source_path=str(path),
            title=path.stem,
            metadata_json=metadata,
        )
        return LoadedDocument(document=document, text=full_text)

    def _build_text_and_spans(
        self, elements: Sequence[object], path: Path
    ) -> tuple[str, list[dict[str, object]]]:
        text_parts: list[str] = []
        element_spans: list[dict[str, object]] = []
        char_offset = 0
        for index, element in enumerate(elements):
            element_type = type(element).__name__
            if element_type not in self.BODY_TYPES:
                continue
            element_text = (getattr(element, "text", "") or "").strip()
            if not element_text:
                continue
            if self._drop_junk_lines:
                reason = is_junk_line(element_text)
                if reason is not None:
                    if self._audit is not None:
                        self._audit.record(
                            stage="text", reason=reason, doc_id=str(path), text=element_text
                        )
                    continue
            if text_parts:
                text_parts.append(self.ELEMENT_SEPARATOR)
                char_offset += len(self.ELEMENT_SEPARATOR)
            start = char_offset
            text_parts.append(element_text)
            char_offset += len(element_text)
            element_spans.append(
                {
                    "element_index": index,
                    "element_type": element_type,
                    "char_start": start,
                    "char_end": char_offset,
                }
            )
        return "".join(text_parts), element_spans


_unstructured_loader = UnstructuredLoader()


#: Registry of file extension → loader. Mutate to add or override loaders.
LOADERS: dict[str, FileLoader] = {
    ".txt": _plaintext_loader,
    ".md": _plaintext_loader,
    ".pdf": _unstructured_loader,
    ".docx": _unstructured_loader,
}


def load_path(
    path: Path | str,
    *,
    junk_filter_enabled: bool = True,
    junk_audit: JunkAudit | None = None,
    ocr_enabled: bool = True,
    ocr_audit: OcrAudit | None = None,
) -> LoadedDocument:
    """Load a single document by path. Dispatches via :data:`LOADERS`.

    Raises ``ValueError`` for extensions with no registered loader, and
    ``FileNotFoundError`` for missing paths.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Document path does not exist: {p}")
    ext = p.suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported extension: {ext!r}. Registered: {sorted(LOADERS)}.")
    if isinstance(loader, UnstructuredLoader):
        loader = loader.with_options(
            drop_junk_lines=junk_filter_enabled,
            audit=junk_audit,
            ocr_enabled=ocr_enabled,
            ocr_audit=ocr_audit,
        )
    return loader(p)


def load_corpus(
    corpus_dir: Path | str,
    *,
    junk_filter_enabled: bool = True,
    junk_audit: JunkAudit | None = None,
    ocr_enabled: bool = True,
    ocr_audit: OcrAudit | None = None,
) -> Iterator[LoadedDocument]:
    """Walk ``corpus_dir`` recursively, yielding loaded documents.

    Files with unregistered extensions are skipped silently (DEBUG log).
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
        yield load_path(
            path,
            junk_filter_enabled=junk_filter_enabled,
            junk_audit=junk_audit,
            ocr_enabled=ocr_enabled,
            ocr_audit=ocr_audit,
        )


def make_metadata_json(payload: dict[str, object]) -> str:
    """Helper for loaders that need to store structured sidecars in ``Document.metadata_json``."""
    return json.dumps(payload, ensure_ascii=False)
