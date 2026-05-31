"""Detection predicate + audit sink for the OCR-fallback path.

`needs_ocr` is a pure predicate that decides whether a fast-strategy
extraction looks empty enough to warrant a one-shot retry with
``strategy="hi_res"``. `looks_empty` is the cheap text-only sub-check that
callers can use to skip the expensive page-count / stat probes when the
extracted text already has plenty of alpha chars. `OcrAudit` mirrors
`JunkAudit`: in-memory counts plus optional JSONL persistence, write
failures swallowed so ingest never aborts on telemetry trouble.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

_log = logging.getLogger(__name__)

__all__ = ["OcrAudit", "looks_empty", "needs_ocr"]

_MIN_ALPHA_CHARS = 100
_MIN_PAGE_COUNT = 2
_MIN_FILE_SIZE = 100_000  # bytes


def _alpha_count(text: str) -> int:
    return sum(1 for c in text if c.isalpha())


def looks_empty(text: str) -> bool:
    """True when fast-path text has too few alpha chars to be a real document body.

    Cheap text-only check. Callers can use this to short-circuit before
    paying for `pypdf` page counts and `path.stat()` — both are wasted I/O
    on text-native PDFs where the alpha count is already plenty.
    """
    return _alpha_count(text) < _MIN_ALPHA_CHARS


def needs_ocr(*, text: str, page_count: int, file_size: int) -> bool:
    """True iff fast-path extraction looks empty for a non-trivial PDF.

    All three guards must trigger:
      * < 100 alpha chars in the extracted body text (cheapest — checked first)
      * >= 2 pages in the PDF (single-page PDFs are often legit-short)
      * >= 100 KB on disk (placeholder PDFs aren't worth a slow retry)
    """
    if not looks_empty(text):
        return False
    if page_count < _MIN_PAGE_COUNT:
        return False
    return file_size >= _MIN_FILE_SIZE


class OcrAudit:
    """Records OCR-fallback events: in-memory counts always, JSONL when path is set."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._counts: dict[str, int] = {}

    def record(
        self,
        *,
        event: str,
        path: str,
        page_count: int,
    ) -> None:
        self._counts[event] = self._counts.get(event, 0) + 1
        if self._path is None:
            return
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            "path": path,
            "page_count": page_count,
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:  # audit must never abort ingest
            _log.warning("ocr audit write failed (%s): %s", self._path, exc)

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)
