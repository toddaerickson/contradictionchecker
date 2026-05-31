"""Detection predicate + audit sink for the OCR-fallback path.

`needs_ocr` is a pure predicate that decides whether a fast-strategy
extraction looks empty enough to warrant a one-shot retry with
``strategy="hi_res"``. `OcrAudit` mirrors `JunkAudit`: in-memory counts plus
optional JSONL persistence, write failures swallowed so ingest never aborts
on telemetry trouble.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

_log = logging.getLogger(__name__)

__all__ = ["OcrAudit", "needs_ocr"]

_MIN_ALPHA_CHARS = 100
_MIN_PAGE_COUNT = 2
_MIN_FILE_SIZE = 100_000  # bytes


def _alpha_count(text: str) -> int:
    return sum(1 for c in text if c.isalpha())


def needs_ocr(*, text: str, page_count: int, file_size: int) -> bool:
    """True iff fast-path extraction looks empty for a non-trivial PDF.

    All three guards must trigger:
      * < 100 alpha chars in the extracted body text
      * >= 2 pages in the PDF (single-page PDFs are often legit-short)
      * >= 100 KB on disk (placeholder PDFs aren't worth a slow retry)
    """
    if page_count < _MIN_PAGE_COUNT:
        return False
    if file_size < _MIN_FILE_SIZE:
        return False
    return _alpha_count(text) < _MIN_ALPHA_CHARS


class OcrAudit:
    """Records OCR-fallback events: in-memory counts always, JSONL when path is set."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._counts: dict[str, int] = {}

    def record(
        self,
        *,
        event: str,
        doc_id: str | None,
        path: str,
        page_count: int,
    ) -> None:
        self._counts[event] = self._counts.get(event, 0) + 1
        if self._path is None:
            return
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            "doc_id": doc_id,
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
