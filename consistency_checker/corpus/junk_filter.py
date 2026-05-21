"""Deterministic junk filter for PDF-extracted text and assertions.

Two pure predicates plus a small audit sink. No project imports — these are
text-in / reason-out functions so they can be lifted into a standalone package
later. Each predicate returns a short reason string when the input is junk, or
``None`` when it is clean. The predicates are total: on unexpected input they
return ``None`` (fail-open), so a filter bug can never delete real content.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

_log = logging.getLogger(__name__)

__all__ = ["JunkAudit", "is_junk_assertion", "is_junk_line"]

_DOT_LEADER_RE = re.compile(r"\.{5,}")
_PAGE_NUMBER_RE = re.compile(r"^(?:page\s+)?\d{1,4}$", re.IGNORECASE)
_DASH_PAGE_RE = re.compile(r"^[-–—]\s*\d{1,4}\s*[-–—]$")  # noqa: RUF001
_CROSS_REF_START_RE = re.compile(
    r"^(?:as\s+(?:defined|set\s+forth|described|provided|used|referenced)|see)\b",
    re.IGNORECASE,
)
_REF_TARGET_RE = re.compile(
    r"\b(?:article|section|paragraph|clause|bylaws?|agreement|exhibit|schedule|"
    r"herein|hereof|hereunder|above|below)\b",
    re.IGNORECASE,
)

_MIN_ALPHA_ASSERTION = 10  # below this many alpha chars → near_empty
_MAX_ALPHA_CROSS_REF = 60  # cross-ref pointer with fewer alpha chars carries no substance
_MIN_LEN_NON_ALPHA = 8  # mostly_non_alpha only applies at/above this length
_MIN_ALPHA_RATIO = 0.15  # below this alpha fraction → mostly non-alphabetic


def _alpha_count(text: str) -> int:
    return sum(1 for c in text if c.isalpha())


def _alpha_ratio(text: str) -> float:
    return _alpha_count(text) / len(text) if text else 0.0


def is_junk_line(text: str) -> str | None:
    """Reason string if ``text`` (one extracted body element) is structural junk."""
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None  # the loader already skips empty elements
    if _PAGE_NUMBER_RE.match(stripped) or _DASH_PAGE_RE.match(stripped):
        return "page_number"
    if _DOT_LEADER_RE.search(stripped):
        return "dot_leader"
    if len(stripped) >= _MIN_LEN_NON_ALPHA and _alpha_ratio(stripped) < _MIN_ALPHA_RATIO:
        return "mostly_non_alpha"
    return None


def is_junk_assertion(text: str) -> str | None:
    """Reason string if an extracted assertion/definition is junk."""
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if _DOT_LEADER_RE.search(stripped):
        return "dot_fragment"
    if len(stripped) >= _MIN_LEN_NON_ALPHA and _alpha_ratio(stripped) < _MIN_ALPHA_RATIO:
        return "mostly_non_alpha"
    if _alpha_count(stripped) < _MIN_ALPHA_ASSERTION:
        return "near_empty"
    if (
        _CROSS_REF_START_RE.match(stripped)
        and _REF_TARGET_RE.search(stripped)
        and _alpha_count(stripped) < _MAX_ALPHA_CROSS_REF
    ):
        return "cross_reference"
    return None


class JunkAudit:
    """Records dropped items: in-memory counts always, JSONL when a path is set."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._counts: dict[str, int] = {}

    def record(self, *, stage: str, reason: str, doc_id: str | None, text: str) -> None:
        self._counts[reason] = self._counts.get(reason, 0) + 1
        if self._path is None:
            return
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "stage": stage,
            "reason": reason,
            "doc_id": doc_id,
            "text_snippet": text[:200],
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:  # audit must never abort ingest
            _log.warning("junk audit write failed (%s): %s", self._path, exc)

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)
