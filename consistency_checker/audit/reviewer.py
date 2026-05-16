"""Reviewer-verdict types and the canonical pair_key builder.

The pair_key formula here MUST match the SQL CASE WHEN expression used by
render-time joins (see migration 0009 header for the formula).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

ReviewerVerdictLabel = Literal["confirmed", "false_positive", "dismissed"]
DetectorType = Literal["contradiction", "definition_inconsistency", "multi_party"]


def build_pair_key(*assertion_ids: str) -> str:
    """Canonical pair_key for a finding (works for pair or triangle)."""
    if len(assertion_ids) < 2:
        raise ValueError("build_pair_key needs at least 2 assertion ids")
    return ":".join(sorted(assertion_ids))


@dataclass(frozen=True, slots=True)
class ReviewerVerdict:
    """One reviewer-set verdict for a finding."""

    pair_key: str
    detector_type: DetectorType
    verdict: ReviewerVerdictLabel
    set_at: datetime | None = None
    note: str | None = None
