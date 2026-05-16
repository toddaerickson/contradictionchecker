"""Data models for the UI redesign.

Frozen dataclasses for corpus metadata, run state, and user verdicts.
All fields use complete type hints; classes are immutable once constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True, slots=True)
class Corpus:
    """Metadata for a corpus of documents."""

    corpus_id: str
    corpus_name: str
    corpus_path: str
    judge_provider: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(
        cls,
        row: tuple[str, str, str, str, str, str],
    ) -> Corpus:
        """Construct a Corpus from a database row tuple.

        Args:
            row: Tuple of (corpus_id, corpus_name, corpus_path, judge_provider,
                 created_at, updated_at) where timestamps are ISO strings.

        Returns:
            A Corpus instance with datetime objects.
        """
        corpus_id, corpus_name, corpus_path, judge_provider, created_at_str, updated_at_str = row
        return cls(
            corpus_id=corpus_id,
            corpus_name=corpus_name,
            corpus_path=corpus_path,
            judge_provider=judge_provider,
            created_at=datetime.fromisoformat(created_at_str),
            updated_at=datetime.fromisoformat(updated_at_str),
        )


@dataclass(frozen=True, slots=True)
class Run:
    """State of a consistency check run for a corpus."""

    run_id: str
    corpus_id: str
    started_at: datetime
    completed_at: datetime | None
    status: Literal["in_progress", "completed", "failed"]
    message_log: str | None = None

    def __post_init__(self) -> None:
        """Validate status field."""
        valid_statuses = ("in_progress", "completed", "failed")
        if self.status not in valid_statuses:
            raise ValueError(
                f"Invalid status '{self.status}'. Must be one of: {', '.join(valid_statuses)}"
            )

    @classmethod
    def from_row(
        cls,
        row: tuple[str, str, str, str | None, str, str | None],
    ) -> Run:
        """Construct a Run from a database row tuple.

        Args:
            row: Tuple of (run_id, corpus_id, started_at, completed_at, status, message_log)
                 where timestamps are ISO strings or None.

        Returns:
            A Run instance with datetime objects.
        """
        run_id, corpus_id, started_at_str, completed_at_str, status, message_log = row
        return cls(
            run_id=run_id,
            corpus_id=corpus_id,
            started_at=datetime.fromisoformat(started_at_str),
            completed_at=datetime.fromisoformat(completed_at_str) if completed_at_str else None,
            status=status,  # type: ignore[arg-type]
            message_log=message_log,
        )


@dataclass(frozen=True, slots=True)
class Verdict:
    """User verdict for a finding (claim/definition contradiction)."""

    finding_id: str
    user_verdict: Literal["confirmed", "false_positive", "dismissed", "pending"] | None
    is_multi_party: bool

    def __post_init__(self) -> None:
        """Validate verdict field if not None."""
        if self.user_verdict is not None:
            valid_verdicts = ("confirmed", "false_positive", "dismissed", "pending")
            if self.user_verdict not in valid_verdicts:
                raise ValueError(
                    f"Invalid verdict '{self.user_verdict}'. "
                    f"Must be one of: {', '.join(valid_verdicts)} or None."
                )

    def is_resolved(self) -> bool:
        """Return True if the verdict is resolved (not pending or None).

        Returns:
            True if verdict is 'confirmed', 'false_positive', or 'dismissed'.
            False if verdict is 'pending' or None.
        """
        return self.user_verdict in ("confirmed", "false_positive", "dismissed")
