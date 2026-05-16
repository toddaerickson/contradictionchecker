"""Tests for UI redesign data models (Corpus, Run, Verdict)."""

from __future__ import annotations

from datetime import datetime

import pytest

from consistency_checker.models.ui import Corpus, Run, Verdict

# =============================================================================
# Corpus tests
# =============================================================================


def test_corpus_construction_with_all_fields() -> None:
    """A Corpus must be constructible with all required fields."""
    now = datetime.now()
    corpus = Corpus(
        corpus_id="corp_abc123",
        corpus_name="Annual Reports 2024",
        corpus_path="/data/reports",
        judge_provider="anthropic",
        created_at=now,
        updated_at=now,
    )
    assert corpus.corpus_id == "corp_abc123"
    assert corpus.corpus_name == "Annual Reports 2024"
    assert corpus.corpus_path == "/data/reports"
    assert corpus.judge_provider == "anthropic"
    assert corpus.created_at == now
    assert corpus.updated_at == now


def test_corpus_is_frozen() -> None:
    """Corpus instances must be immutable."""
    now = datetime.now()
    corpus = Corpus(
        corpus_id="corp_1",
        corpus_name="Test",
        corpus_path="/test",
        judge_provider="anthropic",
        created_at=now,
        updated_at=now,
    )
    with pytest.raises((AttributeError, TypeError)):
        corpus.corpus_name = "Modified"  # type: ignore[misc]


def test_corpus_from_row_with_iso_strings() -> None:
    """Corpus.from_row() must convert ISO timestamp strings to datetime objects."""
    row = (
        "corp_xyz",
        "Quarterly Filings",
        "/data/filings",
        "openai",
        "2026-05-16T10:30:00",
        "2026-05-16T12:00:00",
    )
    corpus = Corpus.from_row(row)
    assert corpus.corpus_id == "corp_xyz"
    assert corpus.corpus_name == "Quarterly Filings"
    assert corpus.corpus_path == "/data/filings"
    assert corpus.judge_provider == "openai"
    assert isinstance(corpus.created_at, datetime)
    assert isinstance(corpus.updated_at, datetime)
    assert corpus.created_at.year == 2026
    assert corpus.created_at.month == 5


# =============================================================================
# Run tests
# =============================================================================


def test_run_construction_with_all_fields() -> None:
    """A Run must be constructible with all fields."""
    started = datetime(2026, 5, 16, 10, 0, 0)
    completed = datetime(2026, 5, 16, 10, 5, 0)
    run = Run(
        run_id="run_123",
        corpus_id="corp_abc",
        started_at=started,
        completed_at=completed,
        status="completed",
        message_log="Processed 42 documents.",
    )
    assert run.run_id == "run_123"
    assert run.corpus_id == "corp_abc"
    assert run.started_at == started
    assert run.completed_at == completed
    assert run.status == "completed"
    assert run.message_log == "Processed 42 documents."


def test_run_accepts_none_completed_at() -> None:
    """A Run must accept None for completed_at (in-progress run)."""
    started = datetime(2026, 5, 16, 10, 0, 0)
    run = Run(
        run_id="run_456",
        corpus_id="corp_def",
        started_at=started,
        completed_at=None,
        status="in_progress",
    )
    assert run.completed_at is None
    assert run.status == "in_progress"


def test_run_accepts_none_message_log() -> None:
    """A Run must accept None for message_log."""
    started = datetime(2026, 5, 16, 10, 0, 0)
    run = Run(
        run_id="run_789",
        corpus_id="corp_ghi",
        started_at=started,
        completed_at=None,
        status="in_progress",
        message_log=None,
    )
    assert run.message_log is None


def test_run_status_validation_accepts_valid_values() -> None:
    """Run must accept valid status values: 'in_progress', 'completed', 'failed'."""
    started = datetime.now()
    for status in ("in_progress", "completed", "failed"):
        run = Run(
            run_id="run_test",
            corpus_id="corp_test",
            started_at=started,
            completed_at=None,
            status=status,
        )
        assert run.status == status


def test_run_status_validation_rejects_invalid_values() -> None:
    """Run must reject invalid status values."""
    started = datetime.now()
    with pytest.raises(ValueError, match="Invalid status"):
        Run(
            run_id="run_bad",
            corpus_id="corp_bad",
            started_at=started,
            completed_at=None,
            status="unknown",  # type: ignore[arg-type]
        )


def test_run_is_frozen() -> None:
    """Run instances must be immutable."""
    started = datetime.now()
    run = Run(
        run_id="run_1",
        corpus_id="corp_1",
        started_at=started,
        completed_at=None,
        status="in_progress",
    )
    with pytest.raises((AttributeError, TypeError)):
        run.status = "completed"  # type: ignore[misc]


def test_run_from_row_with_iso_strings() -> None:
    """Run.from_row() must convert ISO timestamp strings to datetime objects."""
    row = (
        "run_abc",
        "corp_xyz",
        "2026-05-16T10:00:00",
        "2026-05-16T10:30:00",
        "completed",
        "All checks passed.",
    )
    run = Run.from_row(row)
    assert run.run_id == "run_abc"
    assert run.corpus_id == "corp_xyz"
    assert isinstance(run.started_at, datetime)
    assert isinstance(run.completed_at, datetime)
    assert run.status == "completed"
    assert run.message_log == "All checks passed."


def test_run_from_row_with_none_completed_at() -> None:
    """Run.from_row() must handle None for completed_at."""
    row = (
        "run_def",
        "corp_abc",
        "2026-05-16T10:00:00",
        None,
        "in_progress",
        None,
    )
    run = Run.from_row(row)
    assert run.run_id == "run_def"
    assert run.completed_at is None
    assert run.status == "in_progress"
    assert run.message_log is None


# =============================================================================
# Verdict tests
# =============================================================================


def test_verdict_construction_with_all_fields() -> None:
    """A Verdict must be constructible with all fields."""
    verdict = Verdict(
        finding_id="find_123",
        user_verdict="confirmed",
        is_multi_party=False,
    )
    assert verdict.finding_id == "find_123"
    assert verdict.user_verdict == "confirmed"
    assert verdict.is_multi_party is False


def test_verdict_accepts_none_user_verdict() -> None:
    """A Verdict must accept None for user_verdict (unreviewed)."""
    verdict = Verdict(
        finding_id="find_456",
        user_verdict=None,
        is_multi_party=True,
    )
    assert verdict.user_verdict is None
    assert verdict.is_multi_party is True


def test_verdict_user_verdict_validation_accepts_valid_values() -> None:
    """Verdict must accept valid verdict values."""
    valid_verdicts = ("confirmed", "false_positive", "dismissed", "pending")
    for vdict in valid_verdicts:
        verdict = Verdict(
            finding_id="find_test",
            user_verdict=vdict,  # type: ignore[arg-type]
            is_multi_party=False,
        )
        assert verdict.user_verdict == vdict


def test_verdict_user_verdict_validation_rejects_invalid_values() -> None:
    """Verdict must reject invalid verdict values."""
    with pytest.raises(ValueError, match="Invalid verdict"):
        Verdict(
            finding_id="find_bad",
            user_verdict="unknown",  # type: ignore[arg-type]
            is_multi_party=False,
        )


def test_verdict_is_resolved_returns_true_for_confirmed() -> None:
    """is_resolved() must return True for 'confirmed'."""
    verdict = Verdict(
        finding_id="find_1",
        user_verdict="confirmed",
        is_multi_party=False,
    )
    assert verdict.is_resolved() is True


def test_verdict_is_resolved_returns_true_for_false_positive() -> None:
    """is_resolved() must return True for 'false_positive'."""
    verdict = Verdict(
        finding_id="find_2",
        user_verdict="false_positive",
        is_multi_party=False,
    )
    assert verdict.is_resolved() is True


def test_verdict_is_resolved_returns_true_for_dismissed() -> None:
    """is_resolved() must return True for 'dismissed'."""
    verdict = Verdict(
        finding_id="find_3",
        user_verdict="dismissed",
        is_multi_party=False,
    )
    assert verdict.is_resolved() is True


def test_verdict_is_resolved_returns_false_for_pending() -> None:
    """is_resolved() must return False for 'pending'."""
    verdict = Verdict(
        finding_id="find_4",
        user_verdict="pending",
        is_multi_party=False,
    )
    assert verdict.is_resolved() is False


def test_verdict_is_resolved_returns_false_for_none() -> None:
    """is_resolved() must return False for None."""
    verdict = Verdict(
        finding_id="find_5",
        user_verdict=None,
        is_multi_party=False,
    )
    assert verdict.is_resolved() is False


def test_verdict_is_frozen() -> None:
    """Verdict instances must be immutable."""
    verdict = Verdict(
        finding_id="find_1",
        user_verdict="pending",
        is_multi_party=False,
    )
    with pytest.raises((AttributeError, TypeError)):
        verdict.user_verdict = "confirmed"  # type: ignore[misc]


# =============================================================================
# Edge case: all three classes must be immutable
# =============================================================================


def test_frozen_prevents_modification_corpus() -> None:
    """Attempting to modify a frozen Corpus field must raise."""
    now = datetime.now()
    corpus = Corpus(
        corpus_id="corp_1",
        corpus_name="Test",
        corpus_path="/test",
        judge_provider="anthropic",
        created_at=now,
        updated_at=now,
    )
    with pytest.raises((AttributeError, TypeError)):
        corpus.corpus_id = "corp_2"  # type: ignore[misc]


def test_frozen_prevents_modification_run() -> None:
    """Attempting to modify a frozen Run field must raise."""
    started = datetime.now()
    run = Run(
        run_id="run_1",
        corpus_id="corp_1",
        started_at=started,
        completed_at=None,
        status="in_progress",
    )
    with pytest.raises((AttributeError, TypeError)):
        run.run_id = "run_2"  # type: ignore[misc]


def test_frozen_prevents_modification_verdict() -> None:
    """Attempting to modify a frozen Verdict field must raise."""
    verdict = Verdict(
        finding_id="find_1",
        user_verdict="pending",
        is_multi_party=False,
    )
    with pytest.raises((AttributeError, TypeError)):
        verdict.finding_id = "find_2"  # type: ignore[misc]
