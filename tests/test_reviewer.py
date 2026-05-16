"""Tests for the reviewer-verdict types and pair_key helper."""

from __future__ import annotations

import pytest

from consistency_checker.audit.reviewer import (
    ReviewerVerdict,
    build_pair_key,
)


def test_build_pair_key_two_assertions_sorted() -> None:
    assert build_pair_key("b", "a") == "a:b"
    assert build_pair_key("a", "b") == "a:b"


def test_build_pair_key_three_assertions_sorted() -> None:
    assert build_pair_key("c", "a", "b") == "a:b:c"


def test_build_pair_key_rejects_singleton() -> None:
    with pytest.raises(ValueError, match=r"at least 2 assertion ids"):
        build_pair_key("a")
    with pytest.raises(ValueError, match=r"at least 2 assertion ids"):
        build_pair_key()


def test_reviewer_verdict_dataclass_defaults() -> None:
    v = ReviewerVerdict(
        pair_key="a:b",
        detector_type="contradiction",
        verdict="confirmed",
    )
    assert v.set_at is None
    assert v.note is None
