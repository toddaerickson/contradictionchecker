"""Tests for the ``serve`` non-loopback bind guard.

These exercise the pure ``_assert_safe_bind`` helper directly so the guard is
verified without booting uvicorn or touching the network.
"""

from __future__ import annotations

import pytest
import typer

from consistency_checker.cli.main import _assert_safe_bind


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_hosts_allowed(host: str) -> None:
    assert _assert_safe_bind(host, unsafe_no_auth=False) is None


def test_unsafe_flag_allows_non_loopback() -> None:
    assert _assert_safe_bind("0.0.0.0", unsafe_no_auth=True) is None


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.50"])
def test_non_loopback_refused_without_flag(host: str) -> None:
    with pytest.raises(typer.BadParameter):
        _assert_safe_bind(host, unsafe_no_auth=False)
