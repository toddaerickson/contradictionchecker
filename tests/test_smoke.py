"""Smoke test — confirms the package is importable and metadata is present.

This exists primarily so ``pytest`` returns exit code 0 on a fresh scaffold
(no-tests-collected returns 5). Real per-module tests live alongside their
modules.
"""

import consistency_checker


def test_package_imports() -> None:
    assert consistency_checker.__version__ == "0.1.0"
