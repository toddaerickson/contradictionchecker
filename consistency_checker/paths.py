"""Path helpers anchored to the project root.

Mirrors the shape of ``kb_guardian/utils/paths.py`` (referenced during planning)
but drops yaml-loading responsibilities — those live in
``consistency_checker.config`` so paths and configuration stay separate concerns.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root (the parent of the ``consistency_checker`` package)."""
    return Path(__file__).resolve().parent.parent


def default_data_dir() -> Path:
    """Default location for the SQLite store and FAISS sidecar."""
    return project_root() / "data" / "store"


def default_log_dir() -> Path:
    """Default location for rotating log files."""
    return project_root() / "data" / "logs"
