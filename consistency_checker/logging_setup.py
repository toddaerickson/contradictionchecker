"""Logging configuration.

One named logger (``consistency_checker``) with two handlers: console for
short-form progress, a rotating file under ``config.log_dir`` for the full
record. Idempotent — calling :func:`configure` twice does not duplicate handlers.

Findings (the contradiction records) are written to the SQLite audit store in
Step 12; this module handles only execution logging.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "consistency_checker"
LOG_FILENAME = "execution.log"

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure(
    log_dir: Path,
    *,
    level: int = logging.INFO,
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
) -> logging.Logger:
    """Configure the package logger. Returns the root ``consistency_checker`` logger."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    target_path = log_dir / LOG_FILENAME
    formatter = logging.Formatter(_FORMAT)

    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
        for h in logger.handlers
    ):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

    has_file_handler = any(
        isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == target_path.resolve()
        for h in logger.handlers
    )
    if not has_file_handler:
        file_handler = RotatingFileHandler(
            target_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the package logger. Use ``name=__name__`` from caller modules."""
    if name is None:
        return logging.getLogger(LOGGER_NAME)
    if not name.startswith(LOGGER_NAME):
        name = f"{LOGGER_NAME}.{name}"
    return logging.getLogger(name)
