"""Tests for config loading, env overrides, validation, and logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from consistency_checker.config import Config
from consistency_checker.logging_setup import LOG_FILENAME, LOGGER_NAME, configure, get_logger
from consistency_checker.paths import project_root


def write_yaml(path: Path, data: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def test_loads_example_yaml() -> None:
    """The shipped example yaml must be loadable and produce expected defaults."""
    cfg = Config.from_yaml(project_root() / "config.example.yml", env={})
    assert cfg.corpus_dir == Path("corpus")
    assert cfg.judge_provider == "anthropic"
    assert cfg.judge_model.startswith("claude-")
    assert cfg.embedder_model == "sentence-transformers/all-mpnet-base-v2"
    assert cfg.nli_contradiction_threshold == 0.5
    assert cfg.gate_top_k == 20


def test_env_overrides_yaml(tmp_path: Path) -> None:
    """``CC_<FIELD>`` env vars must override yaml values."""
    yml = write_yaml(tmp_path / "c.yml", {"corpus_dir": str(tmp_path / "corpus")})
    cfg = Config.from_yaml(yml, env={"CC_NLI_CONTRADICTION_THRESHOLD": "0.8"})
    assert cfg.nli_contradiction_threshold == 0.8


def test_missing_corpus_dir_raises(tmp_path: Path) -> None:
    """``corpus_dir`` is required — missing it must fail validation clearly."""
    yml = write_yaml(tmp_path / "c.yml", {"judge_provider": "fixture"})
    with pytest.raises(ValidationError) as excinfo:
        Config.from_yaml(yml, env={})
    assert "corpus_dir" in str(excinfo.value)


def test_unknown_field_raises(tmp_path: Path) -> None:
    """``extra="forbid"`` means typos in the yaml fail loudly rather than silently."""
    yml = write_yaml(
        tmp_path / "c.yml",
        {"corpus_dir": str(tmp_path), "judge_proder": "anthropic"},  # typo
    )
    with pytest.raises(ValidationError):
        Config.from_yaml(yml, env={})


def test_threshold_out_of_range(tmp_path: Path) -> None:
    """Thresholds outside [0, 1] must fail validation."""
    yml = write_yaml(tmp_path / "c.yml", {"corpus_dir": str(tmp_path)})
    with pytest.raises(ValidationError):
        Config.from_yaml(yml, env={"CC_NLI_CONTRADICTION_THRESHOLD": "1.5"})


def test_multi_party_defaults_off(tmp_path: Path) -> None:
    """``enable_multi_party`` defaults to off and ``max_triangles_per_run`` to 1000."""
    yml = write_yaml(tmp_path / "c.yml", {"corpus_dir": str(tmp_path)})
    cfg = Config.from_yaml(yml, env={})
    assert cfg.enable_multi_party is False
    assert cfg.max_triangles_per_run == 1000


def test_multi_party_can_be_enabled_via_yaml(tmp_path: Path) -> None:
    yml = write_yaml(
        tmp_path / "c.yml",
        {
            "corpus_dir": str(tmp_path),
            "enable_multi_party": True,
            "max_triangles_per_run": 50,
        },
    )
    cfg = Config.from_yaml(yml, env={})
    assert cfg.enable_multi_party is True
    assert cfg.max_triangles_per_run == 50


def test_max_triangles_per_run_rejects_negative(tmp_path: Path) -> None:
    yml = write_yaml(
        tmp_path / "c.yml",
        {"corpus_dir": str(tmp_path), "max_triangles_per_run": -1},
    )
    with pytest.raises(ValidationError):
        Config.from_yaml(yml, env={})


def test_derived_paths(tmp_path: Path) -> None:
    """``db_path`` and ``faiss_path`` are derived from ``data_dir``."""
    yml = write_yaml(
        tmp_path / "c.yml",
        {"corpus_dir": str(tmp_path), "data_dir": str(tmp_path / "store")},
    )
    cfg = Config.from_yaml(yml, env={})
    assert cfg.db_path == tmp_path / "store" / "assertions.db"
    assert cfg.faiss_path == tmp_path / "store" / "assertions.faiss"


def test_config_is_frozen(tmp_path: Path) -> None:
    """Config instances must be immutable to prevent surprise mutation across modules."""
    yml = write_yaml(tmp_path / "c.yml", {"corpus_dir": str(tmp_path)})
    cfg = Config.from_yaml(yml, env={})
    with pytest.raises(ValidationError):
        cfg.nli_contradiction_threshold = 0.9  # type: ignore[misc]


def test_configure_logger_writes_file(tmp_path: Path) -> None:
    """``configure`` creates the log dir and a real, working file handler."""
    logger = configure(log_dir=tmp_path / "logs")
    try:
        logger.info("hello from test_configure_logger_writes_file")
        for handler in logger.handlers:
            handler.flush()
        log_file = tmp_path / "logs" / LOG_FILENAME
        assert log_file.exists()
        contents = log_file.read_text()
        assert "hello from test_configure_logger_writes_file" in contents
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def test_configure_is_idempotent(tmp_path: Path) -> None:
    """Calling ``configure`` twice must not duplicate handlers."""
    logger = configure(log_dir=tmp_path / "logs")
    initial_count = len(logger.handlers)
    configure(log_dir=tmp_path / "logs")
    try:
        assert len(logger.handlers) == initial_count
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def test_get_logger_namespaces_under_package() -> None:
    """``get_logger("foo")`` must return a child of the package logger."""
    child = get_logger("foo")
    assert child.name == f"{LOGGER_NAME}.foo"
    assert isinstance(child, logging.Logger)
