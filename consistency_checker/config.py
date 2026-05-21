"""Configuration model for the consistency checker.

The config is plain Pydantic v2 — loaded from a YAML file, with optional
environment-variable overrides via the ``CC_`` prefix (e.g.
``CC_NLI_CONTRADICTION_THRESHOLD=0.6``). YAML is the source of truth for the
shape; env vars are reserved for per-run overrides (CI, ad-hoc tuning).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from consistency_checker.paths import default_data_dir, default_log_dir

JudgeProvider = Literal["anthropic", "openai", "moonshot", "fixture"]


def load_local_env(
    path: str | Path = ".env", *, environ: dict[str, str] | None = None
) -> list[str]:
    """Populate the environment from a local ``.env`` file (``KEY=VALUE`` lines).

    Secrets (e.g. ``MOONSHOT_API_KEY``) live in the gitignored ``.env`` so they
    never need to be typed into a shell or pasted anywhere. A key already
    present in the environment is never overwritten, so a real shell export
    still wins over the file. Returns the key *names* loaded (never values).
    """
    target = os.environ if environ is None else environ
    env_path = Path(path)
    if not env_path.is_file():
        return []
    loaded: list[str] = []
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key or key in target:
            continue
        target[key] = value.strip().strip('"').strip("'")
        loaded.append(key)
    return loaded


class Config(BaseModel):
    """Top-level configuration for a checker run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_dir: Path = Field(description="Directory to ingest documents from.")
    judge_provider: JudgeProvider = Field(
        default="anthropic",
        description=(
            "Choice of judge provider. "
            '"anthropic": Claude (Anthropic), tool-use structured output. '
            '"openai": GPT-4 (OpenAI), JSON schema structured output. '
            '"moonshot": Kimi (Moonshot AI), experimental, JSON schema via OpenAI SDK. '
            '"fixture": Test fixture, deterministic verdicts.'
        ),
    )
    judge_model: str = Field(default="claude-sonnet-4-6")

    data_dir: Path = Field(default_factory=default_data_dir)
    log_dir: Path = Field(default_factory=default_log_dir)

    embedder_model: str = Field(default="sentence-transformers/all-mpnet-base-v2")
    nli_model: str = Field(default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

    nli_contradiction_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    numeric_disagreement_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    gate_top_k: int = Field(default=20, ge=1)
    gate_similarity_threshold: float = Field(default=0.7, ge=-1.0, le=1.0)
    triangle_weak_top_k: int = Field(default=50, ge=1)
    triangle_weak_threshold: float = Field(default=0.5, ge=-1.0, le=1.0)

    chunk_max_chars: int = Field(default=1000, ge=50)
    chunk_overlap_chars: int = Field(default=0, ge=0)

    enable_multi_party: bool = Field(default=False)
    max_triangles_per_run: int = Field(default=1000, ge=0)

    max_memory_mb: int | None = Field(
        default=None,
        ge=512,
        description=(
            "When set, `consistency-check check` aborts before loading the NLI "
            "model if MemAvailable is below this threshold. Leave unset to "
            "skip the pre-flight check entirely."
        ),
    )

    @field_validator("corpus_dir", "data_dir", "log_dir", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        if isinstance(value, Path):
            return value
        return Path(str(value))

    @property
    def db_path(self) -> Path:
        """Canonical SQLite store location."""
        return self.data_dir / "assertions.db"

    @property
    def faiss_path(self) -> Path:
        """Derived FAISS index location."""
        return self.data_dir / "assertions.faiss"

    @classmethod
    def from_yaml(cls, path: str | Path, *, env: dict[str, str] | None = None) -> Config:
        """Load a Config from a YAML file. Env vars (``CC_<FIELD>``) override yaml values.

        Args:
            path: Path to the YAML config file.
            env: Optional dict of env-var-style overrides (defaults to ``os.environ``).
        """
        env_map = dict(os.environ if env is None else env)
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config file {path} must contain a YAML mapping at the root.")

        for field_name in cls.model_fields:
            env_key = f"CC_{field_name.upper()}"
            if env_key in env_map:
                data[field_name] = env_map[env_key]

        return cls.model_validate(data)
