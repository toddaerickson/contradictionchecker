"""Shared pytest fixtures and helpers.

A deterministic ``HashEmbedder`` lives here so any test that needs vector
embeddings can use it without spinning up a real sentence-transformer model
(saves ~400MB of downloads + ~1s startup). The hash-based scheme is *not*
semantically meaningful — paraphrase neighbours won't surface — but identical
strings always produce identical vectors, which is enough for round-trip and
schema tests.
"""

from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class HashEmbedder:
    """Hash-derived deterministic embeddings for hermetic tests."""

    def __init__(self, dim: int = 64) -> None:
        if dim % 4 != 0:
            raise ValueError("dim must be a multiple of 4")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            data = sha256(text.encode("utf-8")).digest()
            while len(data) < self._dim * 4:
                data = data + sha256(data).digest()
            # uint32 → float64 in [-1, 1] avoids NaN/inf bit patterns that
            # raw float32 reinterpretation would produce.
            raw = np.frombuffer(data[: self._dim * 4], dtype=np.uint32).astype(np.float64)
            arr = (raw / np.iinfo(np.uint32).max) * 2.0 - 1.0
            norm = float(np.linalg.norm(arr))
            if norm > 0.0:
                arr = arr / norm
            out[i] = arr.astype(np.float32)
        return out
