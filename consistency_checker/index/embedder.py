"""Embedding model wrapper and the SQLite→FAISS pumping helper.

Two pieces:

- :class:`Embedder` Protocol with one concrete implementation,
  :class:`SentenceTransformerEmbedder`, defaulting to mpnet (ADR 0002).
- :func:`embed_pending` reads assertions without a ``faiss_row`` from the
  :class:`AssertionStore`, embeds them in batches via any :class:`Embedder`,
  appends to the :class:`FaissStore`, and stamps ``faiss_row`` + ``embedded_at``
  back on the SQLite rows.

Test embedders (hashed-bytes, deterministic) live in ``tests/conftest.py`` so
this module never carries test scaffolding into production wheels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.index.faiss_store import FaissStore
from consistency_checker.logging_setup import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

_log = get_logger(__name__)

DEFAULT_BATCH_SIZE = 32


class Embedder(Protocol):
    """Anything that turns texts into a (n, dim) float32 array."""

    @property
    def dim(self) -> int: ...

    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]: ...


class SentenceTransformerEmbedder:
    """Wraps a :class:`sentence_transformers.SentenceTransformer` model.

    Embeddings are L2-normalised by sentence-transformers so the FAISS
    inner-product index yields cosine similarity directly.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def embed_pending(
    store: AssertionStore,
    faiss_store: FaissStore,
    embedder: Embedder,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Embed all assertions in ``store`` that do not yet have a ``faiss_row``.

    Vectors are written to ``faiss_store`` (which is persisted at the end of the
    call); the SQLite store is updated with the assigned ``faiss_row`` and
    ``embedded_at`` for each assertion. Returns the number of assertions newly
    embedded.
    """
    pending = [a for a in store.iter_assertions() if a.faiss_row is None]
    if not pending:
        return 0

    total = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        texts = [a.assertion_text for a in batch]
        vectors = embedder.embed_texts(texts)
        if vectors.shape[1] != faiss_store.dim:
            raise ValueError(
                f"Embedder dim {vectors.shape[1]} does not match FAISS dim {faiss_store.dim}"
            )
        ids = [a.assertion_id for a in batch]
        rows = faiss_store.add(ids, vectors)
        store.attach_embeddings(list(zip(ids, rows, strict=True)))
        total += len(batch)

    faiss_store.save()
    _log.info("Embedded %d assertions into FAISS at dim=%d", total, faiss_store.dim)
    return total


def rebuild_index(
    store: AssertionStore,
    faiss_store: FaissStore,
    embedder: Embedder,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Re-embed every assertion in ``store`` into a fresh FAISS store.

    Caller is responsible for opening ``faiss_store`` against an empty path
    (or for ensuring the existing index is intentionally about to be replaced).
    Useful after an embedder model change (ADR 0002).
    """
    all_assertions = list(store.iter_assertions())
    if not all_assertions:
        return 0

    total = 0
    for start in range(0, len(all_assertions), batch_size):
        batch = all_assertions[start : start + batch_size]
        texts = [a.assertion_text for a in batch]
        vectors = embedder.embed_texts(texts)
        ids = [a.assertion_id for a in batch]
        rows = faiss_store.add(ids, vectors)
        store.attach_embeddings(list(zip(ids, rows, strict=True)))
        total += len(batch)

    faiss_store.save()
    return total
