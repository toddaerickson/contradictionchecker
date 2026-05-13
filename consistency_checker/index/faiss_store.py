"""FAISS sidecar to the SQLite assertion store.

Holds vectors keyed by FAISS row index. A JSON ``id_map`` keeps the parallel
list of assertion ids so a row in the index can be resolved back to an
assertion id. Vectors are L2-normalized before insertion; using
:class:`faiss.IndexFlatIP` then yields cosine similarity for free.

This module is intentionally embedding-agnostic — it doesn't import
sentence-transformers and doesn't care which model produced the vectors. It
only knows about dimensions and floats.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

import faiss
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

VECTOR_DTYPE = np.float32


def _normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized: NDArray[np.float32] = (vectors / norms).astype(VECTOR_DTYPE)
    return normalized


class FaissStore:
    """Append-only vector index with an on-disk JSON id map.

    Open an existing store (``index.faiss`` + ``id_map.json``) with
    :meth:`open_or_create`; the dimension is locked at creation time so future
    embeddings of a different size fail loudly.
    """

    def __init__(
        self,
        *,
        dim: int,
        index_path: Path,
        id_map_path: Path,
        index: faiss.Index,
        id_map: list[str],
    ) -> None:
        self._dim = dim
        self._index_path = Path(index_path)
        self._id_map_path = Path(id_map_path)
        self._index = index
        self._id_map = list(id_map)

    # --- construction -------------------------------------------------------

    @classmethod
    def open_or_create(
        cls, *, index_path: Path | str, id_map_path: Path | str, dim: int
    ) -> FaissStore:
        index_path = Path(index_path)
        id_map_path = Path(id_map_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        if index_path.exists() and id_map_path.exists():
            index = faiss.read_index(str(index_path))
            id_map = json.loads(id_map_path.read_text())
            if index.d != dim:
                raise ValueError(
                    f"FAISS index dim {index.d} does not match expected dim {dim}; "
                    "delete the index or rebuild with --rebuild-index."
                )
            return cls(
                dim=dim,
                index_path=index_path,
                id_map_path=id_map_path,
                index=index,
                id_map=id_map,
            )

        index = faiss.IndexFlatIP(dim)
        return cls(
            dim=dim,
            index_path=index_path,
            id_map_path=id_map_path,
            index=index,
            id_map=[],
        )

    # --- properties ---------------------------------------------------------

    @property
    def dim(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return int(self._index.ntotal)

    # --- writes -------------------------------------------------------------

    def add(self, assertion_ids: list[str], vectors: NDArray[np.float32]) -> list[int]:
        """Append vectors and return the FAISS row index assigned to each."""
        if vectors.shape[0] != len(assertion_ids):
            raise ValueError(
                f"vectors batch size ({vectors.shape[0]}) does not match "
                f"assertion_ids length ({len(assertion_ids)})"
            )
        if vectors.shape[1] != self._dim:
            raise ValueError(f"vector dim {vectors.shape[1]} does not match index dim {self._dim}")
        normalized = _normalize(vectors.astype(VECTOR_DTYPE))
        start = self._index.ntotal
        self._index.add(normalized)
        assigned = list(range(start, start + len(assertion_ids)))
        self._id_map.extend(assertion_ids)
        return assigned

    def save(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))
        self._id_map_path.write_text(json.dumps(self._id_map))

    # --- reads --------------------------------------------------------------

    def search(self, query: NDArray[np.float32], k: int) -> list[list[tuple[str, float]]]:
        """k-NN search. ``query`` may be 1D or 2D; result is one list per row."""
        if self._index.ntotal == 0:
            return [[] for _ in range(query.shape[0] if query.ndim == 2 else 1)]
        if query.ndim == 1:
            query = query[None, :]
        normalized = _normalize(query.astype(VECTOR_DTYPE))
        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(normalized, k)
        out: list[list[tuple[str, float]]] = []
        for row_indices, row_scores in zip(indices, scores, strict=True):
            row_out: list[tuple[str, float]] = []
            for idx, score in zip(row_indices, row_scores, strict=True):
                if idx < 0 or idx >= len(self._id_map):
                    continue
                row_out.append((self._id_map[int(idx)], float(score)))
            out.append(row_out)
        return out

    def get_id(self, faiss_row: int) -> str:
        return self._id_map[faiss_row]

    # --- context manager ----------------------------------------------------

    def __enter__(self) -> FaissStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # No connection to close — leave persistence to explicit save().
        return None
