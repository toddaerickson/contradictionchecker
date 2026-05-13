"""Candidate-pair generation for Stage A.

Two gates implement the :class:`CandidateGate` Protocol:

- :class:`AllPairsGate` enumerates every unordered pair (n²/2). For small
  corpora or tests where exhaustive scan is acceptable.
- :class:`AnnGate` retrieves the FAISS top-k neighbours per assertion, dedupes
  ``(A, B) == (B, A)``, applies a similarity threshold, and (by default)
  excludes intra-document pairs so a document can't contradict itself.

Both gates emit :class:`CandidatePair` records ordered canonically by
``assertion_id`` so downstream stages can treat pairs as unordered.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from consistency_checker.extract.schema import Assertion
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.index.faiss_store import FaissStore
from consistency_checker.logging_setup import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CandidatePair:
    """An unordered pair of assertions plus the gate score that promoted it."""

    a: Assertion
    b: Assertion
    score: float


def _canonical(a: Assertion, b: Assertion) -> tuple[Assertion, Assertion]:
    """Order by ``assertion_id`` so ``(X, Y)`` and ``(Y, X)`` map to one tuple."""
    return (a, b) if a.assertion_id < b.assertion_id else (b, a)


class CandidateGate(Protocol):
    """Strategy for producing candidate pairs from an assertion store."""

    def candidates(self, store: AssertionStore) -> Iterator[CandidatePair]: ...


class AllPairsGate:
    """Exhaustive C(n, 2) enumeration.

    Use for tiny corpora (<≈ 500 assertions) or as a baseline in tests where
    you want to be sure no candidates were missed.
    """

    def __init__(self, *, allow_same_document: bool = False) -> None:
        self._allow_same_document = allow_same_document

    def candidates(self, store: AssertionStore) -> Iterator[CandidatePair]:
        all_assertions = list(store.iter_assertions())
        for i, first in enumerate(all_assertions):
            for second in all_assertions[i + 1 :]:
                if not self._allow_same_document and first.doc_id == second.doc_id:
                    continue
                a, b = _canonical(first, second)
                yield CandidatePair(a=a, b=b, score=1.0)


class AnnGate:
    """FAISS top-k retrieval with dedup, threshold, and same-document filtering."""

    def __init__(
        self,
        faiss_store: FaissStore,
        *,
        top_k: int = 20,
        similarity_threshold: float = 0.7,
        allow_same_document: bool = False,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")
        self._faiss_store = faiss_store
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold
        self._allow_same_document = allow_same_document

    def candidates(self, store: AssertionStore) -> Iterator[CandidatePair]:
        seen: set[tuple[str, str]] = set()
        emitted = 0
        # Query with k+1 because each assertion's nearest neighbour is itself.
        k = self._top_k + 1
        for source in store.iter_assertions():
            if source.faiss_row is None:
                continue
            vec = self._faiss_store.get_vector(source.faiss_row)
            [results] = self._faiss_store.search(vec, k=k)
            for neighbour_id, similarity in results:
                if neighbour_id == source.assertion_id:
                    continue
                if similarity < self._similarity_threshold:
                    continue

                neighbour = store.get_assertion(neighbour_id)
                if neighbour is None:
                    continue
                if not self._allow_same_document and neighbour.doc_id == source.doc_id:
                    continue

                key = (
                    min(source.assertion_id, neighbour.assertion_id),
                    max(source.assertion_id, neighbour.assertion_id),
                )
                if key in seen:
                    continue
                seen.add(key)

                a, b = _canonical(source, neighbour)
                emitted += 1
                yield CandidatePair(a=a, b=b, score=float(similarity))

        _log.info(
            "AnnGate emitted %d candidate pairs (top_k=%d, threshold=%.2f)",
            emitted,
            self._top_k,
            self._similarity_threshold,
        )
