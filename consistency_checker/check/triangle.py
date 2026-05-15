"""Triangle detector over the pairwise gate output (ADR-0006, F2).

Given the same iterator of :class:`CandidatePair` the pairwise judge consumes,
:func:`find_triangles` enumerates 3-cliques whose three edges all cleared the
gate threshold. Triangles span at least 2 distinct documents (a single-doc
triangle is irrelevant — the pair gate already filters intra-doc edges) and
are deduplicated by construction (the enumeration only emits ``u < v < w``
triples).

Output is sorted by minimum edge similarity descending; only the top
``max_per_run`` survive. The cap is enforced via a bounded min-heap *during*
enumeration, so a dense graph cannot blow memory with millions of
``Triangle`` dataclasses before the cap fires. Ties on ``min_edge_score`` are
broken by the canonical ``(a_id, b_id, c_id)`` tuple so output is
deterministic.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from consistency_checker.check.gate import CandidatePair
from consistency_checker.extract.schema import Assertion
from consistency_checker.logging_setup import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Triangle:
    """Three assertions plus the three edge similarities that joined them.

    Members are sorted by ``assertion_id`` so triangle identity matches the
    natural dedupe key used by :func:`find_triangles`. ``edge_scores`` is
    ordered ``(a-b, a-c, b-c)`` matching the same id ordering.
    """

    a: Assertion
    b: Assertion
    c: Assertion
    edge_scores: tuple[tuple[str, str, float], tuple[str, str, float], tuple[str, str, float]]

    @property
    def assertion_ids(self) -> tuple[str, str, str]:
        return (self.a.assertion_id, self.b.assertion_id, self.c.assertion_id)

    @property
    def doc_ids(self) -> tuple[str, str, str]:
        return (self.a.doc_id, self.b.doc_id, self.c.doc_id)

    @property
    def min_edge_score(self) -> float:
        return min(score for _, _, score in self.edge_scores)


def _edge_key(a_id: str, b_id: str) -> tuple[str, str]:
    return (a_id, b_id) if a_id < b_id else (b_id, a_id)


@dataclass(frozen=True, slots=True)
class _HeapEntry:
    """Min-heap entry whose ordering inverts the output sort.

    The heap top is the entry to evict first — i.e. the *worst* triangle by
    output sort key ``(-min_edge_score, assertion_ids)``. "Worst" = lower
    ``min_edge_score``, or on tie, lexicographically larger ``assertion_ids``.
    """

    min_score: float
    ids: tuple[str, str, str]
    triangle: Triangle

    def __lt__(self, other: _HeapEntry) -> bool:
        if self.min_score == other.min_score:
            # Equal scores: the lexicographically larger id-tuple is "worse"
            # (loses the output tiebreak), so it sorts lower in the heap.
            return self.ids > other.ids
        return self.min_score < other.min_score


def find_triangles(
    pairs: Iterable[CandidatePair],
    *,
    max_per_run: int = 1000,
) -> Iterator[Triangle]:
    """Enumerate triangles in the pairwise gate graph.

    Args:
        pairs: Iterable of :class:`CandidatePair` from the gate (any order;
            ``(A, B)`` and ``(B, A)`` are treated as the same edge).
        max_per_run: Cap on the number of triangles retained. Enforced *during*
            enumeration via a bounded heap, so memory stays O(``max_per_run``)
            rather than O(total triangles in the graph). The highest-confidence
            triangles by ``min_edge_score`` always survive the cap.

    Yields:
        :class:`Triangle` records, sorted by ``min_edge_score`` descending
        (ties broken by canonical assertion-id tuple).
    """
    if max_per_run < 0:
        raise ValueError("max_per_run must be >= 0")

    adjacency: dict[str, set[str]] = {}
    edge_score: dict[tuple[str, str], float] = {}
    assertions: dict[str, Assertion] = {}

    for pair in pairs:
        a_id, b_id = pair.a.assertion_id, pair.b.assertion_id
        if a_id == b_id:
            continue
        assertions.setdefault(a_id, pair.a)
        assertions.setdefault(b_id, pair.b)
        key = _edge_key(a_id, b_id)
        if key in edge_score:
            edge_score[key] = max(edge_score[key], pair.score)
        else:
            edge_score[key] = pair.score
            adjacency.setdefault(a_id, set()).add(b_id)
            adjacency.setdefault(b_id, set()).add(a_id)

    if max_per_run == 0:
        _log.info("find_triangles: %d edges → cap=0, yielding nothing", len(edge_score))
        return

    heap: list[_HeapEntry] = []
    n_enumerated = 0

    # u < v < w by construction (`higher` is sorted, `w` is later in it than
    # `v`), so no dedupe set is needed — every triple is emitted at most once.
    for u in sorted(adjacency):
        u_neighbours = adjacency[u]
        higher = sorted(v for v in u_neighbours if v > u)
        for i, v in enumerate(higher):
            v_neighbours = adjacency[v]
            for w in higher[i + 1 :]:
                if w not in v_neighbours:
                    continue

                a, b, c = assertions[u], assertions[v], assertions[w]
                if len({a.doc_id, b.doc_id, c.doc_id}) < 2:
                    continue

                triangle = Triangle(
                    a=a,
                    b=b,
                    c=c,
                    edge_scores=(
                        (u, v, edge_score[_edge_key(u, v)]),
                        (u, w, edge_score[_edge_key(u, w)]),
                        (v, w, edge_score[_edge_key(v, w)]),
                    ),
                )
                n_enumerated += 1
                entry = _HeapEntry(triangle.min_edge_score, triangle.assertion_ids, triangle)
                if len(heap) < max_per_run:
                    heapq.heappush(heap, entry)
                else:
                    heapq.heappushpop(heap, entry)

    result = sorted(
        (e.triangle for e in heap),
        key=lambda t: (-t.min_edge_score, t.assertion_ids),
    )

    _log.info(
        "find_triangles: %d edges → %d enumerated, yielding top %d (cap=%d)",
        len(edge_score),
        n_enumerated,
        len(result),
        max_per_run,
    )

    yield from result
