"""Triangle detector over the pairwise gate output (ADR-0006, F2).

Given the same iterator of :class:`CandidatePair` the pairwise judge consumes,
:func:`find_triangles` enumerates 3-cliques whose three edges all cleared the
gate threshold. Triangles span at least 2 distinct documents (a single-doc
triangle is irrelevant — the pair gate already filters intra-doc edges) and
are deduplicated by sorted ``assertion_id`` tuple.

Output is sorted by minimum edge similarity descending; the top
``max_per_run`` are yielded so dense corpora can't blow up the multi-party
judge budget. Ties on ``min_edge_score`` are broken by the canonical
``(a_id, b_id, c_id)`` tuple so output is deterministic.
"""

from __future__ import annotations

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


def find_triangles(
    pairs: Iterable[CandidatePair],
    *,
    max_per_run: int = 1000,
) -> Iterator[Triangle]:
    """Enumerate triangles in the pairwise gate graph.

    Args:
        pairs: Iterable of :class:`CandidatePair` from the gate (any order;
            ``(A, B)`` and ``(B, A)`` are treated as the same edge).
        max_per_run: Cap on the number of triangles yielded. Triangles are
            ranked by minimum edge similarity descending before truncation,
            so the highest-confidence triangles always survive the cap.

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

    triangles: list[Triangle] = []
    seen: set[tuple[str, str, str]] = set()

    for u in sorted(adjacency):
        u_neighbours = adjacency[u]
        higher = sorted(v for v in u_neighbours if v > u)
        for i, v in enumerate(higher):
            v_neighbours = adjacency[v]
            for w in higher[i + 1 :]:
                if w not in v_neighbours:
                    continue
                triple = (u, v, w)
                if triple in seen:
                    continue
                seen.add(triple)

                a, b, c = assertions[u], assertions[v], assertions[w]
                if len({a.doc_id, b.doc_id, c.doc_id}) < 2:
                    continue

                triangles.append(
                    Triangle(
                        a=a,
                        b=b,
                        c=c,
                        edge_scores=(
                            (u, v, edge_score[_edge_key(u, v)]),
                            (u, w, edge_score[_edge_key(u, w)]),
                            (v, w, edge_score[_edge_key(v, w)]),
                        ),
                    )
                )

    triangles.sort(key=lambda t: (-t.min_edge_score, t.assertion_ids))

    _log.info(
        "find_triangles: %d edges → %d triangles (yielding top %d)",
        len(edge_score),
        len(triangles),
        min(len(triangles), max_per_run),
    )

    yield from triangles[:max_per_run]
