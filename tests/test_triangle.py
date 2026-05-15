"""Tests for the graph-triangle detector (F2)."""

from __future__ import annotations

import pytest

from consistency_checker.check.gate import CandidatePair
from consistency_checker.check.triangle import Triangle, find_triangles
from consistency_checker.extract.schema import Assertion, Document


def _make_assertion(doc_id: str, text: str) -> Assertion:
    return Assertion.build(doc_id, text)


def _doc(source_path: str) -> Document:
    return Document.from_content(f"Body of {source_path}.", source_path=source_path)


def _pair(a: Assertion, b: Assertion, score: float) -> CandidatePair:
    """Order assertions by id so CandidatePair contract holds, but find_triangles
    should be order-insensitive either way."""
    if a.assertion_id < b.assertion_id:
        return CandidatePair(a=a, b=b, score=score)
    return CandidatePair(a=b, b=a, score=score)


# --- shape tests -----------------------------------------------------------


def test_simple_triangle_three_documents() -> None:
    d1, d2, d3 = _doc("d1.txt"), _doc("d2.txt"), _doc("d3.txt")
    a = _make_assertion(d1.doc_id, "All employees get four weeks vacation.")
    b = _make_assertion(d2.doc_id, "Engineers are employees.")
    c = _make_assertion(d3.doc_id, "Engineers get two weeks vacation.")
    pairs = [_pair(a, b, 0.82), _pair(b, c, 0.91), _pair(a, c, 0.74)]

    triangles = list(find_triangles(pairs))

    assert len(triangles) == 1
    t = triangles[0]
    assert set(t.assertion_ids) == {a.assertion_id, b.assertion_id, c.assertion_id}
    assert t.assertion_ids == tuple(sorted([a.assertion_id, b.assertion_id, c.assertion_id]))
    assert t.min_edge_score == pytest.approx(0.74)


def test_chain_of_three_yields_no_triangle() -> None:
    """A-B and B-C edges with no A-C edge: not a triangle."""
    d1, d2, d3 = _doc("d1.txt"), _doc("d2.txt"), _doc("d3.txt")
    a = _make_assertion(d1.doc_id, "alpha")
    b = _make_assertion(d2.doc_id, "beta")
    c = _make_assertion(d3.doc_id, "gamma")
    pairs = [_pair(a, b, 0.8), _pair(b, c, 0.8)]

    assert list(find_triangles(pairs)) == []


def test_clique_of_four_yields_four_triangles() -> None:
    docs = [_doc(f"d{i}.txt") for i in range(4)]
    assertions = [_make_assertion(d.doc_id, f"text-{i}") for i, d in enumerate(docs)]
    pairs: list[CandidatePair] = []
    for i in range(4):
        for j in range(i + 1, 4):
            pairs.append(_pair(assertions[i], assertions[j], 0.5 + 0.01 * (i + j)))

    triangles = list(find_triangles(pairs))

    assert len(triangles) == 4  # C(4, 3)
    triple_ids = {t.assertion_ids for t in triangles}
    assert len(triple_ids) == 4


def test_disconnected_components_independent() -> None:
    d1, d2, d3, d4, d5, d6 = (_doc(f"d{i}.txt") for i in range(6))
    a, b, c = (
        _make_assertion(d1.doc_id, "a"),
        _make_assertion(d2.doc_id, "b"),
        _make_assertion(d3.doc_id, "c"),
    )
    x, y, z = (
        _make_assertion(d4.doc_id, "x"),
        _make_assertion(d5.doc_id, "y"),
        _make_assertion(d6.doc_id, "z"),
    )
    pairs = [
        _pair(a, b, 0.9),
        _pair(b, c, 0.9),
        _pair(a, c, 0.9),
        _pair(x, y, 0.8),
        _pair(y, z, 0.8),
        _pair(x, z, 0.8),
    ]

    triangles = list(find_triangles(pairs))
    assert len(triangles) == 2


# --- multi-doc filter ------------------------------------------------------


def test_single_document_triangle_excluded() -> None:
    """Three assertions all from one document do not form a multi-party triangle."""
    d = _doc("d.txt")
    a = _make_assertion(d.doc_id, "alpha")
    b = _make_assertion(d.doc_id, "beta")
    c = _make_assertion(d.doc_id, "gamma")
    pairs = [_pair(a, b, 0.9), _pair(b, c, 0.9), _pair(a, c, 0.9)]

    assert list(find_triangles(pairs)) == []


def test_two_document_triangle_included() -> None:
    """Two distinct documents is enough — three is not required."""
    d1, d2 = _doc("d1.txt"), _doc("d2.txt")
    a = _make_assertion(d1.doc_id, "alpha")
    b = _make_assertion(d1.doc_id, "beta")
    c = _make_assertion(d2.doc_id, "gamma")
    pairs = [_pair(a, b, 0.9), _pair(b, c, 0.9), _pair(a, c, 0.9)]

    triangles = list(find_triangles(pairs))
    assert len(triangles) == 1


# --- determinism + dedupe --------------------------------------------------


def test_duplicate_edges_collapse_with_max_score() -> None:
    """Repeated (A, B) pairs in the input collapse to a single edge whose score
    is the max of the two — the gate already canonicalises but be defensive."""
    d1, d2, d3 = _doc("d1.txt"), _doc("d2.txt"), _doc("d3.txt")
    a = _make_assertion(d1.doc_id, "alpha")
    b = _make_assertion(d2.doc_id, "beta")
    c = _make_assertion(d3.doc_id, "gamma")
    pairs = [
        _pair(a, b, 0.6),
        _pair(a, b, 0.9),  # higher score wins
        _pair(b, c, 0.7),
        _pair(a, c, 0.5),
    ]

    triangles = list(find_triangles(pairs))
    assert len(triangles) == 1
    scores = {(low, high): s for low, high, s in triangles[0].edge_scores}
    ab_key = tuple(sorted([a.assertion_id, b.assertion_id]))
    assert scores[ab_key] == pytest.approx(0.9)


def test_output_deterministic_with_random_input_order() -> None:
    """Permuting input order yields identical triangle id-tuples and ordering."""
    docs = [_doc(f"d{i}.txt") for i in range(4)]
    assertions = [_make_assertion(d.doc_id, f"text-{i}") for i, d in enumerate(docs)]
    base: list[CandidatePair] = []
    for i in range(4):
        for j in range(i + 1, 4):
            base.append(_pair(assertions[i], assertions[j], 0.5 + 0.03 * (i + j)))

    forward = [t.assertion_ids for t in find_triangles(base)]
    reverse = [t.assertion_ids for t in find_triangles(list(reversed(base)))]
    shuffled = [
        t.assertion_ids
        for t in find_triangles([base[2], base[5], base[0], base[1], base[4], base[3]])
    ]

    assert forward == reverse == shuffled


def test_output_sorted_by_min_edge_score_descending() -> None:
    d = [_doc(f"d{i}.txt") for i in range(6)]
    a = [_make_assertion(doc.doc_id, f"text-{i}") for i, doc in enumerate(d)]
    pairs = [
        # First triangle: (a0, a1, a2) — min edge 0.6
        _pair(a[0], a[1], 0.9),
        _pair(a[1], a[2], 0.6),
        _pair(a[0], a[2], 0.8),
        # Second triangle: (a3, a4, a5) — min edge 0.85
        _pair(a[3], a[4], 0.95),
        _pair(a[4], a[5], 0.85),
        _pair(a[3], a[5], 0.9),
    ]
    triangles = list(find_triangles(pairs))
    assert len(triangles) == 2
    assert triangles[0].min_edge_score >= triangles[1].min_edge_score
    assert triangles[0].min_edge_score == pytest.approx(0.85)


# --- max_per_run ----------------------------------------------------------


def test_max_per_run_caps_output_by_min_score() -> None:
    """With max_per_run=1, only the higher-min-score triangle survives."""
    d = [_doc(f"d{i}.txt") for i in range(6)]
    a = [_make_assertion(doc.doc_id, f"text-{i}") for i, doc in enumerate(d)]
    pairs = [
        _pair(a[0], a[1], 0.9),
        _pair(a[1], a[2], 0.6),
        _pair(a[0], a[2], 0.8),
        _pair(a[3], a[4], 0.95),
        _pair(a[4], a[5], 0.85),
        _pair(a[3], a[5], 0.9),
    ]
    triangles = list(find_triangles(pairs, max_per_run=1))
    assert len(triangles) == 1
    assert triangles[0].min_edge_score == pytest.approx(0.85)
    assert {
        triangles[0].a.assertion_id,
        triangles[0].b.assertion_id,
        triangles[0].c.assertion_id,
    } == {
        a[3].assertion_id,
        a[4].assertion_id,
        a[5].assertion_id,
    }


def test_max_per_run_zero_yields_nothing() -> None:
    d1, d2, d3 = _doc("d1.txt"), _doc("d2.txt"), _doc("d3.txt")
    a = _make_assertion(d1.doc_id, "alpha")
    b = _make_assertion(d2.doc_id, "beta")
    c = _make_assertion(d3.doc_id, "gamma")
    pairs = [_pair(a, b, 0.9), _pair(b, c, 0.9), _pair(a, c, 0.9)]
    assert list(find_triangles(pairs, max_per_run=0)) == []


def test_max_per_run_keeps_top_k_in_order() -> None:
    """With max_per_run=2, the two highest-min-score triangles survive, ordered desc."""
    d = [_doc(f"d{i}.txt") for i in range(9)]
    a = [_make_assertion(doc.doc_id, f"text-{i}") for i, doc in enumerate(d)]
    # Three disjoint triangles with min-edge scores 0.85, 0.70, 0.55.
    pairs = [
        _pair(a[0], a[1], 0.95),
        _pair(a[1], a[2], 0.90),
        _pair(a[0], a[2], 0.85),
        _pair(a[3], a[4], 0.80),
        _pair(a[4], a[5], 0.75),
        _pair(a[3], a[5], 0.70),
        _pair(a[6], a[7], 0.65),
        _pair(a[7], a[8], 0.60),
        _pair(a[6], a[8], 0.55),
    ]
    triangles = list(find_triangles(pairs, max_per_run=2))
    assert len(triangles) == 2
    assert triangles[0].min_edge_score == pytest.approx(0.85)
    assert triangles[1].min_edge_score == pytest.approx(0.70)


def test_max_per_run_tie_break_is_deterministic() -> None:
    """Two triangles with identical min_edge_score: smaller assertion-id tuple wins."""
    d = [_doc(f"d{i}.txt") for i in range(6)]
    a = [_make_assertion(doc.doc_id, f"text-{i}") for i, doc in enumerate(d)]
    # Both triangles share min_edge_score=0.50; canonical ordering picks the
    # one with the lexicographically smaller assertion-id tuple.
    pairs = [
        _pair(a[0], a[1], 0.90),
        _pair(a[1], a[2], 0.50),
        _pair(a[0], a[2], 0.80),
        _pair(a[3], a[4], 0.90),
        _pair(a[4], a[5], 0.50),
        _pair(a[3], a[5], 0.80),
    ]
    cap_one = list(find_triangles(pairs, max_per_run=1))
    assert len(cap_one) == 1
    expected = min(
        (
            tuple(sorted([a[0].assertion_id, a[1].assertion_id, a[2].assertion_id])),
            tuple(sorted([a[3].assertion_id, a[4].assertion_id, a[5].assertion_id])),
        ),
    )
    assert cap_one[0].assertion_ids == expected


def test_max_per_run_rejects_negative() -> None:
    with pytest.raises(ValueError, match="max_per_run"):
        list(find_triangles([], max_per_run=-1))


# --- empty / degenerate ----------------------------------------------------


def test_empty_input_yields_no_triangles() -> None:
    assert list(find_triangles([])) == []


def test_two_isolated_edges_yield_no_triangle() -> None:
    d1, d2, d3, d4 = (_doc(f"d{i}.txt") for i in range(4))
    a = _make_assertion(d1.doc_id, "a")
    b = _make_assertion(d2.doc_id, "b")
    c = _make_assertion(d3.doc_id, "c")
    d_ = _make_assertion(d4.doc_id, "d")
    pairs = [_pair(a, b, 0.9), _pair(c, d_, 0.9)]
    assert list(find_triangles(pairs)) == []


def test_self_loop_pair_skipped() -> None:
    """A pair whose two assertions share an id (shouldn't happen, be defensive)."""
    d1, d2, d3 = _doc("d1.txt"), _doc("d2.txt"), _doc("d3.txt")
    a = _make_assertion(d1.doc_id, "alpha")
    b = _make_assertion(d2.doc_id, "beta")
    c = _make_assertion(d3.doc_id, "gamma")
    pairs = [
        CandidatePair(a=a, b=a, score=1.0),  # self-loop: ignored
        _pair(a, b, 0.9),
        _pair(b, c, 0.9),
        _pair(a, c, 0.9),
    ]
    triangles = list(find_triangles(pairs))
    assert len(triangles) == 1


# --- dataclass shape -------------------------------------------------------


def test_triangle_is_frozen_dataclass() -> None:
    d1, d2, d3 = _doc("d1.txt"), _doc("d2.txt"), _doc("d3.txt")
    a = _make_assertion(d1.doc_id, "alpha")
    b = _make_assertion(d2.doc_id, "beta")
    c = _make_assertion(d3.doc_id, "gamma")
    t = Triangle(
        a=a,
        b=b,
        c=c,
        edge_scores=(
            (a.assertion_id, b.assertion_id, 0.9),
            (a.assertion_id, c.assertion_id, 0.8),
            (b.assertion_id, c.assertion_id, 0.7),
        ),
    )
    with pytest.raises(AttributeError):
        t.a = b  # type: ignore[misc]
    assert t.min_edge_score == pytest.approx(0.7)
