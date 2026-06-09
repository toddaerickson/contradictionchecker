import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from benchmarks.definition_eval.harness import _metrics
from benchmarks.definition_eval.label import _load_candidates, _load_labels, create_app
from benchmarks.definition_eval.mine_pairs import (
    _is_cross_reference,
    build_candidates,
    resolve_corpus_id,
)
from consistency_checker.extract.schema import Assertion


def _defn(doc, term, text):
    return Assertion.build(
        doc, f'"{term}" means {text}.', kind="definition", term=term, definition_text=text
    )


def test_build_candidates_pairs_same_canonical_term():
    defs = [
        _defn("d1", "Board", "the board of directors"),
        _defn("d2", "board", "the board of directors"),  # identical text, canonical-equal term
        _defn("d3", "Board", "the supervisory board only"),  # divergent candidate
    ]
    cands = build_candidates(defs, max_pairs=100)
    assert len(cands) == 3  # 3 unordered pairs over 3 same-canonical-term defs
    cats = {c["category"] for c in cands}
    assert "identical" in cats and "review" in cats
    for c in cands:
        assert c["label"] == ""
        assert c["term"] and c["def_a"] and c["def_b"]
        assert c["doc_a"] and c["doc_b"]


def test_build_candidates_skips_singletons():
    defs = [_defn("d1", "Quorum", "a majority"), _defn("d2", "Notice", "written notice")]
    assert build_candidates(defs, max_pairs=100) == []  # no term has >= 2 defs


def test_build_candidates_caps_and_prioritizes_review():
    # one term, 3 distinct texts -> 3 pairs all "review"; plus an identical pair on another term.
    defs = [
        _defn("a", "Term", "alpha"),
        _defn("b", "Term", "beta"),
        _defn("c", "Term", "gamma"),
        _defn("d", "Other", "same"),
        _defn("e", "Other", "same"),
    ]
    cands = build_candidates(defs, max_pairs=2)
    assert len(cands) == 2
    assert all(c["category"] == "review" for c in cands)  # review pairs sorted ahead of identical


def test_build_candidates_pair_ids_unique_and_stable():
    defs = [
        _defn("a", "Term", "alpha"),
        _defn("b", "Term", "beta"),
        _defn("c", "Term", "gamma"),
    ]
    first = build_candidates(defs, max_pairs=100)
    ids = [c["pair_id"] for c in first]
    assert len(ids) == len(set(ids))  # unique
    # stable across re-runs (content hash, not run-order)
    second = build_candidates(defs, max_pairs=100)
    assert [c["pair_id"] for c in second] == ids


def test_is_cross_reference():
    assert _is_cross_reference("has the meaning assigned to such term in Section 2.11(a)")
    assert _is_cross_reference("the meaning specified in Section 2.11")
    assert _is_cross_reference("As defined in the Credit Agreement")
    assert _is_cross_reference("shall have the meaning given to it below")
    # real definitions are NOT cross-references
    assert not _is_cross_reference("the board of directors of the Corporation")
    assert not _is_cross_reference("a majority of the members then in office")
    assert not _is_cross_reference(
        "the intended commercial meaning of the parties"
    )  # 'meaning' mid-text
    # over-drop guard: "shall have the meaning" WITHOUT a locator is a real def, not a pointer
    assert not _is_cross_reference("shall have the meaning in ordinary commercial usage")
    assert not _is_cross_reference("has the meaning the parties intend in plain English")
    # broadened under-drop coverage
    assert _is_cross_reference("the meaning ascribed to it in Section 4.1")
    assert _is_cross_reference("the meanings specified in Annex A")
    assert _is_cross_reference("as set forth in Section 9.1")


def test_build_candidates_drops_cross_reference_pairs():
    defs = [
        _defn("d1", "Acceptable Discount", "has the meaning assigned in Section 2.11"),  # xref
        _defn(
            "d2", "Acceptable Discount", "the largest of the Offered Discounts specified"
        ),  # real
        _defn("d3", "Acceptable Discount", "the smallest qualifying discount offered"),  # real
    ]
    cands = build_candidates(defs, max_pairs=100)
    # pairs touching the xref def are dropped; only the (real, real) pair survives
    assert len(cands) == 1
    assert "has the meaning" not in cands[0]["def_a"]
    assert "has the meaning" not in cands[0]["def_b"]


def test_build_candidates_prioritizes_cross_document():
    # Two same-doc defs (enumerated first by combinations) + one in another doc.
    # Pairs in combinations order: (s1,s2) same-doc; (s1,x) cross-doc; (s2,x) cross-doc.
    # Old sort (review-only) would keep the same-doc pair first; new sort must promote a cross-doc pair.
    defs = [
        _defn("base", "Lender", "a bank that lends money"),  # s1 (same doc as s2)
        _defn("base", "Lender", "a bank or fund that lends"),  # s2
        _defn("amend", "Lender", "a bank, fund, or trust that lends"),  # x (other doc)
    ]
    cands = build_candidates(defs, max_pairs=100)
    assert cands[0]["doc_a"] != cands[0]["doc_b"]  # a cross-document pair is first


def _pred(label, predicted):
    return {"label": label, "predicted": predicted}


def test_metrics_perfect():
    preds = [_pred("divergent", "divergent"), _pred("consistent", "consistent")]
    m = _metrics(preds)
    assert m["confusion"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 1}
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_metrics_mixed():
    preds = [
        _pred("divergent", "divergent"),
        _pred("divergent", "consistent"),
        _pred("consistent", "divergent"),
        _pred("consistent", "consistent"),
    ]
    m = _metrics(preds)
    assert m["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5


def test_metrics_no_positive_predictions_is_none():
    preds = [_pred("consistent", "consistent")]
    m = _metrics(preds)
    assert m["precision"] is None
    assert m["recall"] is None
    assert m["f1"] is None


def test_metrics_all_divergent_perfect():
    preds = [_pred("divergent", "divergent"), _pred("divergent", "divergent")]
    m = _metrics(preds)
    assert m["confusion"] == {"tp": 2, "fp": 0, "fn": 0, "tn": 0}
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_metrics_all_positives_wrong_is_zero_not_none():
    # tp=0 but predictions exist in both classes → F1 must be 0.0, not None.
    preds = [_pred("divergent", "consistent"), _pred("consistent", "divergent")]
    m = _metrics(preds)
    assert m["confusion"] == {"tp": 0, "fp": 1, "fn": 1, "tn": 0}
    assert m["precision"] == 0.0  # tp/(tp+fp) = 0/1
    assert m["recall"] == 0.0  # tp/(tp+fn) = 0/1
    assert m["f1"] == 0.0


def _corpus(cid, name):
    return SimpleNamespace(corpus_id=cid, corpus_name=name)


def test_resolve_corpus_id_by_name_id_none_and_error():
    corpora = [_corpus("uuid-abc", "atkins"), _corpus("uuid-def", "fcs-call-report")]
    assert resolve_corpus_id(corpora, "atkins") == "uuid-abc"  # by name
    assert resolve_corpus_id(corpora, "uuid-def") == "uuid-def"  # by raw id
    assert resolve_corpus_id(corpora, None) is None  # all corpora
    with pytest.raises(ValueError, match="unknown corpus"):
        resolve_corpus_id(corpora, "nope")


# --- labeler (label.py) ------------------------------------------------------


def _write_candidates(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _candidate(pid: str, term: str = "Board") -> dict:
    return {
        "pair_id": pid,
        "category": "review",
        "term": term,
        "def_a": "the board of directors",
        "def_b": "the supervisory board only",
        "doc_a": "deadbeef00000000",
        "doc_b": "feedface00000000",
        "label": "",
    }


def test_load_candidates_dedupes_across_files(tmp_path: Path) -> None:
    f1 = tmp_path / "a.jsonl"
    f2 = tmp_path / "b.jsonl"
    _write_candidates(f1, [_candidate("p1"), _candidate("p2")])
    _write_candidates(f2, [_candidate("p2"), _candidate("p3")])  # p2 overlaps
    cands = _load_candidates([f1, f2])
    assert [c["pair_id"] for c in cands] == ["p1", "p2", "p3"]


def test_labeler_writes_harness_schema_and_resumes(tmp_path: Path) -> None:
    src = tmp_path / "cand.jsonl"
    _write_candidates(src, [_candidate("p1"), _candidate("p2")])
    out = tmp_path / "labeled.jsonl"
    client = TestClient(create_app(_load_candidates([src]), out))

    assert client.post("/label", json={"pair_id": "p1", "label": "divergent"}).json() == {
        "ok": True,
        "labeled": 1,
        "total": 2,
    }
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["pair_id"] == "p1" and rows[0]["label"] == "divergent"
    # harness reads term/def_a/def_b/label/category — all present
    assert {"pair_id", "category", "term", "def_a", "def_b", "label"} <= set(rows[0])

    # A fresh app over the same out file resumes the label.
    assert _load_labels(out) == {"p1": "divergent"}


def test_labeler_relabel_and_clear(tmp_path: Path) -> None:
    src = tmp_path / "cand.jsonl"
    _write_candidates(src, [_candidate("p1")])
    out = tmp_path / "labeled.jsonl"
    client = TestClient(create_app(_load_candidates([src]), out))

    client.post("/label", json={"pair_id": "p1", "label": "divergent"})
    client.post("/label", json={"pair_id": "p1", "label": "consistent"})  # correction
    assert json.loads(out.read_text().splitlines()[0])["label"] == "consistent"

    client.post("/label", json={"pair_id": "p1", "label": ""})  # clear
    assert [line for line in out.read_text().splitlines() if line.strip()] == []


def test_labeler_rejects_bad_input(tmp_path: Path) -> None:
    src = tmp_path / "cand.jsonl"
    _write_candidates(src, [_candidate("p1")])
    client = TestClient(create_app(_load_candidates([src]), tmp_path / "out.jsonl"))
    assert client.post("/label", json={"pair_id": "p1", "label": "maybe"}).status_code == 400
    assert client.post("/label", json={"pair_id": "ghost", "label": "divergent"}).status_code == 404


def test_labeler_page_injects_data_not_placeholder(tmp_path: Path) -> None:
    src = tmp_path / "cand.jsonl"
    _write_candidates(src, [_candidate("p1", term="Quorum")])
    client = TestClient(create_app(_load_candidates([src]), tmp_path / "out.jsonl"))
    body = client.get("/").text
    assert "__DATA__" not in body  # template token replaced
    assert "Quorum" in body


def test_labeler_refuses_to_clobber_foreign_labels(tmp_path: Path) -> None:
    """--out containing labels for pair_ids outside the candidate set must abort,
    not get silently rewritten (e.g. someone aims --out at the curated set)."""
    src = tmp_path / "cand.jsonl"
    _write_candidates(src, [_candidate("p1")])
    out = tmp_path / "curated.jsonl"
    out.write_text(json.dumps({"pair_id": "other", "label": "consistent"}) + "\n")
    with pytest.raises(ValueError, match="not in the"):
        create_app(_load_candidates([src]), out)


def test_labeler_escapes_script_close_in_embedded_json(tmp_path: Path) -> None:
    """Corpus text with </script> must not break out of the inline <script>."""
    src = tmp_path / "cand.jsonl"
    evil = _candidate("p1")
    evil["def_a"] = "ends here </script><script>alert(1)</script>"
    _write_candidates(src, [evil])
    client = TestClient(create_app(_load_candidates([src]), tmp_path / "out.jsonl"))
    body = client.get("/").text
    assert "</script><script>alert(1)" not in body
    assert "<\\/script>" in body  # neutralized form present


def test_build_candidates_excludes_reference_assertions():
    """A kind=definition assertion whose source is a usage/reference (no
    `"Term" means`) is dropped — the eval candidate set must not pair the
    extractor's reference-as-definition mistakes against real definitions."""
    real = _defn("d1", "Affiliated Lender", "any Lender that is an Affiliate of Holdings")
    ref = Assertion.build(
        "d2",
        "Any Lender may assign its rights to an Affiliated Lender subject to limitations.",
        kind="definition",
        term="Affiliated Lender",
        definition_text="an Affiliated Lender subject to limitations",
    )
    assert build_candidates([real, ref], max_pairs=100) == []  # ref dropped -> singleton


def test_harness_run_passes_org_tuples_to_checker(tmp_path):
    """run() must hand (Assertion, org_key) tuples to the checker — the org-scope
    API. Passing bare Assertions makes DefinitionChecker._group raise on unpack;
    this exercises the full run() path with the fixture judge (no LLM)."""
    import json

    from benchmarks.definition_eval.harness import run

    cfg = tmp_path / "config.yml"
    cfg.write_text(
        "corpus_dir: ./c\njudge_provider: fixture\njudge_model: x\ndata_dir: ./d\nlog_dir: ./l\n"
    )
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        json.dumps(
            {
                "pair_id": "p1",
                "category": "review",
                "term": "Quorum",
                "def_a": "a majority of the directors",
                "def_b": "two-thirds of the directors",
                "label": "divergent",
            }
        )
        + "\n"
    )
    result = run(pairs, cfg)  # raised "cannot unpack non-iterable Assertion" before the fix
    assert result["metrics"]["n"] == 1
