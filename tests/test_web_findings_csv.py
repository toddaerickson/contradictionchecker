"""Tests for the findings CSV export route (ADR-0017 gap closer).

``GET /corpora/{id}/findings.csv`` exports the active corpus's LATEST run
honoring the active filter chip — exactly what the findings pane shows. The
stdlib ``csv`` module owns quoting/escaping; these tests round-trip a cell
that contains a comma and a double-quote with ``csv.reader`` to prove it.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.check.gate import CandidatePair
from consistency_checker.check.llm_judge import FixtureJudge, JudgeVerdict
from consistency_checker.check.nli_checker import FixtureNliChecker, NliResult
from consistency_checker.config import Config
from consistency_checker.extract.atomic_facts import FixtureExtractor
from consistency_checker.extract.schema import Assertion, Document
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.web.app import create_app
from tests.conftest import HashEmbedder


def _config(tmp_path: Path) -> Config:
    return Config(
        corpus_dir=tmp_path / "corpus",
        judge_provider="fixture",
        judge_model="test",
        data_dir=tmp_path / "store",
        log_dir=tmp_path / "logs",
        embedder_model="hash",
        nli_model="fixture",
        gate_similarity_threshold=-1.0,
        nli_contradiction_threshold=0.0,
    )


def _client(cfg: Config) -> TestClient:
    return TestClient(
        create_app(
            cfg,
            extractor=FixtureExtractor({}),
            embedder=HashEmbedder(dim=64),
            nli_checker=FixtureNliChecker({}),
            judge=FixtureJudge({}),
        )
    )


def _seed_one_corpus(cfg: Config, *, name: str = "Empty corpus") -> str:
    store = AssertionStore(cfg.db_path)
    store.migrate()
    cid = store.get_or_create_corpus(name, f"/{name}", "moonshot")
    store.close()
    return cid


def _record_finding(
    audit: AuditLogger,
    run_id: str,
    a: Assertion,
    b: Assertion,
    *,
    rationale: str,
) -> str:
    """Record one contradiction finding; return its pair_key."""
    audit.record_finding(
        run_id,
        candidate=CandidatePair(a=a, b=b, score=0.92),
        nli=NliResult.from_scores(p_contradiction=0.83, p_entailment=0.05, p_neutral=0.12),
        verdict=JudgeVerdict(
            assertion_a_id=a.assertion_id,
            assertion_b_id=b.assertion_id,
            verdict="contradiction",
            rationale=rationale,
            evidence_spans=[],
        ),
    )
    return ":".join(sorted([a.assertion_id, b.assertion_id]))


def _seed_corpus_with_two_findings(
    cfg: Config, *, name: str = "CSV corpus", text_a: str = "Revenue grew 12% in fiscal 2025."
) -> tuple[str, str, str, str]:
    """Seed a corpus + run with two contradiction findings.

    Returns (corpus_id, run_id, pair_key_1, text_a) — pair_key_1 is the first
    finding's key so a test can mark it and exercise the filter.
    """
    store = AssertionStore(cfg.db_path)
    store.migrate()
    cid = store.get_or_create_corpus(name, f"/{name}", "moonshot")
    doc_a = Document.from_content("Alpha body.", source_path="alpha.md", title="Alpha")
    doc_b = Document.from_content("Beta body.", source_path="beta.txt", title="Beta")
    store.add_document(doc_a, corpus_id=cid)
    store.add_document(doc_b, corpus_id=cid)
    a1 = Assertion.build(doc_a.doc_id, text_a)
    b1 = Assertion.build(doc_b.doc_id, "Revenue declined 5% in fiscal 2025.")
    a2 = Assertion.build(doc_a.doc_id, "EBITDA up 8%.")
    b2 = Assertion.build(doc_b.doc_id, "EBITDA down 3%.")
    store.add_assertions([a1, b1, a2, b2])

    audit = AuditLogger(store)
    run_id = audit.begin_run(corpus_id=cid)
    pair_key_1 = _record_finding(audit, run_id, a1, b1, rationale="Revenue contradiction.")
    _record_finding(audit, run_id, a2, b2, rationale="EBITDA contradiction.")
    audit.end_run(run_id, n_assertions=4, n_pairs_gated=2, n_pairs_judged=2, n_findings=2)
    store.close()
    return cid, run_id, pair_key_1, text_a


# --- 1) happy path: header + rows + known assertion text -----------------


def test_findings_csv_returns_csv_with_header_and_rows(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid, _rid, _pk, text_a = _seed_corpus_with_two_findings(cfg)
    client = _client(cfg)

    resp = client.get(f"/corpora/{cid}/findings.csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    cd = resp.headers["content-disposition"]
    assert cd.startswith("attachment; filename=")
    assert f"findings-{cid}-all.csv" in cd

    rows = list(csv.reader(io.StringIO(resp.text)))
    # Header + one row per finding (2 findings).
    assert rows[0] == [
        "finding_type",
        "judge_verdict",
        "reviewer_verdict",
        "doc_a",
        "assertion_a",
        "doc_b",
        "assertion_b",
        "rationale",
    ]
    assert len(rows) == 3
    # The known assertion text appears in the body.
    assert text_a in resp.text
    # Spot-check a data row's columns.
    body_cells = [cell for row in rows[1:] for cell in row]
    assert "contradiction" in body_cells
    assert "Revenue contradiction." in body_cells


# --- 2) filter narrows the rows ------------------------------------------


def test_findings_csv_filter_narrows_rows(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid, _rid, pair_key_1, _text_a = _seed_corpus_with_two_findings(cfg)

    # Mark the first finding confirmed.
    store = AssertionStore(cfg.db_path)
    try:
        AuditLogger(store).set_reviewer_verdict(
            pair_key=pair_key_1, detector_type="contradiction", verdict="confirmed"
        )
    finally:
        store.close()

    client = _client(cfg)

    resp_conf = client.get(f"/corpora/{cid}/findings.csv?filter=confirmed")
    assert resp_conf.status_code == 200
    assert "filename=" in resp_conf.headers["content-disposition"]
    assert f"findings-{cid}-confirmed.csv" in resp_conf.headers["content-disposition"]
    rows_conf = list(csv.reader(io.StringIO(resp_conf.text)))
    # Header + exactly the one confirmed finding.
    assert len(rows_conf) == 2
    assert "Revenue contradiction." in resp_conf.text
    assert "EBITDA contradiction." not in resp_conf.text
    # reviewer_verdict column reflects the mark.
    reviewer_idx = rows_conf[0].index("reviewer_verdict")
    assert rows_conf[1][reviewer_idx] == "confirmed"

    resp_open = client.get(f"/corpora/{cid}/findings.csv?filter=open")
    assert resp_open.status_code == 200
    rows_open = list(csv.reader(io.StringIO(resp_open.text)))
    # Header + the one unmarked (open) finding.
    assert len(rows_open) == 2
    assert "EBITDA contradiction." in resp_open.text
    assert "Revenue contradiction." not in resp_open.text


# --- 3) CSV escaping round-trips with csv.reader -------------------------


def test_findings_csv_escapes_comma_and_quote(tmp_path: Path) -> None:
    """An assertion text containing a comma AND a double-quote must round-trip
    exactly through Python's csv.reader — proving we rely on the csv module's
    quoting rather than hand-rolled escaping."""
    cfg = _config(tmp_path)
    nasty = 'Revenue grew 12%, then fell to "low" levels.'
    cid, _rid, _pk, _text = _seed_corpus_with_two_findings(cfg, text_a=nasty)
    client = _client(cfg)

    resp = client.get(f"/corpora/{cid}/findings.csv")
    assert resp.status_code == 200

    rows = list(csv.reader(io.StringIO(resp.text)))
    header = rows[0]
    assertion_a_idx = header.index("assertion_a")
    cells = [row[assertion_a_idx] for row in rows[1:]]
    # The nasty text round-trips into exactly one cell, byte-for-byte.
    assert nasty in cells


# --- 4) unknown corpus → 404 ---------------------------------------------


def test_findings_csv_404_on_unknown_corpus(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _seed_one_corpus(cfg, name="Real corpus")
    client = _client(cfg)
    resp = client.get("/corpora/does-not-exist/findings.csv")
    assert resp.status_code == 404


# --- 5) corpus with no run → header-only CSV, HTTP 200 -------------------


def test_findings_csv_header_only_when_no_run(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid = _seed_one_corpus(cfg, name="Quiet corpus")
    client = _client(cfg)
    resp = client.get(f"/corpora/{cid}/findings.csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    rows = list(csv.reader(io.StringIO(resp.text)))
    assert len(rows) == 1
    assert rows[0][0] == "finding_type"


# --- 6) unknown filter falls back to "all" -------------------------------


def test_findings_csv_unknown_filter_falls_back_to_all(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid, _rid, _pk, _text = _seed_corpus_with_two_findings(cfg)
    client = _client(cfg)
    resp = client.get(f"/corpora/{cid}/findings.csv?filter=banana")
    assert resp.status_code == 200
    # Filename reflects the sanitized filter, and both findings are present.
    assert f"findings-{cid}-all.csv" in resp.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(resp.text)))
    assert len(rows) == 3


# --- 7) the Export CSV button renders when findings exist ----------------


def test_export_button_renders_when_findings_present(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cid, _rid, _pk, _text = _seed_corpus_with_two_findings(cfg)
    client = _client(cfg)
    resp = client.get(f"/?corpus={cid}")
    assert resp.status_code == 200
    body = resp.text
    assert f'href="/corpora/{cid}/findings.csv?filter=all"' in body
    assert "Export CSV" in body
    assert "download" in body


def test_export_button_renders_when_active_filter_yields_zero_rows(tmp_path: Path) -> None:
    """The Export button is gated on ``counts.all`` (the corpus HAS findings),
    not on the filtered list. Selecting a filter chip with zero matches (here
    ``confirmed`` against two unconfirmed findings) must still show the button —
    consistent with the always-visible filter chips."""
    cfg = _config(tmp_path)
    cid, _rid, _pk, _text = _seed_corpus_with_two_findings(cfg)
    client = _client(cfg)
    resp = client.get(f"/?corpus={cid}&filter=confirmed")
    assert resp.status_code == 200
    body = resp.text
    # Active filter yields no confirmed rows...
    assert "No findings yet." in body
    # ...but the Export button is present and carries the active filter.
    assert "Export CSV" in body
    assert f'href="/corpora/{cid}/findings.csv?filter=confirmed"' in body


@pytest.mark.parametrize("missing_state", ["no_run", "no_corpus"])
def test_export_button_hidden_without_findings(tmp_path: Path, missing_state: str) -> None:
    cfg = _config(tmp_path)
    if missing_state == "no_run":
        _seed_one_corpus(cfg, name="Quiet corpus")
    client = _client(cfg)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Export CSV" not in resp.text


def test_export_button_carries_active_filter(tmp_path: Path) -> None:
    """The export link must carry the current filter so the download matches
    the displayed list."""
    cfg = _config(tmp_path)
    cid, _rid, pair_key_1, _text = _seed_corpus_with_two_findings(cfg)
    store = AssertionStore(cfg.db_path)
    try:
        AuditLogger(store).set_reviewer_verdict(
            pair_key=pair_key_1, detector_type="contradiction", verdict="confirmed"
        )
    finally:
        store.close()
    client = _client(cfg)
    resp = client.get(f"/?corpus={cid}&filter=confirmed")
    assert resp.status_code == 200
    assert f'href="/corpora/{cid}/findings.csv?filter=confirmed"' in resp.text
