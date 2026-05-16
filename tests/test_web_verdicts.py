"""Tests for the verdict-setting POST endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from consistency_checker.check.llm_judge import FixtureJudge
from consistency_checker.check.nli_checker import FixtureNliChecker
from consistency_checker.config import Config
from consistency_checker.extract.atomic_facts import FixtureExtractor
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.web.app import create_app
from tests.conftest import HashEmbedder


@pytest.fixture
def app_client(tmp_path: Path) -> tuple[TestClient, Config]:
    cfg = Config(
        corpus_dir=tmp_path / "corpus",
        judge_provider="fixture",
        judge_model="test",
        data_dir=tmp_path / "store",
        log_dir=tmp_path / "logs",
        embedder_model="hash",
        nli_model="fixture",
    )
    AssertionStore(cfg.db_path).migrate()
    app = create_app(
        cfg,
        extractor=FixtureExtractor({}),
        embedder=HashEmbedder(dim=64),
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
    )
    return TestClient(app), cfg


def test_post_verdicts_inserts_row(
    app_client: tuple[TestClient, Config],
) -> None:
    client, cfg = app_client
    resp = client.post(
        "/verdicts",
        data={
            "pair_key": "a:b",
            "detector_type": "contradiction",
            "verdict": "confirmed",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    store = AssertionStore(cfg.db_path)
    rows = store._conn.execute(
        "SELECT verdict FROM reviewer_verdicts WHERE pair_key = ?", ("a:b",)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["verdict"] == "confirmed"
    store.close()


def test_post_verdicts_response_contains_oob_swaps(
    app_client: tuple[TestClient, Config],
) -> None:
    client, _cfg = app_client
    resp = client.post(
        "/verdicts",
        data={
            "pair_key": "a:b",
            "detector_type": "contradiction",
            "verdict": "confirmed",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    assert 'id="cc-toast"' in resp.text
    assert 'hx-swap-oob="outerHTML"' in resp.text
    assert 'id="cc-progress-count-contradiction"' in resp.text
    assert "Real issue" in resp.text


def test_post_verdicts_rejects_bogus_verdict(
    app_client: tuple[TestClient, Config],
) -> None:
    client, _cfg = app_client
    resp = client.post(
        "/verdicts",
        data={
            "pair_key": "a:b",
            "detector_type": "contradiction",
            "verdict": "banana",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 400


def test_post_verdicts_rejects_bogus_detector_type(
    app_client: tuple[TestClient, Config],
) -> None:
    client, _cfg = app_client
    resp = client.post(
        "/verdicts",
        data={
            "pair_key": "a:b",
            "detector_type": "bogus",
            "verdict": "confirmed",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 400


def test_post_verdicts_undo_first_click_case_deletes(
    app_client: tuple[TestClient, Config],
) -> None:
    """Undo with empty prior_verdict deletes the row (first-click case)."""
    client, cfg = app_client
    client.post(
        "/verdicts",
        data={
            "pair_key": "a:b", "detector_type": "contradiction",
            "verdict": "confirmed", "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    resp = client.post(
        "/verdicts/undo",
        data={
            "pair_key": "a:b", "detector_type": "contradiction", "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    store = AssertionStore(cfg.db_path)
    rows = store._conn.execute("SELECT COUNT(*) FROM reviewer_verdicts").fetchone()
    assert rows[0] == 0
    store.close()


def test_post_verdicts_undo_rejudge_case_restores_prior(
    app_client: tuple[TestClient, Config],
) -> None:
    """Undo with non-empty prior_verdict re-sets to the prior value."""
    client, cfg = app_client
    client.post(
        "/verdicts",
        data={
            "pair_key": "a:b", "detector_type": "contradiction",
            "verdict": "confirmed", "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    client.post(
        "/verdicts",
        data={
            "pair_key": "a:b", "detector_type": "contradiction",
            "verdict": "false_positive", "prior_verdict": "confirmed",
        },
        headers={"HX-Request": "true"},
    )
    resp = client.post(
        "/verdicts/undo",
        data={
            "pair_key": "a:b", "detector_type": "contradiction",
            "prior_verdict": "confirmed",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    store = AssertionStore(cfg.db_path)
    row = store._conn.execute(
        "SELECT verdict FROM reviewer_verdicts WHERE pair_key = ?", ("a:b",)
    ).fetchone()
    assert row["verdict"] == "confirmed"
    store.close()


def test_post_verdicts_undo_rejects_bogus_prior_verdict(
    app_client: tuple[TestClient, Config],
) -> None:
    client, _cfg = app_client
    resp = client.post(
        "/verdicts/undo",
        data={
            "pair_key": "a:b", "detector_type": "contradiction",
            "prior_verdict": "banana",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 400
