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
from consistency_checker.web.app import _PAIR_KEY_RE, create_app
from tests.conftest import HashEmbedder

# A well-formed pair_key: ":".join(sorted([a_id, b_id])) of two 16-char sha256 hex ids.
VALID_PAIR_KEY = "0123456789abcdef:fedcba9876543210"
# A 3-ary key (multi-party finding): three 16-char sha256 hex ids joined by ':'.
VALID_TRIPLE_PAIR_KEY = "0123456789abcdef:fedcba9876543210:0011223344556677"


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
            "pair_key": VALID_PAIR_KEY,
            "detector_type": "contradiction",
            "verdict": "confirmed",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    store = AssertionStore(cfg.db_path)
    rows = store._conn.execute(
        "SELECT verdict FROM reviewer_verdicts WHERE pair_key = ?", (VALID_PAIR_KEY,)
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
            "pair_key": VALID_PAIR_KEY,
            "detector_type": "contradiction",
            "verdict": "confirmed",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    # The card's actions span is re-rendered in marked state and swapped into
    # the originating finding by its stable id (main-target swap).
    pk_safe = VALID_PAIR_KEY.replace(":", "-")
    assert f'id="cc-actions-{pk_safe}-contradiction"' in resp.text
    # The toast now OOB-inserts into the region (an outerHTML OOB needs a
    # pre-existing #cc-toast, which the shell never had — that was the bug).
    assert 'hx-swap-oob="afterbegin:#cc-toast-region"' in resp.text
    # The progress count still rides along as an outerHTML OOB swap.
    assert 'id="cc-progress-count-contradiction"' in resp.text
    assert 'hx-swap-oob="outerHTML"' in resp.text
    assert "Real issue" in resp.text
    # The filter chips refresh off this trigger.
    assert resp.headers.get("HX-Trigger") == "verdict-changed"


def test_post_verdicts_rejects_bogus_verdict(
    app_client: tuple[TestClient, Config],
) -> None:
    client, _cfg = app_client
    resp = client.post(
        "/verdicts",
        data={
            "pair_key": VALID_PAIR_KEY,
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
            "pair_key": VALID_PAIR_KEY,
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
            "pair_key": VALID_PAIR_KEY,
            "detector_type": "contradiction",
            "verdict": "confirmed",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    resp = client.post(
        "/verdicts/undo",
        data={
            "pair_key": VALID_PAIR_KEY,
            "detector_type": "contradiction",
            "prior_verdict": "",
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
            "pair_key": VALID_PAIR_KEY,
            "detector_type": "contradiction",
            "verdict": "confirmed",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    client.post(
        "/verdicts",
        data={
            "pair_key": VALID_PAIR_KEY,
            "detector_type": "contradiction",
            "verdict": "false_positive",
            "prior_verdict": "confirmed",
        },
        headers={"HX-Request": "true"},
    )
    resp = client.post(
        "/verdicts/undo",
        data={
            "pair_key": VALID_PAIR_KEY,
            "detector_type": "contradiction",
            "prior_verdict": "confirmed",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    store = AssertionStore(cfg.db_path)
    row = store._conn.execute(
        "SELECT verdict FROM reviewer_verdicts WHERE pair_key = ?", (VALID_PAIR_KEY,)
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
            "pair_key": VALID_PAIR_KEY,
            "detector_type": "contradiction",
            "prior_verdict": "banana",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 400


def test_post_verdicts_rejects_malformed_pair_key(
    app_client: tuple[TestClient, Config],
) -> None:
    """A pair_key that isn't ^[0-9a-f]{16}:[0-9a-f]{16}$ is rejected with 400."""
    client, cfg = app_client
    resp = client.post(
        "/verdicts",
        data={
            "pair_key": "nope",
            "detector_type": "contradiction",
            "verdict": "confirmed",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "invalid pair_key format"
    store = AssertionStore(cfg.db_path)
    count = store._conn.execute("SELECT COUNT(*) FROM reviewer_verdicts").fetchone()[0]
    assert count == 0  # rejected before any DB write
    store.close()

    # A well-formed pair_key on the same endpoint still succeeds.
    ok = client.post(
        "/verdicts",
        data={
            "pair_key": VALID_PAIR_KEY,
            "detector_type": "contradiction",
            "verdict": "confirmed",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    assert ok.status_code == 200


def test_pair_key_re_accepts_two_and_three_segments() -> None:
    """The widened guard accepts 2-ary and N-ary keys, rejects malformed input."""
    assert _PAIR_KEY_RE.fullmatch(VALID_PAIR_KEY) is not None
    assert _PAIR_KEY_RE.fullmatch(VALID_TRIPLE_PAIR_KEY) is not None
    assert _PAIR_KEY_RE.fullmatch("nope") is None
    # A single segment is no longer a valid pair/N-ary key.
    assert _PAIR_KEY_RE.fullmatch("0123456789abcdef") is None
    # Hex must be lowercase.
    assert _PAIR_KEY_RE.fullmatch("0123456789ABCDEF:fedcba9876543210") is None


def test_post_verdicts_accepts_three_ary_pair_key(
    app_client: tuple[TestClient, Config],
) -> None:
    """A 3-ary (multi-party) pair_key is accepted under the widened guard."""
    client, cfg = app_client
    resp = client.post(
        "/verdicts",
        data={
            "pair_key": VALID_TRIPLE_PAIR_KEY,
            "detector_type": "multi_party",
            "verdict": "confirmed",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    store = AssertionStore(cfg.db_path)
    rows = store._conn.execute(
        "SELECT verdict FROM reviewer_verdicts WHERE pair_key = ?",
        (VALID_TRIPLE_PAIR_KEY,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["verdict"] == "confirmed"
    store.close()


def test_post_verdicts_undo_rejects_malformed_pair_key(
    app_client: tuple[TestClient, Config],
) -> None:
    """undo mirrors post_verdict: a malformed pair_key is rejected before any DB write."""
    client, cfg = app_client
    resp = client.post(
        "/verdicts/undo",
        data={
            "pair_key": "nope",
            "detector_type": "contradiction",
            "prior_verdict": "",
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "invalid pair_key format"
    store = AssertionStore(cfg.db_path)
    count = store._conn.execute("SELECT COUNT(*) FROM reviewer_verdicts").fetchone()[0]
    assert count == 0  # rejected before any DB write
    store.close()
