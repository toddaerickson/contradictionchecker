"""Tests for runs progress streaming and management API endpoints.

Tests cover all endpoints with success and error cases:
- GET /api/runs (list all runs)
- GET /api/runs/{run_id} (get one run)
- GET /api/runs/{run_id}/progress (stream progress via SSE)
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from consistency_checker.config import Config


def _create_test_run(
    db_path: str,
    run_id: str = "test_run_001",
    corpus_id: str = "test_corpus",
    status: str = "in_progress",
    message_log: str | None = None,
) -> None:
    """Helper to create a test run in the database."""
    import sqlite3
    from datetime import datetime

    from consistency_checker.index.assertion_store import AssertionStore

    # Ensure database is migrated
    store = AssertionStore(str(db_path))
    store.migrate()
    store.close()

    # Now insert the test data
    conn = sqlite3.connect(str(db_path))
    try:
        # Ensure corpora table exists
        conn.execute(
            """
            INSERT OR IGNORE INTO corpora
            (corpus_id, corpus_name, corpus_path, judge_provider, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                corpus_id,
                f"Corpus {corpus_id}",
                f"/tmp/{corpus_id}",
                "moonshot",
                datetime.now().isoformat(),
                datetime.now().isoformat(),
            ),
        )

        # Insert the run
        now = datetime.now().isoformat(timespec="microseconds")
        conn.execute(
            """
            INSERT OR REPLACE INTO runs
            (run_id, corpus_id, started_at, completed_at, status, message_log)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, corpus_id, now, None, status, message_log),
        )
        conn.commit()
    finally:
        conn.close()


def test_list_runs_empty(web_client: TestClient, web_config: Config) -> None:
    """Test listing runs when no runs exist."""
    response = web_client.get("/api/runs")

    assert response.status_code == 200
    data = response.json()

    assert "runs" in data
    assert data["runs"] == []


def test_list_runs_with_multiple_runs(web_client: TestClient, web_config: Config) -> None:
    """Test listing multiple runs in descending order by started_at."""
    import time

    # Create test runs with a small delay to ensure distinct timestamps
    _create_test_run(
        str(web_config.db_path),
        run_id="run_001",
        status="completed",
        message_log='{"timestamp": "2026-05-16T10:00:00.000000", "phase": "parsing", "message": "Started"}\n',
    )
    time.sleep(0.01)
    _create_test_run(
        str(web_config.db_path),
        run_id="run_002",
        status="in_progress",
        message_log='{"timestamp": "2026-05-16T11:00:00.000000", "phase": "parsing", "message": "Started"}\n',
    )

    response = web_client.get("/api/runs")

    assert response.status_code == 200
    data = response.json()

    assert len(data["runs"]) == 2
    # Most recent first
    assert data["runs"][0]["run_id"] == "run_002"
    assert data["runs"][1]["run_id"] == "run_001"


def test_list_runs_message_count(web_client: TestClient, web_config: Config) -> None:
    """Test that message_count is correctly reported."""
    message_log = (
        '{"timestamp": "2026-05-16T10:00:00.000000", "phase": "parsing", "message": "Msg 1"}\n'
        '{"timestamp": "2026-05-16T10:01:00.000000", "phase": "extracting", "message": "Msg 2"}\n'
        '{"timestamp": "2026-05-16T10:02:00.000000", "phase": "judging", "message": "Msg 3"}\n'
    )
    _create_test_run(
        str(web_config.db_path),
        run_id="run_with_messages",
        message_log=message_log,
    )

    response = web_client.get("/api/runs")

    assert response.status_code == 200
    data = response.json()

    assert len(data["runs"]) == 1
    assert data["runs"][0]["message_count"] == 3


def test_get_run_success(web_client: TestClient, web_config: Config) -> None:
    """Test retrieving a specific run."""
    _create_test_run(
        str(web_config.db_path),
        run_id="test_run",
        corpus_id="test_corpus",
        status="in_progress",
    )

    response = web_client.get("/api/runs/test_run")

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "run_id" in data
    assert "corpus_id" in data
    assert "started_at" in data
    assert "completed_at" in data
    assert "status" in data
    assert "message_count" in data

    # Verify content
    assert data["run_id"] == "test_run"
    assert data["corpus_id"] == "test_corpus"
    assert data["status"] == "in_progress"
    assert data["completed_at"] is None
    assert data["message_count"] == 0


def test_get_run_not_found(web_client: TestClient) -> None:
    """Test retrieving a non-existent run returns 404."""
    response = web_client.get("/api/runs/nonexistent_run")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_get_run_with_messages(web_client: TestClient, web_config: Config) -> None:
    """Test retrieving a run with messages counts them."""
    message_log = (
        '{"timestamp": "2026-05-16T10:00:00.000000", "phase": "parsing", "message": "Msg 1"}\n'
        '{"timestamp": "2026-05-16T10:01:00.000000", "phase": "extracting", "message": "Msg 2"}\n'
    )
    _create_test_run(
        str(web_config.db_path),
        run_id="run_with_msgs",
        message_log=message_log,
    )

    response = web_client.get("/api/runs/run_with_msgs")

    assert response.status_code == 200
    data = response.json()
    assert data["message_count"] == 2


def test_stream_progress_run_not_found(web_client: TestClient) -> None:
    """Test streaming progress for non-existent run returns 404."""
    response = web_client.get("/api/runs/nonexistent_run/progress")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_stream_progress_empty_log(web_client: TestClient, web_config: Config) -> None:
    """Test streaming progress for completed run with no messages."""
    _create_test_run(
        str(web_config.db_path),
        run_id="run_no_messages",
        status="completed",
        message_log=None,
    )

    response = web_client.get("/api/runs/run_no_messages/progress")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    # Collect all events
    events = []
    for line in response.text.split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    # Should have only the final completion message
    assert len(events) == 1
    assert events[0]["phase"] == "complete"


def test_stream_progress_message_format(web_client: TestClient, web_config: Config) -> None:
    """Test that streamed messages have correct format."""
    message_log = (
        '{"timestamp": "2026-05-16T14:23:45.123456", "phase": "parsing", "message": "Parsing document 1/12: report.pdf"}\n'
        '{"timestamp": "2026-05-16T14:23:47.456789", "phase": "extracting", "message": "Extracted 45 assertions from report.pdf"}\n'
    )
    _create_test_run(
        str(web_config.db_path),
        run_id="run_with_format",
        status="completed",
        message_log=message_log,
    )

    response = web_client.get("/api/runs/run_with_format/progress")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    # Verify SSE event format
    lines = response.text.split("\n\n")
    event_lines = [line for line in lines if line.startswith("data: ")]
    assert len(event_lines) >= 2

    # Parse first two messages
    events = [json.loads(line[6:]) for line in event_lines[:2]]

    # Verify message structure
    assert events[0]["timestamp"] == "2026-05-16T14:23:45.123456"
    assert events[0]["phase"] == "parsing"
    assert events[0]["message"] == "Parsing document 1/12: report.pdf"

    assert events[1]["timestamp"] == "2026-05-16T14:23:47.456789"
    assert events[1]["phase"] == "extracting"
    assert events[1]["message"] == "Extracted 45 assertions from report.pdf"


def test_stream_progress_iso8601_timestamps(web_client: TestClient, web_config: Config) -> None:
    """Test that streamed messages have valid ISO8601 timestamps."""
    message_log = (
        '{"timestamp": "2026-05-16T14:23:45.123456", "phase": "parsing", "message": "Test"}\n'
    )
    _create_test_run(
        str(web_config.db_path),
        run_id="run_iso_timestamps",
        status="completed",
        message_log=message_log,
    )

    response = web_client.get("/api/runs/run_iso_timestamps/progress")

    assert response.status_code == 200

    events = []
    for line in response.text.split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    # Check ISO8601 format for all events (including completion message)
    for event in events:
        if event.get("timestamp"):
            # Should parse successfully as ISO8601
            dt = datetime.fromisoformat(event["timestamp"])
            assert isinstance(dt, datetime)


def test_stream_progress_sse_headers(web_client: TestClient, web_config: Config) -> None:
    """Test that SSE response has correct headers."""
    _create_test_run(
        str(web_config.db_path),
        run_id="run_headers",
        status="completed",
    )

    response = web_client.get("/api/runs/run_headers/progress")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["connection"] == "keep-alive"


def test_stream_progress_multiple_messages(web_client: TestClient, web_config: Config) -> None:
    """Test streaming multiple messages in order."""
    message_log = (
        '{"timestamp": "2026-05-16T10:00:00.000000", "phase": "parsing", "message": "Msg 1"}\n'
        '{"timestamp": "2026-05-16T10:01:00.000000", "phase": "extracting", "message": "Msg 2"}\n'
        '{"timestamp": "2026-05-16T10:02:00.000000", "phase": "judging", "message": "Msg 3"}\n'
        '{"timestamp": "2026-05-16T10:03:00.000000", "phase": "summarizing", "message": "Msg 4"}\n'
    )
    _create_test_run(
        str(web_config.db_path),
        run_id="run_multiple_msgs",
        status="completed",
        message_log=message_log,
    )

    response = web_client.get("/api/runs/run_multiple_msgs/progress")

    assert response.status_code == 200

    events = []
    for line in response.text.split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    # Should have all 4 original messages + completion message
    assert len(events) == 5
    assert events[0]["message"] == "Msg 1"
    assert events[1]["message"] == "Msg 2"
    assert events[2]["message"] == "Msg 3"
    assert events[3]["message"] == "Msg 4"
    assert events[4]["phase"] == "complete"


def test_stream_progress_malformed_json_graceful(
    web_client: TestClient, web_config: Config
) -> None:
    """Test that malformed JSON in message_log is skipped gracefully."""
    message_log = (
        '{"timestamp": "2026-05-16T10:00:00.000000", "phase": "parsing", "message": "Valid"}\n'
        "this is not json\n"
        '{"timestamp": "2026-05-16T10:01:00.000000", "phase": "extracting", "message": "Also Valid"}\n'
    )
    _create_test_run(
        str(web_config.db_path),
        run_id="run_malformed",
        status="completed",
        message_log=message_log,
    )

    response = web_client.get("/api/runs/run_malformed/progress")

    assert response.status_code == 200

    events = []
    for line in response.text.split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    # Should skip the malformed line and have 2 valid messages + completion
    assert len(events) == 3
    assert events[0]["message"] == "Valid"
    assert events[1]["message"] == "Also Valid"
    assert events[2]["phase"] == "complete"


def test_stream_progress_large_message_log(web_client: TestClient, web_config: Config) -> None:
    """Test streaming a large message log (1000+ messages)."""
    # Generate 1000 messages
    lines = []
    for i in range(1000):
        msg = {
            "timestamp": f"2026-05-16T10:{i % 60:02d}:{i % 60:02d}.000000",
            "phase": "processing",
            "message": f"Processing item {i}",
        }
        lines.append(json.dumps(msg))

    message_log = "\n".join(lines) + "\n"

    _create_test_run(
        str(web_config.db_path),
        run_id="run_large_log",
        status="completed",
        message_log=message_log,
    )

    response = web_client.get("/api/runs/run_large_log/progress")

    assert response.status_code == 200

    events = []
    for line in response.text.split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    # Should have all 1000 messages + completion message
    assert len(events) == 1001
    # Verify order is preserved
    for i in range(999):
        assert events[i]["message"] == f"Processing item {i}"


def test_stream_progress_in_progress_run_sends_completion_event(
    web_client: TestClient, web_config: Config
) -> None:
    """Test that a completed run sends a final completion event."""
    _create_test_run(
        str(web_config.db_path),
        run_id="run_in_progress",
        status="completed",
        message_log=None,
    )

    response = web_client.get("/api/runs/run_in_progress/progress")

    assert response.status_code == 200

    events = []
    for line in response.text.split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    # Should at least have completion event
    assert len(events) >= 1
    # Last event should be completion
    assert events[-1]["phase"] == "complete"


def test_stream_progress_failed_run_shows_failed_status(
    web_client: TestClient, web_config: Config
) -> None:
    """Test that failed run shows proper completion message."""
    _create_test_run(
        str(web_config.db_path),
        run_id="run_failed",
        status="failed",
        message_log='{"timestamp": "2026-05-16T10:00:00.000000", "phase": "parsing", "message": "Error occurred"}\n',
    )

    response = web_client.get("/api/runs/run_failed/progress")

    assert response.status_code == 200

    events = []
    for line in response.text.split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    # Should have original message + completion
    assert len(events) == 2
    assert events[0]["message"] == "Error occurred"
    assert events[1]["phase"] == "complete"
    assert "Failed" in events[1]["message"]


def test_stream_progress_empty_lines_in_log(web_client: TestClient, web_config: Config) -> None:
    """Test that empty lines in message_log are handled."""
    message_log = (
        '{"timestamp": "2026-05-16T10:00:00.000000", "phase": "parsing", "message": "Msg 1"}\n'
        "\n"
        "\n"
        '{"timestamp": "2026-05-16T10:01:00.000000", "phase": "extracting", "message": "Msg 2"}\n'
    )
    _create_test_run(
        str(web_config.db_path),
        run_id="run_empty_lines",
        status="completed",
        message_log=message_log,
    )

    response = web_client.get("/api/runs/run_empty_lines/progress")

    assert response.status_code == 200

    events = []
    for line in response.text.split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    # Should skip empty lines and have 2 messages + completion
    assert len(events) == 3
    assert events[0]["message"] == "Msg 1"
    assert events[1]["message"] == "Msg 2"
    assert events[2]["phase"] == "complete"


# ---------------------------------------------------------------------------
# POST /api/corpora/{corpus_id}/runs — start a run
# ---------------------------------------------------------------------------


def _create_test_corpus(db_path: str, corpus_id: str = "test_corpus") -> None:
    """Helper to create a corpus in the database."""
    import sqlite3
    from datetime import datetime

    from consistency_checker.index.assertion_store import AssertionStore

    store = AssertionStore(str(db_path))
    store.migrate()
    store.close()

    conn = sqlite3.connect(str(db_path))
    try:
        now = datetime.now().isoformat(timespec="microseconds")
        conn.execute(
            """
            INSERT OR IGNORE INTO corpora
            (corpus_id, corpus_name, corpus_path, judge_provider, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (corpus_id, f"Corpus {corpus_id}", f"/tmp/{corpus_id}", "moonshot", now, now),
        )
        conn.commit()
    finally:
        conn.close()


def test_start_run_creates_run(web_client: TestClient, web_config: Config) -> None:
    """Test that POST /api/corpora/{corpus_id}/runs creates a run and returns 201."""
    _create_test_corpus(str(web_config.db_path), corpus_id="corpus-for-run")

    response = web_client.post("/api/corpora/corpus-for-run/runs", json={})

    assert response.status_code == 201
    data = response.json()

    assert "run_id" in data
    assert data["corpus_id"] == "corpus-for-run"
    assert "started_at" in data
    assert data["status"] == "in_progress"
    assert data["run_id"] != ""


def test_start_run_returns_unique_ids(web_client: TestClient, web_config: Config) -> None:
    """Test that two runs created for the same corpus get different run_ids."""
    _create_test_corpus(str(web_config.db_path), corpus_id="corpus-two-runs")

    r1 = web_client.post("/api/corpora/corpus-two-runs/runs", json={})
    r2 = web_client.post("/api/corpora/corpus-two-runs/runs", json={})

    assert r1.status_code == 201
    assert r2.status_code == 201
    assert r1.json()["run_id"] != r2.json()["run_id"]


def test_start_run_corpus_not_found(web_client: TestClient) -> None:
    """Test that POST to non-existent corpus returns 404."""
    response = web_client.post("/api/corpora/nonexistent-corpus/runs", json={})

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_start_run_with_judge_provider(web_client: TestClient, web_config: Config) -> None:
    """Test that judge_provider can be specified."""
    _create_test_corpus(str(web_config.db_path), corpus_id="corpus-with-provider")

    response = web_client.post(
        "/api/corpora/corpus-with-provider/runs",
        json={"judge_provider": "anthropic"},
    )

    assert response.status_code == 201
    assert response.json()["status"] == "in_progress"


def test_start_run_invalid_judge_provider(web_client: TestClient, web_config: Config) -> None:
    """Test that invalid judge_provider returns 400."""
    _create_test_corpus(str(web_config.db_path), corpus_id="corpus-bad-provider")

    response = web_client.post(
        "/api/corpora/corpus-bad-provider/runs",
        json={"judge_provider": "openai"},
    )

    assert response.status_code == 400
    assert "judge_provider" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /api/runs/{run_id}/messages — append message to run log
# ---------------------------------------------------------------------------


def test_append_message_returns_200(web_client: TestClient, web_config: Config) -> None:
    """Test appending a message returns 200 with the appended message data."""
    _create_test_run(str(web_config.db_path), run_id="run-append-1")

    response = web_client.post(
        "/api/runs/run-append-1/messages",
        json={"message": "Parsing document 1/5", "phase": "parsing"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == "run-append-1"
    assert data["message"] == "Parsing document 1/5"
    assert data["phase"] == "parsing"
    assert "appended_at" in data


def test_append_message_run_not_found(web_client: TestClient) -> None:
    """Test appending to a non-existent run returns 404."""
    response = web_client.post(
        "/api/runs/nonexistent-run/messages",
        json={"message": "hello", "phase": "parsing"},
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_append_multiple_messages(web_client: TestClient, web_config: Config) -> None:
    """Test that multiple appended messages accumulate in the log."""
    _create_test_run(str(web_config.db_path), run_id="run-multi-append")

    # Append three messages
    web_client.post(
        "/api/runs/run-multi-append/messages",
        json={"message": "First", "phase": "parsing"},
    )
    web_client.post(
        "/api/runs/run-multi-append/messages",
        json={"message": "Second", "phase": "extraction"},
    )
    web_client.post(
        "/api/runs/run-multi-append/messages",
        json={"message": "Third", "phase": "judging"},
    )

    # Verify via GET /api/runs/{run_id}
    r = web_client.get("/api/runs/run-multi-append")
    assert r.status_code == 200
    assert r.json()["message_count"] == 3


def test_append_message_missing_message_field(web_client: TestClient, web_config: Config) -> None:
    """Test that missing 'message' field returns 400."""
    _create_test_run(str(web_config.db_path), run_id="run-missing-msg-field")

    response = web_client.post(
        "/api/runs/run-missing-msg-field/messages",
        json={"phase": "parsing"},
    )

    assert response.status_code == 400


def test_append_message_missing_phase_field(web_client: TestClient, web_config: Config) -> None:
    """Test that missing 'phase' field returns 400."""
    _create_test_run(str(web_config.db_path), run_id="run-missing-phase-field")

    response = web_client.post(
        "/api/runs/run-missing-phase-field/messages",
        json={"message": "Something happened"},
    )

    assert response.status_code == 400


# ---------------------------------------------------------------------------
# PATCH /api/runs/{run_id} — update run status
# ---------------------------------------------------------------------------


def test_patch_run_completed(web_client: TestClient, web_config: Config) -> None:
    """Test updating run status to 'completed' returns updated run."""
    _create_test_run(str(web_config.db_path), run_id="run-patch-complete")

    response = web_client.patch(
        "/api/runs/run-patch-complete",
        json={"status": "completed"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == "run-patch-complete"
    assert data["status"] == "completed"
    assert data["completed_at"] is not None


def test_patch_run_failed(web_client: TestClient, web_config: Config) -> None:
    """Test updating run status to 'failed' returns updated run."""
    _create_test_run(str(web_config.db_path), run_id="run-patch-failed")

    response = web_client.patch(
        "/api/runs/run-patch-failed",
        json={"status": "failed"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"
    assert data["completed_at"] is not None


def test_patch_run_with_explicit_completed_at(web_client: TestClient, web_config: Config) -> None:
    """Test that an explicit completed_at is stored and returned as a valid ISO8601 timestamp."""
    _create_test_run(str(web_config.db_path), run_id="run-patch-explicit-ts")
    explicit_ts = "2026-05-16T14:30:00.000000"

    response = web_client.patch(
        "/api/runs/run-patch-explicit-ts",
        json={"status": "completed", "completed_at": explicit_ts},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    # completed_at should be parseable and match the date/time we provided
    from datetime import datetime as _dt

    completed = _dt.fromisoformat(data["completed_at"])
    expected = _dt.fromisoformat(explicit_ts)
    assert completed == expected


def test_patch_run_not_found(web_client: TestClient) -> None:
    """Test PATCH on non-existent run returns 404."""
    response = web_client.patch(
        "/api/runs/nonexistent-run",
        json={"status": "completed"},
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_patch_run_invalid_status(web_client: TestClient, web_config: Config) -> None:
    """Test PATCH with invalid status returns 400."""
    _create_test_run(str(web_config.db_path), run_id="run-patch-bad-status")

    response = web_client.patch(
        "/api/runs/run-patch-bad-status",
        json={"status": "pending"},
    )

    assert response.status_code == 400
    assert "Invalid status" in response.json()["detail"]
