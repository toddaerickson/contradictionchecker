"""Progress streaming and run management API endpoints.

Provides REST endpoints for streaming run progress via Server-Sent Events (SSE)
and retrieving run metadata. Each run is a processing job for a corpus.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from consistency_checker.logging_setup import get_logger
from consistency_checker.models.ui import Run

_log = get_logger(__name__)

router = APIRouter(prefix="/api/runs", tags=["runs"])
corpora_runs_router = APIRouter(prefix="/api/corpora", tags=["runs"])


def _parse_message_log(message_log: str | None) -> list[dict[str, Any]]:
    """Parse NDJSON message log into a list of message dictionaries.

    Args:
        message_log: Newline-delimited JSON string, or None if empty.

    Returns:
        List of parsed message objects, preserving order.
        Gracefully skips malformed JSON lines.
    """
    if not message_log:
        return []

    messages = []
    for line in message_log.strip().split("\n"):
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
            messages.append(msg)
        except json.JSONDecodeError as e:
            _log.warning("Skipping malformed JSON in message_log: %s", e)
            continue

    return messages


@router.get("", status_code=200)
def list_runs(request: Request) -> dict[str, Any]:
    """List all runs.

    Returns:
        {
            "runs": [
                {
                    "run_id": "...",
                    "corpus_id": "...",
                    "started_at": "ISO8601",
                    "completed_at": "ISO8601 or null",
                    "status": "in_progress|completed|failed",
                    "message_count": N
                },
                ...
            ]
        }

    Behavior:
        - Return all runs from database, ordered by started_at descending
        - Include count of messages per run
    """
    config = request.app.state.config

    try:
        from consistency_checker.index.assertion_store import AssertionStore

        store = AssertionStore(config.db_path)
        store.migrate()
        try:
            conn = store._conn
            rows = conn.execute(
                """
                SELECT run_id, corpus_id, started_at, completed_at, status, message_log
                FROM runs
                ORDER BY started_at DESC
                """
            ).fetchall()

            runs = []
            for row in rows:
                run = Run.from_row(row)
                message_count = len(_parse_message_log(run.message_log))
                runs.append(
                    {
                        "run_id": run.run_id,
                        "corpus_id": run.corpus_id,
                        "started_at": run.started_at.isoformat(),
                        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                        "status": run.status,
                        "message_count": message_count,
                    }
                )

            return {"runs": runs}
        finally:
            store.close()
    except Exception as e:
        _log.error("Failed to list runs: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to list runs: {e}") from e


@router.get("/{run_id}", status_code=200)
def get_run(request: Request, run_id: str) -> dict[str, Any]:
    """Get a specific run.

    Returns:
        {
            "run_id": "...",
            "corpus_id": "...",
            "started_at": "ISO8601",
            "completed_at": "ISO8601 or null",
            "status": "in_progress|completed|failed",
            "message_count": N
        }

    Raises:
        404: run_id doesn't exist.
    """
    config = request.app.state.config

    try:
        from consistency_checker.index.assertion_store import AssertionStore

        store = AssertionStore(config.db_path)
        store.migrate()
        try:
            conn = store._conn
            row = conn.execute(
                """
                SELECT run_id, corpus_id, started_at, completed_at, status, message_log
                FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

            if row is None:
                raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")

            run = Run.from_row(row)
            message_count = len(_parse_message_log(run.message_log))

            return {
                "run_id": run.run_id,
                "corpus_id": run.corpus_id,
                "started_at": run.started_at.isoformat(),
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "status": run.status,
                "message_count": message_count,
            }
        finally:
            store.close()
    except HTTPException:
        raise
    except Exception as e:
        _log.error("Failed to get run %s: %s", run_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to get run: {e}") from e


async def _sse_event_generator(run_id: str, config: Any) -> AsyncGenerator[str, None]:
    """Generate Server-Sent Events stream for a run's progress.

    Yields SSE-formatted lines: `data: <JSON>\n\n`

    Behavior:
        1. Fetch run from database
        2. Return 404 if not found (handled by caller)
        3. Parse message_log and emit all existing messages
        4. If run is in_progress, poll for new messages every 1 second
        5. Stop polling when status changes to 'completed' or 'failed'
        6. Send final message with phase="complete"

    Args:
        run_id: The run ID to stream progress for.
        config: FastAPI app config object.

    Yields:
        SSE-formatted event lines.
    """
    from consistency_checker.index.assertion_store import AssertionStore

    # Initial fetch to check if run exists and get current state
    store = AssertionStore(config.db_path)
    store.migrate()
    try:
        conn = store._conn
        row = conn.execute(
            """
            SELECT run_id, corpus_id, started_at, completed_at, status, message_log
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()

        if row is None:
            # Can't use HTTPException in async generator, so just close and let caller handle
            store.close()
            return

        run = Run.from_row(row)
        messages = _parse_message_log(run.message_log)

        # Emit all existing messages
        for msg in messages:
            yield f"data: {json.dumps(msg)}\n\n"

        # Poll for new messages if run is still in progress
        last_message_count = len(messages)
        poll_count = 0
        max_polls = 1800  # ~30 minutes at 1-second polling

        while run.status == "in_progress" and poll_count < max_polls:
            await asyncio.sleep(1)
            poll_count += 1

            # Re-fetch run status and message log
            row = conn.execute(
                """
                SELECT run_id, corpus_id, started_at, completed_at, status, message_log
                FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

            if row is None:
                break

            run = Run.from_row(row)
            messages = _parse_message_log(run.message_log)

            # Emit any new messages
            if len(messages) > last_message_count:
                for msg in messages[last_message_count:]:
                    yield f"data: {json.dumps(msg)}\n\n"
                last_message_count = len(messages)

        # Send final completion message
        final_msg = {
            "timestamp": datetime.now().isoformat(timespec="microseconds"),
            "phase": "complete",
            "message": "Done." if run.status == "completed" else f"Failed: {run.status}",
        }
        yield f"data: {json.dumps(final_msg)}\n\n"

    finally:
        store.close()


@router.get("/{run_id}/progress")
async def stream_progress(request: Request, run_id: str) -> StreamingResponse:
    """Stream progress messages for a run via Server-Sent Events (SSE).

    Returns: Server-Sent Events stream (text/event-stream content type)

    Event format: `data: {"timestamp": "ISO8601", "phase": "...", "message": "..."}\n\n`

    Behavior:
        - Look up run_id in `runs` table
        - Return 404 if run_id doesn't exist
        - Stream all messages from `message_log` (one JSON object per line, newline-delimited)
        - If run is still in progress, stream completed messages then keep connection open
        - Auto-close when run.status changes to 'completed' or 'failed'
        - Clients can listen for `data: {"phase": "complete"}` to know streaming is done
        - Each message should be a JSON object with: timestamp (ISO8601), phase, message

    Raises:
        404: run_id doesn't exist.
    """
    config = request.app.state.config

    # Check if run exists before returning streaming response
    try:
        from consistency_checker.index.assertion_store import AssertionStore

        store = AssertionStore(config.db_path)
        store.migrate()
        try:
            conn = store._conn
            row = conn.execute(
                """
                SELECT run_id FROM runs WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

            if row is None:
                raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")
        finally:
            store.close()
    except HTTPException:
        raise
    except Exception as e:
        _log.error("Failed to check run %s: %s", run_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to check run: {e}") from e

    return StreamingResponse(
        _sse_event_generator(run_id, config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/{run_id}/messages", status_code=200)
def append_message(
    request: Request,
    run_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Append a progress message to a run's log (internal, used by pipeline).

    Request body:
        {
            "message": "string",
            "phase": "string"
        }

    Returns:
        {
            "run_id": "...",
            "message": "...",
            "phase": "...",
            "appended_at": "ISO8601"
        }

    Raises:
        400: Missing required fields (message or phase).
        404: run_id doesn't exist.
        500: Database update failed.
    """
    message = payload.get("message", "").strip()
    phase = payload.get("phase", "").strip()

    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")
    if not phase:
        raise HTTPException(status_code=400, detail="phase cannot be empty")

    config = request.app.state.config

    try:
        from consistency_checker.index.assertion_store import AssertionStore

        store = AssertionStore(config.db_path)
        store.migrate()
        try:
            conn = store._conn

            row = conn.execute(
                "SELECT run_id, status, message_log FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()

            if row is None:
                raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")

            _current_run_id, _status, existing_log = row

            now = datetime.now().isoformat(timespec="microseconds")
            new_entry = json.dumps({"message": message, "phase": phase, "timestamp": now})

            updated_log = (existing_log or "") + new_entry + "\n"

            conn.execute(
                "UPDATE runs SET message_log = ? WHERE run_id = ?",
                (updated_log, run_id),
            )
            conn.commit()

            return {
                "run_id": run_id,
                "message": message,
                "phase": phase,
                "appended_at": now,
            }
        finally:
            store.close()
    except HTTPException:
        raise
    except Exception as e:
        _log.error("Failed to append message to run %s: %s", run_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to append message: {e}") from e


@router.patch("/{run_id}", status_code=200)
def update_run_status(
    request: Request,
    run_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Update run status (e.g., mark as completed or failed).

    Request body:
        {
            "status": "completed" | "failed",
            "completed_at": "ISO8601"  (optional)
        }

    Returns:
        {
            "run_id": "...",
            "corpus_id": "...",
            "started_at": "ISO8601",
            "completed_at": "ISO8601 or null",
            "status": "...",
            "message_count": N
        }

    Raises:
        400: Invalid status value.
        404: run_id doesn't exist.
        500: Database update failed.
    """
    status = payload.get("status", "").strip()
    completed_at_str = payload.get("completed_at")

    valid_statuses = ("completed", "failed", "in_progress")
    if status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{status}'. Must be one of: {', '.join(valid_statuses)}",
        )

    config = request.app.state.config

    try:
        from consistency_checker.index.assertion_store import AssertionStore

        store = AssertionStore(config.db_path)
        store.migrate()
        try:
            conn = store._conn

            row = conn.execute(
                """
                SELECT run_id, corpus_id, started_at, completed_at, status, message_log
                FROM runs WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

            if row is None:
                raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")

            now = datetime.now().isoformat(timespec="microseconds")
            completed_at = completed_at_str or (now if status in ("completed", "failed") else None)

            conn.execute(
                "UPDATE runs SET status = ?, completed_at = ? WHERE run_id = ?",
                (status, completed_at, run_id),
            )
            conn.commit()

            updated_row = conn.execute(
                """
                SELECT run_id, corpus_id, started_at, completed_at, status, message_log
                FROM runs WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

            run = Run.from_row(updated_row)
            message_count = len(_parse_message_log(run.message_log))

            return {
                "run_id": run.run_id,
                "corpus_id": run.corpus_id,
                "started_at": run.started_at.isoformat(),
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "status": run.status,
                "message_count": message_count,
            }
        finally:
            store.close()
    except HTTPException:
        raise
    except Exception as e:
        _log.error("Failed to update run %s: %s", run_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to update run: {e}") from e


@corpora_runs_router.post("/{corpus_id}/runs", status_code=201)
def start_run(
    request: Request,
    corpus_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Start a new processing run for a corpus.

    Request body (all optional):
        {
            "judge_provider": "moonshot" | "anthropic"
        }

    Returns:
        {
            "run_id": "...",
            "corpus_id": "...",
            "started_at": "ISO8601",
            "status": "in_progress"
        }

    Raises:
        400: Invalid judge_provider.
        404: corpus_id doesn't exist.
        500: Database error.
    """
    judge_provider = payload.get("judge_provider")

    if judge_provider is not None and judge_provider not in ("moonshot", "anthropic"):
        raise HTTPException(
            status_code=400,
            detail="judge_provider must be 'moonshot' or 'anthropic'",
        )

    config = request.app.state.config

    try:
        from consistency_checker.index.assertion_store import AssertionStore

        store = AssertionStore(config.db_path)
        store.migrate()
        try:
            conn = store._conn

            # Validate corpus exists
            row = conn.execute(
                "SELECT corpus_id FROM corpora WHERE corpus_id = ?",
                (corpus_id,),
            ).fetchone()

            if row is None:
                raise HTTPException(status_code=404, detail=f"corpus_id '{corpus_id}' not found")

            # Generate unique run_id
            now = datetime.now().isoformat(timespec="microseconds")
            run_id = str(uuid.uuid4())

            conn.execute(
                """
                INSERT INTO runs (run_id, corpus_id, started_at, completed_at, status, message_log)
                VALUES (?, ?, ?, NULL, 'in_progress', NULL)
                """,
                (run_id, corpus_id, now),
            )
            conn.commit()

            return {
                "run_id": run_id,
                "corpus_id": corpus_id,
                "started_at": now,
                "status": "in_progress",
            }
        finally:
            store.close()
    except HTTPException:
        raise
    except sqlite3.IntegrityError as e:
        _log.error("Database integrity error starting run for corpus %s: %s", corpus_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to start run: {e}") from e
    except Exception as e:
        _log.error("Failed to start run for corpus %s: %s", corpus_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to start run: {e}") from e
