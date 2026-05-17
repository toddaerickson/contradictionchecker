"""Corpus management API endpoints.

Provides REST endpoints for creating, listing, and retrieving corpus metadata.
Each corpus is a collection of documents analyzed together.
"""

from __future__ import annotations

import re
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from consistency_checker.logging_setup import get_logger
from consistency_checker.models.ui import Corpus

_log = get_logger(__name__)

router = APIRouter(prefix="/api/corpora", tags=["corpora"])

# Valid characters for corpus_id: alphanumeric + hyphen
_VALID_CORPUS_ID_CHARS = re.compile(r"^[a-z0-9\-]+$")


def _make_corpus_id(name: str) -> str:
    """Generate a deterministic corpus_id slug from corpus_name.

    Converts to lowercase, replaces spaces/underscores with hyphens, removes
    invalid chars, and appends a 4-char random suffix to handle collisions.

    Args:
        name: The human-readable corpus name (e.g., "Q1 Audit").

    Returns:
        A corpus_id slug (e.g., "q1-audit-x7kp").

    Raises:
        ValueError: If the slug is empty after sanitization.
    """
    # Convert to lowercase and replace spaces/underscores with hyphens
    slug = name.lower().replace(" ", "-").replace("_", "-")

    # Remove invalid characters (keep only alphanumeric and hyphens)
    slug = re.sub(r"[^a-z0-9\-]", "", slug)

    # Remove leading/trailing hyphens and collapse consecutive hyphens
    slug = re.sub(r"-+", "-", slug).strip("-")

    if not slug:
        raise ValueError(f"corpus_name '{name}' produces empty slug after sanitization")

    # Append 4-char random suffix to handle collisions
    suffix = secrets.token_hex(2)  # 2 bytes = 4 hex chars
    return f"{slug}-{suffix}"


@router.post("", status_code=201)
def create_corpus(
    request: Request,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Create a new corpus.

    Request body:
        {
            "corpus_name": "string",
            "judge_provider": "moonshot" | "anthropic"
        }

    Returns:
        {
            "corpus_id": "...",
            "corpus_name": "...",
            "corpus_path": "...",
            "judge_provider": "...",
            "created_at": "ISO8601",
            "updated_at": "ISO8601"
        }

    Raises:
        400: Invalid corpus_name (empty, contains filesystem-invalid chars).
        409: corpus_name already exists (unique constraint).
        500: Filesystem or database I/O error.
    """
    corpus_name = payload.get("corpus_name", "").strip()
    judge_provider = payload.get("judge_provider", "moonshot").strip()

    # Validate corpus_name
    if not corpus_name:
        raise HTTPException(status_code=400, detail="corpus_name cannot be empty")

    # Check for invalid filesystem characters
    invalid_chars = set(r'\/:"*?<>|')
    if any(c in corpus_name for c in invalid_chars):
        raise HTTPException(
            status_code=400,
            detail=f"corpus_name contains invalid characters: {', '.join(invalid_chars)}",
        )

    # Validate judge_provider
    if judge_provider not in ("moonshot", "anthropic"):
        raise HTTPException(
            status_code=400,
            detail="judge_provider must be 'moonshot' or 'anthropic'",
        )

    # Generate corpus_id
    try:
        corpus_id = _make_corpus_id(corpus_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Create directory structure
    config = request.app.state.config
    corpus_path = config.data_dir / "corpora" / corpus_id
    documents_path = corpus_path / "documents"

    try:
        documents_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _log.error("Failed to create corpus directory %s: %s", corpus_path, e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create corpus directory: {e}",
        ) from e

    # Store in database
    try:
        from consistency_checker.index.assertion_store import AssertionStore

        store = AssertionStore(config.db_path)
        store.migrate()
        try:
            conn = store._conn
            now = datetime.now().isoformat(timespec="microseconds")
            try:
                conn.execute(
                    """
                    INSERT INTO corpora (corpus_id, corpus_name, corpus_path, judge_provider, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (corpus_id, corpus_name, str(corpus_path), judge_provider, now, now),
                )
                conn.commit()
            except sqlite3.IntegrityError as e:
                import shutil

                shutil.rmtree(corpus_path, ignore_errors=True)
                _log.error("Database integrity error creating corpus: %s", e)
                if "corpus_name" in str(e):
                    raise HTTPException(
                        status_code=409,
                        detail=f"corpus_name '{corpus_name}' already exists",
                    ) from e
                raise HTTPException(status_code=409, detail="Corpus already exists") from e
            except Exception:
                import shutil

                shutil.rmtree(corpus_path, ignore_errors=True)
                raise
        finally:
            store.close()
    except HTTPException:
        raise
    except Exception as e:
        _log.error("Failed to store corpus in database: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store corpus: {e}",
        ) from e

    return {
        "corpus_id": corpus_id,
        "corpus_name": corpus_name,
        "corpus_path": str(corpus_path),
        "judge_provider": judge_provider,
        "created_at": now,
        "updated_at": now,
    }


@router.get("", status_code=200)
def list_corpora(request: Request) -> dict[str, Any]:
    """List all corpora.

    Returns:
        {
            "corpora": [
                {
                    "corpus_id": "...",
                    "corpus_name": "...",
                    "corpus_path": "...",
                    "judge_provider": "...",
                    "created_at": "ISO8601",
                    "updated_at": "ISO8601"
                },
                ...
            ]
        }
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
                SELECT corpus_id, corpus_name, corpus_path, judge_provider, created_at, updated_at
                FROM corpora
                ORDER BY created_at DESC
                """
            ).fetchall()

            corpora = []
            for row in rows:
                corpus = Corpus.from_row(row)
                corpora.append(
                    {
                        "corpus_id": corpus.corpus_id,
                        "corpus_name": corpus.corpus_name,
                        "corpus_path": corpus.corpus_path,
                        "judge_provider": corpus.judge_provider,
                        "created_at": corpus.created_at.isoformat(),
                        "updated_at": corpus.updated_at.isoformat(),
                    }
                )

            return {"corpora": corpora}
        finally:
            store.close()
    except Exception as e:
        _log.error("Failed to list corpora: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to list corpora: {e}") from e


@router.get("/{corpus_id}", status_code=200)
def get_corpus(request: Request, corpus_id: str) -> dict[str, Any]:
    """Get a specific corpus.

    Returns:
        {
            "corpus_id": "...",
            "corpus_name": "...",
            "corpus_path": "...",
            "judge_provider": "...",
            "created_at": "ISO8601",
            "updated_at": "ISO8601",
            "document_count": 42
        }

    Raises:
        404: corpus_id doesn't exist.
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
                SELECT corpus_id, corpus_name, corpus_path, judge_provider, created_at, updated_at
                FROM corpora
                WHERE corpus_id = ?
                """,
                (corpus_id,),
            ).fetchone()

            if row is None:
                raise HTTPException(status_code=404, detail=f"corpus_id '{corpus_id}' not found")

            corpus = Corpus.from_row(row)

            # Count documents in corpus directory
            corpus_dir = Path(corpus.corpus_path)
            documents_dir = corpus_dir / "documents"
            document_count = 0
            if documents_dir.exists():
                document_count = sum(1 for _ in documents_dir.iterdir() if _.is_file())

            return {
                "corpus_id": corpus.corpus_id,
                "corpus_name": corpus.corpus_name,
                "corpus_path": corpus.corpus_path,
                "judge_provider": corpus.judge_provider,
                "created_at": corpus.created_at.isoformat(),
                "updated_at": corpus.updated_at.isoformat(),
                "document_count": document_count,
            }
        finally:
            store.close()
    except HTTPException:
        raise
    except Exception as e:
        _log.error("Failed to get corpus %s: %s", corpus_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to get corpus: {e}") from e


@router.get("/{corpus_id}/documents", status_code=200)
def list_documents(request: Request, corpus_id: str) -> dict[str, Any]:
    """List documents in a corpus.

    Returns:
        {
            "corpus_id": "...",
            "documents": [
                {
                    "filename": "...",
                    "size_bytes": 12345,
                    "uploaded_at": "ISO8601"
                },
                ...
            ]
        }

    Raises:
        404: corpus_id doesn't exist.
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
                SELECT corpus_path FROM corpora WHERE corpus_id = ?
                """,
                (corpus_id,),
            ).fetchone()

            if row is None:
                raise HTTPException(status_code=404, detail=f"corpus_id '{corpus_id}' not found")

            corpus_path = Path(row[0])
            documents_dir = corpus_path / "documents"

            # List files in documents directory
            documents = []
            if documents_dir.exists():
                entries = [(p, p.stat()) for p in documents_dir.iterdir() if p.is_file()]
                entries.sort(key=lambda ps: ps[1].st_mtime, reverse=True)
                documents = [
                    {
                        "filename": p.name,
                        "size_bytes": s.st_size,
                        "uploaded_at": datetime.fromtimestamp(s.st_mtime).isoformat(),
                    }
                    for p, s in entries
                ]

            return {
                "corpus_id": corpus_id,
                "documents": documents,
            }
        finally:
            store.close()
    except HTTPException:
        raise
    except Exception as e:
        _log.error("Failed to list documents for corpus %s: %s", corpus_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}") from e


# Route prefix for findings (not /api/corpora since it's per-finding, not per-corpus)
findings_router = APIRouter(prefix="/api/findings", tags=["findings"])


@findings_router.post("/{finding_id}/verdict", status_code=200)
def set_finding_verdict(
    request: Request,
    finding_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Set user verdict for a finding.

    Request body:
        {
            "user_verdict": "confirmed" | "false_positive" | "dismissed" | "pending" | null
        }

    Returns:
        {
            "finding_id": "...",
            "user_verdict": "...",
            "is_multi_party": bool,
            "updated_at": "ISO8601"
        }

    Behavior:
        - Look up finding_id in either findings or multi_party_findings table
        - Validate user_verdict is one of: confirmed, false_positive, dismissed, pending, or null
        - Update the user_verdict column in the appropriate table
        - Return the updated Verdict object with current timestamp
        - Return 400 Bad Request if user_verdict is not a valid value
        - Return 404 Not Found if finding_id doesn't exist
        - Return 500 Internal Server Error if database update fails

    Raises:
        400: Invalid user_verdict value.
        404: finding_id doesn't exist in either table.
        500: Database update failed.
    """
    config = request.app.state.config

    # Extract and validate user_verdict from payload
    user_verdict = payload.get("user_verdict")

    # Validate: must be one of the allowed values or None
    valid_verdicts = ("confirmed", "false_positive", "dismissed", "pending")
    if user_verdict is not None and user_verdict not in valid_verdicts:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid user_verdict '{user_verdict}'. Must be one of: {', '.join(valid_verdicts)}, or null.",
        )

    try:
        from consistency_checker.index.assertion_store import AssertionStore

        store = AssertionStore(config.db_path)
        store.migrate()
        try:
            conn = store._conn

            # Try to find in findings table first
            row = conn.execute(
                """
                SELECT finding_id, user_verdict FROM findings WHERE finding_id = ? LIMIT 1
                """,
                (finding_id,),
            ).fetchone()

            is_multi_party = False

            if row is None:
                # Try multi_party_findings table
                row = conn.execute(
                    """
                    SELECT finding_id, user_verdict FROM multi_party_findings WHERE finding_id = ? LIMIT 1
                    """,
                    (finding_id,),
                ).fetchone()
                if row is not None:
                    is_multi_party = True

            # If not found in either table, return 404
            if row is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"finding_id '{finding_id}' not found",
                )

            # Update the verdict in the appropriate table
            now = datetime.now().isoformat(timespec="microseconds")
            if is_multi_party:
                conn.execute(
                    """
                    UPDATE multi_party_findings SET user_verdict = ? WHERE finding_id = ?
                    """,
                    (user_verdict, finding_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE findings SET user_verdict = ? WHERE finding_id = ?
                    """,
                    (user_verdict, finding_id),
                )
            conn.commit()

            # Return the updated verdict
            return {
                "finding_id": finding_id,
                "user_verdict": user_verdict,
                "is_multi_party": is_multi_party,
                "updated_at": now,
            }
        finally:
            store.close()
    except HTTPException:
        raise
    except Exception as e:
        _log.error("Failed to set verdict for finding %s: %s", finding_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to set verdict: {e}",
        ) from e
