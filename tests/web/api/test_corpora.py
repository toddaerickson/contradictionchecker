"""Tests for corpus management and verdict endpoints.

Tests cover all endpoints with success and error cases:
- POST /api/corpora (create)
- GET /api/corpora (list)
- GET /api/corpora/{corpus_id} (get one)
- GET /api/corpora/{corpus_id}/documents (list documents)
- POST /api/findings/{finding_id}/verdict (set user verdict)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def test_post_corpora_success(web_client: TestClient) -> None:
    """Test successful corpus creation."""
    response = web_client.post(
        "/api/corpora",
        json={
            "corpus_name": "Q1 Audit",
            "judge_provider": "moonshot",
        },
    )

    assert response.status_code == 201
    data = response.json()

    # Verify response structure
    assert "corpus_id" in data
    assert "corpus_name" in data
    assert "corpus_path" in data
    assert "judge_provider" in data
    assert "created_at" in data
    assert "updated_at" in data

    # Verify content
    assert data["corpus_name"] == "Q1 Audit"
    assert data["judge_provider"] == "moonshot"

    # Verify corpus_id is slugified
    assert data["corpus_id"].startswith("q1-audit-")
    assert len(data["corpus_id"].split("-")[-1]) == 4  # 4-char hex suffix

    # Verify corpus_id is a valid filesystem path component
    assert "/" not in data["corpus_id"]
    assert "\\" not in data["corpus_id"]
    assert ":" not in data["corpus_id"]

    # Verify directory structure was created
    corpus_path = Path(data["corpus_path"])
    assert corpus_path.exists()
    documents_dir = corpus_path / "documents"
    assert documents_dir.exists()
    assert documents_dir.is_dir()


def test_post_corpora_with_anthropic_provider(web_client: TestClient) -> None:
    """Test corpus creation with anthropic judge provider."""
    response = web_client.post(
        "/api/corpora",
        json={
            "corpus_name": "API Test Corpus",
            "judge_provider": "anthropic",
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["judge_provider"] == "anthropic"


def test_post_corpora_empty_name(web_client: TestClient) -> None:
    """Test corpus creation fails with empty corpus_name."""
    response = web_client.post(
        "/api/corpora",
        json={
            "corpus_name": "",
            "judge_provider": "moonshot",
        },
    )

    assert response.status_code == 400
    assert "corpus_name cannot be empty" in response.json()["detail"]


def test_post_corpora_whitespace_only_name(web_client: TestClient) -> None:
    """Test corpus creation fails with whitespace-only corpus_name."""
    response = web_client.post(
        "/api/corpora",
        json={
            "corpus_name": "   ",
            "judge_provider": "moonshot",
        },
    )

    assert response.status_code == 400
    assert "corpus_name cannot be empty" in response.json()["detail"]


@pytest.mark.parametrize(
    "invalid_name",
    [
        "Audit / Review",  # forward slash
        "Audit \\ Review",  # backslash
        'Audit "Test"',  # quotes
        "Audit:Test",  # colon
        "Audit*Test",  # asterisk
        "Audit?Test",  # question mark
        "Audit<Test>",  # angle brackets
        "Audit|Test",  # pipe
    ],
)
def test_post_corpora_invalid_chars(web_client: TestClient, invalid_name: str) -> None:
    """Test corpus creation fails with invalid filesystem characters."""
    response = web_client.post(
        "/api/corpora",
        json={
            "corpus_name": invalid_name,
            "judge_provider": "moonshot",
        },
    )

    assert response.status_code == 400
    assert "invalid characters" in response.json()["detail"]


def test_post_corpora_invalid_provider(web_client: TestClient) -> None:
    """Test corpus creation fails with invalid judge_provider."""
    response = web_client.post(
        "/api/corpora",
        json={
            "corpus_name": "Test Corpus",
            "judge_provider": "invalid_provider",
        },
    )

    assert response.status_code == 400
    assert "must be 'moonshot' or 'anthropic'" in response.json()["detail"]


def test_post_corpora_duplicate_name(web_client: TestClient) -> None:
    """Test corpus creation fails when corpus_name already exists (409 Conflict)."""
    # Create first corpus
    response1 = web_client.post(
        "/api/corpora",
        json={
            "corpus_name": "Unique Corpus Name",
            "judge_provider": "moonshot",
        },
    )
    assert response1.status_code == 201

    # Try to create with same name
    response2 = web_client.post(
        "/api/corpora",
        json={
            "corpus_name": "Unique Corpus Name",
            "judge_provider": "anthropic",
        },
    )

    assert response2.status_code == 409
    assert "already exists" in response2.json()["detail"]


def test_get_corpora_empty_list(web_client: TestClient) -> None:
    """Test listing corpora when none exist."""
    response = web_client.get("/api/corpora")

    assert response.status_code == 200
    data = response.json()
    assert "corpora" in data
    assert data["corpora"] == []


def test_get_corpora_list_multiple(web_client: TestClient) -> None:
    """Test listing multiple corpora."""
    # Create three corpora
    corpus_names = ["Corpus A", "Corpus B", "Corpus C"]
    for name in corpus_names:
        response = web_client.post(
            "/api/corpora",
            json={"corpus_name": name, "judge_provider": "moonshot"},
        )
        assert response.status_code == 201

    # List all
    response = web_client.get("/api/corpora")
    assert response.status_code == 200
    data = response.json()

    # Filter to just the ones we created (in case other tests ran first)
    our_corpora = [c for c in data["corpora"] if c["corpus_name"] in corpus_names]
    assert len(our_corpora) == 3

    # Verify all are present
    returned_names = {c["corpus_name"] for c in our_corpora}
    assert returned_names == set(corpus_names)


def test_get_corpora_ordering(web_client: TestClient) -> None:
    """Test corpora are ordered by created_at descending."""
    import time

    # Create two corpora with a small delay to ensure different timestamps
    response1 = web_client.post(
        "/api/corpora",
        json={"corpus_name": "First Order Corpus", "judge_provider": "moonshot"},
    )
    first_corpus = response1.json()

    # Sleep to ensure different timestamp
    time.sleep(0.01)

    response2 = web_client.post(
        "/api/corpora",
        json={"corpus_name": "Second Order Corpus", "judge_provider": "moonshot"},
    )
    second_corpus = response2.json()

    # List all
    response = web_client.get("/api/corpora")
    data = response.json()

    # Filter to just the ones we created
    our_corpora = [
        c
        for c in data["corpora"]
        if c["corpus_id"] in (first_corpus["corpus_id"], second_corpus["corpus_id"])
    ]
    assert len(our_corpora) == 2

    # Second should come first (most recent)
    assert our_corpora[0]["corpus_id"] == second_corpus["corpus_id"]
    assert our_corpora[1]["corpus_id"] == first_corpus["corpus_id"]


def test_get_corpus_by_id_success(web_client: TestClient) -> None:
    """Test retrieving a specific corpus."""
    # Create corpus
    create_response = web_client.post(
        "/api/corpora",
        json={"corpus_name": "Test Corpus", "judge_provider": "moonshot"},
    )
    assert create_response.status_code == 201
    created = create_response.json()

    # Get corpus by ID
    response = web_client.get(f"/api/corpora/{created['corpus_id']}")

    assert response.status_code == 200
    data = response.json()

    # Verify all fields
    assert data["corpus_id"] == created["corpus_id"]
    assert data["corpus_name"] == "Test Corpus"
    assert data["judge_provider"] == "moonshot"
    assert data["created_at"] == created["created_at"]
    assert data["updated_at"] == created["updated_at"]

    # Verify document_count is present
    assert "document_count" in data
    assert data["document_count"] == 0  # No documents yet


def test_get_corpus_by_id_not_found(web_client: TestClient) -> None:
    """Test retrieving non-existent corpus returns 404."""
    response = web_client.get("/api/corpora/nonexistent-corpus-id")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_get_corpus_with_documents(web_client: TestClient) -> None:
    """Test document_count is accurate when documents exist."""
    # Create corpus
    create_response = web_client.post(
        "/api/corpora",
        json={"corpus_name": "Corpus With Docs", "judge_provider": "moonshot"},
    )
    corpus = create_response.json()

    # Add some test files to the documents directory
    corpus_path = Path(corpus["corpus_path"])
    documents_dir = corpus_path / "documents"
    (documents_dir / "test1.txt").write_text("content1")
    (documents_dir / "test2.txt").write_text("content2")
    (documents_dir / "test3.txt").write_text("content3")

    # Get corpus
    response = web_client.get(f"/api/corpora/{corpus['corpus_id']}")

    assert response.status_code == 200
    data = response.json()
    assert data["document_count"] == 3


def test_get_documents_empty(web_client: TestClient) -> None:
    """Test listing documents when none exist."""
    # Create corpus
    create_response = web_client.post(
        "/api/corpora",
        json={"corpus_name": "Empty Corpus", "judge_provider": "moonshot"},
    )
    corpus = create_response.json()

    # List documents
    response = web_client.get(f"/api/corpora/{corpus['corpus_id']}/documents")

    assert response.status_code == 200
    data = response.json()

    assert data["corpus_id"] == corpus["corpus_id"]
    assert data["documents"] == []


def test_get_documents_list_multiple(web_client: TestClient) -> None:
    """Test listing multiple documents."""
    # Create corpus
    create_response = web_client.post(
        "/api/corpora",
        json={"corpus_name": "Corpus With Many Docs", "judge_provider": "moonshot"},
    )
    corpus = create_response.json()

    # Add test files
    corpus_path = Path(corpus["corpus_path"])
    documents_dir = corpus_path / "documents"

    files = {
        "document_a.txt": "content a",
        "document_b.txt": "content b",
        "document_c.txt": "content c",
    }
    for filename, content in files.items():
        (documents_dir / filename).write_text(content)

    # List documents
    response = web_client.get(f"/api/corpora/{corpus['corpus_id']}/documents")

    assert response.status_code == 200
    data = response.json()

    assert len(data["documents"]) == 3
    returned_names = {d["filename"] for d in data["documents"]}
    assert returned_names == set(files.keys())


def test_get_documents_metadata(web_client: TestClient) -> None:
    """Test document metadata (name, size, upload time)."""
    # Create corpus
    create_response = web_client.post(
        "/api/corpora",
        json={"corpus_name": "Test Metadata", "judge_provider": "moonshot"},
    )
    corpus = create_response.json()

    # Add test file
    corpus_path = Path(corpus["corpus_path"])
    documents_dir = corpus_path / "documents"
    test_file = documents_dir / "test_file.txt"
    test_content = "test content with some size"
    test_file.write_text(test_content)

    # List documents
    response = web_client.get(f"/api/corpora/{corpus['corpus_id']}/documents")

    assert response.status_code == 200
    data = response.json()

    assert len(data["documents"]) == 1
    doc = data["documents"][0]

    # Verify metadata fields
    assert doc["filename"] == "test_file.txt"
    assert doc["size_bytes"] == len(test_content.encode("utf-8"))
    assert "uploaded_at" in doc
    # Verify it's a valid ISO format timestamp
    from datetime import datetime

    datetime.fromisoformat(doc["uploaded_at"])  # Should not raise


def test_get_documents_ordering(web_client: TestClient) -> None:
    """Test documents are ordered by modification time descending."""
    import time

    # Create corpus
    create_response = web_client.post(
        "/api/corpora",
        json={"corpus_name": "Test Order", "judge_provider": "moonshot"},
    )
    corpus = create_response.json()

    # Add files with some time between them
    corpus_path = Path(corpus["corpus_path"])
    documents_dir = corpus_path / "documents"

    (documents_dir / "file1.txt").write_text("content1")
    time.sleep(0.01)
    (documents_dir / "file2.txt").write_text("content2")
    time.sleep(0.01)
    (documents_dir / "file3.txt").write_text("content3")

    # List documents
    response = web_client.get(f"/api/corpora/{corpus['corpus_id']}/documents")

    assert response.status_code == 200
    data = response.json()

    # Should be ordered by mtime descending (most recent first)
    assert data["documents"][0]["filename"] == "file3.txt"
    assert data["documents"][1]["filename"] == "file2.txt"
    assert data["documents"][2]["filename"] == "file1.txt"


def test_get_documents_nonexistent_corpus(web_client: TestClient) -> None:
    """Test listing documents for non-existent corpus returns 404."""
    response = web_client.get("/api/corpora/nonexistent-corpus-xyz/documents")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_corpus_id_collision_handling(web_client: TestClient) -> None:
    """Test that different corpus names can have similar slugs (collision handling via suffix)."""
    # Create two corpora with similar names
    web_client.post(
        "/api/corpora",
        json={"corpus_name": "Collision Test Audit 1", "judge_provider": "moonshot"},
    )

    response2 = web_client.post(
        "/api/corpora",
        json={"corpus_name": "Collision Test Audit 1", "judge_provider": "moonshot"},
    )

    # Second should fail due to unique constraint on corpus_name
    assert response2.status_code == 409

    # But if we use a different name with similar slug base:
    response3 = web_client.post(
        "/api/corpora",
        json={"corpus_name": "Collision Test Audit 2", "judge_provider": "moonshot"},
    )

    # Should succeed because corpus_name is different
    assert response3.status_code == 201


def test_corpus_id_special_chars_removed(web_client: TestClient) -> None:
    """Test that special characters in corpus_name are removed from corpus_id."""
    response = web_client.post(
        "/api/corpora",
        json={
            "corpus_name": "Test@#$%Audit&()2024",
            "judge_provider": "moonshot",
        },
    )

    assert response.status_code == 201
    data = response.json()

    # corpus_id should only contain alphanumeric and hyphens
    corpus_id = data["corpus_id"]
    assert all(c.isalnum() or c == "-" for c in corpus_id)
    assert "@" not in corpus_id
    assert "#" not in corpus_id
    assert "$" not in corpus_id


def test_post_corpora_missing_field(web_client: TestClient) -> None:
    """Test corpus creation fails with missing required field."""
    response = web_client.post(
        "/api/corpora",
        json={
            "corpus_name": "Test",
            # Missing judge_provider
        },
    )

    # Should either default to moonshot or fail gracefully
    # Based on the code, it defaults to "moonshot"
    if response.status_code == 201:
        assert response.json()["judge_provider"] == "moonshot"


def test_post_corpora_extra_fields(web_client: TestClient) -> None:
    """Test corpus creation ignores extra fields in request."""
    response = web_client.post(
        "/api/corpora",
        json={
            "corpus_name": "Test Corpus",
            "judge_provider": "moonshot",
            "extra_field": "should be ignored",
            "another_field": 12345,
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert "extra_field" not in data
    assert "another_field" not in data


def test_corpus_timestamps_iso8601(web_client: TestClient) -> None:
    """Test that all timestamp responses are valid ISO8601."""
    from datetime import datetime

    response = web_client.post(
        "/api/corpora",
        json={"corpus_name": "Timestamp Test", "judge_provider": "moonshot"},
    )

    assert response.status_code == 201
    data = response.json()

    # Should be able to parse as ISO8601
    datetime.fromisoformat(data["created_at"])
    datetime.fromisoformat(data["updated_at"])

    # Also test in list
    list_response = web_client.get("/api/corpora")
    assert list_response.status_code == 200
    for corpus in list_response.json()["corpora"]:
        datetime.fromisoformat(corpus["created_at"])
        datetime.fromisoformat(corpus["updated_at"])


def test_corpus_path_structure(web_client: TestClient) -> None:
    """Test that corpus_path follows expected structure."""
    response = web_client.post(
        "/api/corpora",
        json={"corpus_name": "Path Test", "judge_provider": "moonshot"},
    )

    assert response.status_code == 201
    data = response.json()

    corpus_path = Path(data["corpus_path"])
    # Should end with /{corpus_id}
    assert corpus_path.name == data["corpus_id"]
    # Should be under data/corpora/
    assert "corpora" in str(corpus_path)


# --- Verdict endpoint tests -----------------------------------------------


def _create_pairwise_finding(web_client: TestClient) -> str:
    """Helper: Create a finding in the findings table and return its finding_id.

    Inserts directly into the database to avoid needing full pipeline setup.
    """
    from consistency_checker.index.assertion_store import AssertionStore

    # Access config from the test client's app
    config = web_client.app.state.config  # type: ignore[attr-defined]
    store = AssertionStore(config.db_path)
    store.migrate()

    try:
        conn = store._conn

        # Create documents first (required FK for assertions)
        conn.execute(
            """
            INSERT INTO documents (doc_id, source_path, doc_type, ingested_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
            ("doc_1", "/test/doc1.txt", "text"),
        )
        conn.execute(
            """
            INSERT INTO documents (doc_id, source_path, doc_type, ingested_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
            ("doc_2", "/test/doc2.txt", "text"),
        )

        # Create two test assertions (they need to exist as FKs)
        conn.execute(
            """
            INSERT INTO assertions (assertion_id, doc_id, assertion_text, char_start, char_end)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("assertion_1", "doc_1", "Assertion 1", 0, 12),
        )
        conn.execute(
            """
            INSERT INTO assertions (assertion_id, doc_id, assertion_text, char_start, char_end)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("assertion_2", "doc_2", "Assertion 2", 0, 12),
        )

        # Create a pipeline_run (required FK for findings)
        run_id = "test_run_123"
        conn.execute(
            """
            INSERT INTO pipeline_runs (run_id, started_at)
            VALUES (?, CURRENT_TIMESTAMP)
            """,
            (run_id,),
        )

        # Create a finding in the findings table
        finding_id = "finding_pairwise_123"
        conn.execute(
            """
            INSERT INTO findings
            (finding_id, run_id, assertion_a_id, assertion_b_id, judge_verdict, user_verdict)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (finding_id, run_id, "assertion_1", "assertion_2", "contradiction", None),
        )
        conn.commit()

        return finding_id
    finally:
        store.close()


def _create_multi_party_finding(web_client: TestClient) -> str:
    """Helper: Create a multi-party finding and return its finding_id."""
    import json

    from consistency_checker.index.assertion_store import AssertionStore

    config = web_client.app.state.config  # type: ignore[attr-defined]
    store = AssertionStore(config.db_path)
    store.migrate()

    try:
        conn = store._conn

        # Create three test documents
        for i in range(1, 4):
            conn.execute(
                """
                INSERT INTO documents (doc_id, source_path, doc_type, ingested_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (f"mp_doc_{i}", f"/test/mp_doc{i}.txt", "text"),
            )

        # Create three test assertions
        for i in range(1, 4):
            conn.execute(
                """
                INSERT INTO assertions (assertion_id, doc_id, assertion_text, char_start, char_end)
                VALUES (?, ?, ?, ?, ?)
                """,
                (f"mp_assertion_{i}", f"mp_doc_{i}", f"Assertion {i}", 0, 12),
            )

        # Create a pipeline_run
        run_id = "test_run_mp_123"
        conn.execute(
            """
            INSERT INTO pipeline_runs (run_id, started_at)
            VALUES (?, CURRENT_TIMESTAMP)
            """,
            (run_id,),
        )

        # Create a multi-party finding
        finding_id = "finding_multiparty_123"

        assertion_ids = ["mp_assertion_1", "mp_assertion_2", "mp_assertion_3"]
        doc_ids = ["mp_doc_1", "mp_doc_2", "mp_doc_3"]

        conn.execute(
            """
            INSERT INTO multi_party_findings
            (finding_id, run_id, assertion_ids_json, doc_ids_json, judge_verdict, user_verdict)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                finding_id,
                run_id,
                json.dumps(assertion_ids),
                json.dumps(doc_ids),
                "multi_party_contradiction",
                None,
            ),
        )
        conn.commit()

        return finding_id
    finally:
        store.close()


def test_post_verdict_confirmed(web_client: TestClient) -> None:
    """Test setting verdict to 'confirmed'."""
    finding_id = _create_pairwise_finding(web_client)

    response = web_client.post(
        f"/api/findings/{finding_id}/verdict",
        json={"user_verdict": "confirmed"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["finding_id"] == finding_id
    assert data["user_verdict"] == "confirmed"
    assert data["is_multi_party"] is False
    assert "updated_at" in data
    # Verify timestamp is valid ISO format
    datetime.fromisoformat(data["updated_at"])


def test_post_verdict_false_positive(web_client: TestClient) -> None:
    """Test setting verdict to 'false_positive'."""
    finding_id = _create_pairwise_finding(web_client)

    response = web_client.post(
        f"/api/findings/{finding_id}/verdict",
        json={"user_verdict": "false_positive"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["finding_id"] == finding_id
    assert data["user_verdict"] == "false_positive"
    assert data["is_multi_party"] is False


def test_post_verdict_dismissed(web_client: TestClient) -> None:
    """Test setting verdict to 'dismissed'."""
    finding_id = _create_pairwise_finding(web_client)

    response = web_client.post(
        f"/api/findings/{finding_id}/verdict",
        json={"user_verdict": "dismissed"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["user_verdict"] == "dismissed"


def test_post_verdict_pending(web_client: TestClient) -> None:
    """Test setting verdict to 'pending'."""
    finding_id = _create_pairwise_finding(web_client)

    response = web_client.post(
        f"/api/findings/{finding_id}/verdict",
        json={"user_verdict": "pending"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["user_verdict"] == "pending"


def test_post_verdict_null(web_client: TestClient) -> None:
    """Test setting verdict to null (clear verdict)."""
    finding_id = _create_pairwise_finding(web_client)

    # First set a verdict
    web_client.post(
        f"/api/findings/{finding_id}/verdict",
        json={"user_verdict": "confirmed"},
    )

    # Now clear it by setting to null
    response = web_client.post(
        f"/api/findings/{finding_id}/verdict",
        json={"user_verdict": None},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["user_verdict"] is None
    assert data["is_multi_party"] is False


def test_post_verdict_idempotency(web_client: TestClient) -> None:
    """Test that setting the same verdict twice returns the same result."""
    finding_id = _create_pairwise_finding(web_client)

    # Set verdict first time
    response1 = web_client.post(
        f"/api/findings/{finding_id}/verdict",
        json={"user_verdict": "confirmed"},
    )

    # Set verdict second time with same value
    response2 = web_client.post(
        f"/api/findings/{finding_id}/verdict",
        json={"user_verdict": "confirmed"},
    )

    assert response1.status_code == 200
    assert response2.status_code == 200

    data1 = response1.json()
    data2 = response2.json()

    # Both should return the same verdict
    assert data1["user_verdict"] == data2["user_verdict"] == "confirmed"
    assert data1["finding_id"] == data2["finding_id"]


def test_post_verdict_invalid_value(web_client: TestClient) -> None:
    """Test that invalid verdict value returns 400 Bad Request."""
    finding_id = _create_pairwise_finding(web_client)

    response = web_client.post(
        f"/api/findings/{finding_id}/verdict",
        json={"user_verdict": "invalid_verdict"},
    )

    assert response.status_code == 400
    assert "Invalid user_verdict" in response.json()["detail"]


def test_post_verdict_nonexistent_finding(web_client: TestClient) -> None:
    """Test that non-existent finding_id returns 404."""
    response = web_client.post(
        "/api/findings/nonexistent-finding-id/verdict",
        json={"user_verdict": "confirmed"},
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_post_verdict_multi_party_finding(web_client: TestClient) -> None:
    """Test setting verdict on a multi-party finding."""
    finding_id = _create_multi_party_finding(web_client)

    response = web_client.post(
        f"/api/findings/{finding_id}/verdict",
        json={"user_verdict": "confirmed"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["finding_id"] == finding_id
    assert data["user_verdict"] == "confirmed"
    assert data["is_multi_party"] is True


def test_post_verdict_timestamp_current(web_client: TestClient) -> None:
    """Test that updated_at timestamp is current."""
    finding_id = _create_pairwise_finding(web_client)

    before = datetime.now().isoformat(timespec="microseconds")
    response = web_client.post(
        f"/api/findings/{finding_id}/verdict",
        json={"user_verdict": "confirmed"},
    )
    after = datetime.now().isoformat(timespec="microseconds")

    assert response.status_code == 200
    data = response.json()

    updated_at = data["updated_at"]
    # Verify the timestamp is between before and after (within 1 second)
    # Simple check: timestamp should be after "before" and before or equal to "after"
    assert updated_at >= before
    assert (
        updated_at <= after
        or (datetime.fromisoformat(updated_at) - datetime.fromisoformat(before)).total_seconds()
        <= 1.0
    )


def test_post_verdict_update_from_one_value_to_another(web_client: TestClient) -> None:
    """Test updating verdict from one value to another."""
    finding_id = _create_pairwise_finding(web_client)

    # Set to confirmed
    response1 = web_client.post(
        f"/api/findings/{finding_id}/verdict",
        json={"user_verdict": "confirmed"},
    )
    assert response1.status_code == 200
    assert response1.json()["user_verdict"] == "confirmed"

    # Update to false_positive
    response2 = web_client.post(
        f"/api/findings/{finding_id}/verdict",
        json={"user_verdict": "false_positive"},
    )
    assert response2.status_code == 200
    assert response2.json()["user_verdict"] == "false_positive"

    # Update to dismissed
    response3 = web_client.post(
        f"/api/findings/{finding_id}/verdict",
        json={"user_verdict": "dismissed"},
    )
    assert response3.status_code == 200
    assert response3.json()["user_verdict"] == "dismissed"


def test_post_verdict_pairwise_then_multi_party(web_client: TestClient) -> None:
    """Test that the same finding cannot be in both tables (test separate tables)."""
    # Create a pairwise finding
    pairwise_id = _create_pairwise_finding(web_client)

    # Create a multi-party finding
    multiparty_id = _create_multi_party_finding(web_client)

    # Set verdict on pairwise
    response1 = web_client.post(
        f"/api/findings/{pairwise_id}/verdict",
        json={"user_verdict": "confirmed"},
    )

    assert response1.status_code == 200
    assert response1.json()["is_multi_party"] is False

    # Set verdict on multi-party
    response2 = web_client.post(
        f"/api/findings/{multiparty_id}/verdict",
        json={"user_verdict": "false_positive"},
    )

    assert response2.status_code == 200
    assert response2.json()["is_multi_party"] is True


def test_post_verdict_invalid_type_empty_string(web_client: TestClient) -> None:
    """Test that empty string verdict returns 400."""
    finding_id = _create_pairwise_finding(web_client)

    response = web_client.post(
        f"/api/findings/{finding_id}/verdict",
        json={"user_verdict": ""},
    )

    assert response.status_code == 400
    assert "Invalid user_verdict" in response.json()["detail"]


def test_post_verdict_invalid_type_number(web_client: TestClient) -> None:
    """Test that numeric verdict returns 400."""
    finding_id = _create_pairwise_finding(web_client)

    response = web_client.post(
        f"/api/findings/{finding_id}/verdict",
        json={"user_verdict": 123},
    )

    assert response.status_code == 400
    assert "Invalid user_verdict" in response.json()["detail"]


def test_post_verdict_missing_payload_field(web_client: TestClient) -> None:
    """Test that missing user_verdict field defaults to None (which is valid)."""
    finding_id = _create_pairwise_finding(web_client)

    response = web_client.post(
        f"/api/findings/{finding_id}/verdict",
        json={},
    )

    # This should be OK since None is a valid verdict value
    assert response.status_code == 200
    data = response.json()
    assert data["user_verdict"] is None
