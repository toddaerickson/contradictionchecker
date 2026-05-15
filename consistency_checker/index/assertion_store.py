"""SQLite-backed assertion store.

Canonical persistence for documents and extracted assertions. The FAISS index
(Step 8) is a derived view keyed by ``assertions.faiss_row``; this module never
imports FAISS and never depends on the embedding model.

Migrations live in ``migrations/NNNN_*.sql`` and are applied in version order on
:meth:`AssertionStore.migrate`. Each migration is recorded in
``schema_migrations`` so re-running is a no-op.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from consistency_checker.extract.schema import Assertion, Document

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
_MIGRATION_NAME = re.compile(r"^(\d{4})_.*\.sql$")

DEFAULT_EXPORT_COLUMNS: tuple[str, ...] = ("doc_id", "assertion_id", "assertion_text")
_ALL_ASSERTION_COLUMNS: tuple[str, ...] = (
    "assertion_id",
    "doc_id",
    "assertion_text",
    "chunk_id",
    "char_start",
    "char_end",
    "faiss_row",
    "embedded_at",
    "created_at",
)


def _discover_migrations(migrations_dir: Path = MIGRATIONS_DIR) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for path in sorted(migrations_dir.iterdir()):
        match = _MIGRATION_NAME.match(path.name)
        if match is None:
            continue
        out.append((int(match.group(1)), path))
    return out


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _row_to_document(row: sqlite3.Row) -> Document:
    return Document(
        doc_id=row["doc_id"],
        source_path=row["source_path"],
        title=row["title"],
        doc_date=row["doc_date"],
        doc_type=row["doc_type"],
        metadata_json=row["metadata_json"],
        ingested_at=_parse_timestamp(row["ingested_at"]),
    )


def _row_to_assertion(row: sqlite3.Row) -> Assertion:
    return Assertion(
        assertion_id=row["assertion_id"],
        doc_id=row["doc_id"],
        assertion_text=row["assertion_text"],
        chunk_id=row["chunk_id"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        faiss_row=row["faiss_row"],
        embedded_at=_parse_timestamp(row["embedded_at"]),
        created_at=_parse_timestamp(row["created_at"]),
    )


class AssertionStore:
    """SQLite store for documents and assertions.

    Use as a context manager for guaranteed close:

        >>> with AssertionStore(db_path) as store:
        ...     store.migrate()
        ...     store.add_document(doc)
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")

    # --- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> AssertionStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- migrations ---------------------------------------------------------

    def migrate(self, migrations_dir: Path = MIGRATIONS_DIR) -> list[int]:
        """Apply all pending migrations in version order. Returns versions applied."""
        # Ensure schema_migrations exists before we can read from it.
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version INTEGER PRIMARY KEY, "
            "applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        applied: set[int] = {
            row["version"] for row in self._conn.execute("SELECT version FROM schema_migrations")
        }
        newly_applied: list[int] = []
        for version, path in _discover_migrations(migrations_dir):
            if version in applied:
                continue
            sql = path.read_text(encoding="utf-8")
            with self._conn:
                self._conn.executescript(sql)
                self._conn.execute(
                    "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)",
                    (version,),
                )
            newly_applied.append(version)
        return newly_applied

    # --- writes -------------------------------------------------------------

    def add_document(self, doc: Document) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO documents"
                "(doc_id, source_path, title, doc_date, doc_type, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    doc.doc_id,
                    doc.source_path,
                    doc.title,
                    doc.doc_date,
                    doc.doc_type,
                    doc.metadata_json,
                ),
            )

    def add_assertion(self, assertion: Assertion) -> None:
        self.add_assertions([assertion])

    def add_assertions(self, assertions: Iterable[Assertion]) -> None:
        rows = [
            (
                a.assertion_id,
                a.doc_id,
                a.assertion_text,
                a.chunk_id,
                a.char_start,
                a.char_end,
            )
            for a in assertions
        ]
        if not rows:
            return
        with self._conn:
            self._conn.executemany(
                "INSERT OR IGNORE INTO assertions"
                "(assertion_id, doc_id, assertion_text, chunk_id, char_start, char_end) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

    def attach_embeddings(
        self, pairs: Sequence[tuple[str, int]], *, embedded_at: datetime | None = None
    ) -> None:
        """Record the FAISS row index for each assertion id in ``pairs``."""
        stamp = (embedded_at or datetime.now()).isoformat(timespec="seconds")
        with self._conn:
            self._conn.executemany(
                "UPDATE assertions SET faiss_row = ?, embedded_at = ? WHERE assertion_id = ?",
                [(faiss_row, stamp, assertion_id) for assertion_id, faiss_row in pairs],
            )

    # --- reads --------------------------------------------------------------

    def get_document(self, doc_id: str) -> Document | None:
        row = self._conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return _row_to_document(row) if row else None

    def iter_documents(self, *, limit: int | None = None, offset: int = 0) -> Iterator[Document]:
        """Iterate documents ordered by ingested_at desc, then doc_id desc."""
        sql = "SELECT * FROM documents ORDER BY ingested_at DESC, doc_id DESC"
        params: list[Any] = []
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        for row in self._conn.execute(sql, params):
            yield _row_to_document(row)

    def get_documents_bulk(self, ids: Sequence[str]) -> dict[str, Document]:
        """Fetch multiple documents in one query. Returns a dict keyed by doc_id."""
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT * FROM documents WHERE doc_id IN ({placeholders})", list(ids)
        ).fetchall()
        return {row["doc_id"]: _row_to_document(row) for row in rows}

    def get_assertions_bulk(self, ids: Sequence[str]) -> dict[str, Assertion]:
        """Fetch multiple assertions in one query. Returns a dict keyed by assertion_id."""
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT * FROM assertions WHERE assertion_id IN ({placeholders})", list(ids)
        ).fetchall()
        return {row["assertion_id"]: _row_to_assertion(row) for row in rows}

    def get_assertion(self, assertion_id: str) -> Assertion | None:
        row = self._conn.execute(
            "SELECT * FROM assertions WHERE assertion_id = ?", (assertion_id,)
        ).fetchone()
        return _row_to_assertion(row) if row else None

    def iter_assertions(
        self,
        doc_id: str | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> Iterator[Assertion]:
        if doc_id is None:
            sql = "SELECT * FROM assertions ORDER BY created_at, assertion_id"
            params: list[Any] = []
        else:
            sql = "SELECT * FROM assertions WHERE doc_id = ? ORDER BY created_at, assertion_id"
            params = [doc_id]
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        for row in self._conn.execute(sql, params):
            yield _row_to_assertion(row)

    def stats(self) -> dict[str, int]:
        doc_count = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assertion_count = self._conn.execute("SELECT COUNT(*) FROM assertions").fetchone()[0]
        embedded = self._conn.execute(
            "SELECT COUNT(*) FROM assertions WHERE faiss_row IS NOT NULL"
        ).fetchone()[0]
        return {
            "documents": int(doc_count),
            "assertions": int(assertion_count),
            "embedded_assertions": int(embedded),
        }

    # --- export -------------------------------------------------------------

    def export_csv(
        self, path: Path | str, *, columns: Sequence[str] = DEFAULT_EXPORT_COLUMNS
    ) -> None:
        """Export assertions to CSV. Default columns: ``(doc_id, assertion_id, assertion_text)``."""
        unknown = [c for c in columns if c not in _ALL_ASSERTION_COLUMNS]
        if unknown:
            raise ValueError(f"Unknown export columns: {unknown}")
        column_sql = ", ".join(columns)
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            for row in self._conn.execute(
                f"SELECT {column_sql} FROM assertions ORDER BY doc_id, created_at, assertion_id"
            ):
                writer.writerow([row[c] for c in columns])

    def export_jsonl(self, path: Path | str) -> None:
        """Export assertions as JSONL, one assertion per line, with all columns + source path."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sql = (
            "SELECT a.*, d.source_path, d.doc_date FROM assertions a "
            "JOIN documents d ON d.doc_id = a.doc_id "
            "ORDER BY a.doc_id, a.created_at, a.assertion_id"
        )
        with out_path.open("w", encoding="utf-8") as fh:
            for row in self._conn.execute(sql):
                fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
