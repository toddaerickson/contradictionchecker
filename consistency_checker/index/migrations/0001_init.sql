-- Initial schema for the assertion store.

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    title TEXT,
    doc_date TEXT,
    doc_type TEXT,
    metadata_json TEXT,
    ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS assertions (
    assertion_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    assertion_text TEXT NOT NULL,
    chunk_id TEXT,
    char_start INTEGER,
    char_end INTEGER,
    faiss_row INTEGER UNIQUE,
    embedded_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_assertions_doc ON assertions(doc_id);
CREATE INDEX IF NOT EXISTS idx_assertions_faiss ON assertions(faiss_row);
