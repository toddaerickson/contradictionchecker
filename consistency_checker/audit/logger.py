"""SQLite-backed audit logger.

Persists every judge verdict (contradiction / not_contradiction / uncertain) so
a run is fully reproducible from logged inputs alone. Findings are organised
under a ``pipeline_runs`` row that records the run's config and totals.

The schema lives in ``consistency_checker/index/migrations/0002_audit.sql`` and
is applied via the same :meth:`AssertionStore.migrate` pump as the canonical
tables — there is one database, one migration journal.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from consistency_checker.check.gate import CandidatePair
from consistency_checker.check.llm_judge import JudgeVerdict
from consistency_checker.check.nli_checker import NliResult
from consistency_checker.check.providers.base import CONTRADICTION_VERDICTS
from consistency_checker.extract.schema import hash_id
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.logging_setup import get_logger

if TYPE_CHECKING:
    from consistency_checker.check.definition_checker import DefinitionFinding

_log = get_logger(__name__)

RunStatus = Literal["pending", "running", "done", "failed"]
TERMINAL_RUN_STATUSES: frozenset[RunStatus] = frozenset({"done", "failed"})


@dataclass(frozen=True, slots=True)
class PipelineRun:
    """One scan from ingest through report."""

    run_id: str
    started_at: datetime | None
    finished_at: datetime | None
    config_json: str | None
    n_assertions: int
    n_pairs_gated: int
    n_pairs_judged: int
    n_findings: int
    notes: str | None
    run_status: RunStatus = "pending"
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class Finding:
    """One judge verdict in a run."""

    finding_id: str
    run_id: str
    assertion_a_id: str
    assertion_b_id: str
    gate_score: float | None
    nli_label: str | None
    nli_p_contradiction: float | None
    nli_p_entailment: float | None
    nli_p_neutral: float | None
    judge_verdict: str | None
    judge_confidence: float | None
    judge_rationale: str | None
    evidence_spans: list[str] = field(default_factory=list)
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class MultiPartyFinding:
    """One multi-document conditional contradiction (ADR-0006, F1).

    ``assertion_ids`` has at least three entries; ``doc_ids`` has at least two
    distinct entries. ``triangle_edge_scores`` is the list of FAISS-similarity
    edges that survived Stage A for the three assertion ids, stored verbatim
    so a replay can re-rank triangles without re-running the gate.
    """

    finding_id: str
    run_id: str
    assertion_ids: list[str]
    doc_ids: list[str]
    triangle_edge_scores: list[tuple[str, str, float]]
    judge_verdict: str | None
    judge_confidence: float | None
    judge_rationale: str | None
    evidence_spans: list[str] = field(default_factory=list)
    created_at: datetime | None = None


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _row_to_run(row: sqlite3.Row) -> PipelineRun:
    return PipelineRun(
        run_id=row["run_id"],
        started_at=_parse_ts(row["started_at"]),
        finished_at=_parse_ts(row["finished_at"]),
        config_json=row["config_json"],
        n_assertions=int(row["n_assertions"]),
        n_pairs_gated=int(row["n_pairs_gated"]),
        n_pairs_judged=int(row["n_pairs_judged"]),
        n_findings=int(row["n_findings"]),
        notes=row["notes"],
        run_status=row["run_status"],
        error_message=row["error_message"],
    )


def _row_to_finding(row: sqlite3.Row) -> Finding:
    spans_json = row["evidence_spans_json"]
    spans = json.loads(spans_json) if spans_json else []
    return Finding(
        finding_id=row["finding_id"],
        run_id=row["run_id"],
        assertion_a_id=row["assertion_a_id"],
        assertion_b_id=row["assertion_b_id"],
        gate_score=row["gate_score"],
        nli_label=row["nli_label"],
        nli_p_contradiction=row["nli_p_contradiction"],
        nli_p_entailment=row["nli_p_entailment"],
        nli_p_neutral=row["nli_p_neutral"],
        judge_verdict=row["judge_verdict"],
        judge_confidence=row["judge_confidence"],
        judge_rationale=row["judge_rationale"],
        evidence_spans=spans,
        created_at=_parse_ts(row["created_at"]),
    )


def _row_to_multi_party_finding(row: sqlite3.Row) -> MultiPartyFinding:
    spans_json = row["evidence_spans_json"]
    spans = json.loads(spans_json) if spans_json else []
    edges_json = row["triangle_edge_scores_json"]
    raw_edges = json.loads(edges_json) if edges_json else []
    edges: list[tuple[str, str, float]] = [(str(a), str(b), float(s)) for a, b, s in raw_edges]
    return MultiPartyFinding(
        finding_id=row["finding_id"],
        run_id=row["run_id"],
        assertion_ids=list(json.loads(row["assertion_ids_json"])),
        doc_ids=list(json.loads(row["doc_ids_json"])),
        triangle_edge_scores=edges,
        judge_verdict=row["judge_verdict"],
        judge_confidence=row["judge_confidence"],
        judge_rationale=row["judge_rationale"],
        evidence_spans=spans,
        created_at=_parse_ts(row["created_at"]),
    )


class AuditLogger:
    """Records pipeline runs and per-pair judge verdicts in the assertion store."""

    def __init__(self, store: AssertionStore) -> None:
        self._store = store
        # Reach into the underlying connection — the audit logger lives in the
        # same SQLite database as the canonical tables on purpose.
        self._conn: sqlite3.Connection = store._conn

    # --- run lifecycle ------------------------------------------------------

    def begin_run(
        self,
        *,
        run_id: str | None = None,
        config: dict[str, Any] | None = None,
        notes: str | None = None,
        run_status: RunStatus = "running",
    ) -> str:
        rid = run_id or uuid.uuid4().hex
        config_json = json.dumps(config, default=str) if config is not None else None
        with self._conn:
            self._conn.execute(
                "INSERT INTO pipeline_runs (run_id, config_json, notes, run_status) "
                "VALUES (?, ?, ?, ?)",
                (rid, config_json, notes, run_status),
            )
        return rid

    def end_run(
        self,
        run_id: str,
        *,
        n_assertions: int = 0,
        n_pairs_gated: int = 0,
        n_pairs_judged: int = 0,
        n_findings: int | None = None,
        run_status: RunStatus = "done",
    ) -> None:
        if n_findings is None:
            verdicts = sorted(CONTRADICTION_VERDICTS)
            placeholders = ", ".join("?" * len(verdicts))
            pair_count = self._conn.execute(
                f"SELECT COUNT(*) FROM findings WHERE run_id = ? "
                f"AND judge_verdict IN ({placeholders})",
                (run_id, *verdicts),
            ).fetchone()[0]
            mp_count = self._conn.execute(
                "SELECT COUNT(*) FROM multi_party_findings WHERE run_id = ? "
                "AND judge_verdict = 'multi_party_contradiction'",
                (run_id,),
            ).fetchone()[0]
            n_findings = int(pair_count) + int(mp_count)
        finished = datetime.now().isoformat(timespec="seconds")
        with self._conn:
            self._conn.execute(
                "UPDATE pipeline_runs SET finished_at = ?, n_assertions = ?, "
                "n_pairs_gated = ?, n_pairs_judged = ?, n_findings = ?, "
                "run_status = ? WHERE run_id = ?",
                (
                    finished,
                    n_assertions,
                    n_pairs_gated,
                    n_pairs_judged,
                    n_findings,
                    run_status,
                    run_id,
                ),
            )

    def update_run_status(
        self, run_id: str, status: RunStatus, *, error_message: str | None = None
    ) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE pipeline_runs SET run_status = ?, error_message = ? WHERE run_id = ?",
                (status, error_message, run_id),
            )

    # --- writes -------------------------------------------------------------

    def record_finding(
        self,
        run_id: str,
        *,
        candidate: CandidatePair,
        nli: NliResult | None,
        verdict: JudgeVerdict,
    ) -> str:
        a_id = candidate.a.assertion_id
        b_id = candidate.b.assertion_id
        finding_id = hash_id(run_id, a_id, b_id)
        spans_json = json.dumps(verdict.evidence_spans)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO findings ("
                "finding_id, run_id, assertion_a_id, assertion_b_id, "
                "gate_score, nli_label, nli_p_contradiction, nli_p_entailment, nli_p_neutral, "
                "judge_verdict, judge_confidence, judge_rationale, evidence_spans_json"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    finding_id,
                    run_id,
                    a_id,
                    b_id,
                    candidate.score,
                    nli.label if nli else None,
                    nli.p_contradiction if nli else None,
                    nli.p_entailment if nli else None,
                    nli.p_neutral if nli else None,
                    verdict.verdict,
                    verdict.confidence,
                    verdict.rationale,
                    spans_json,
                ),
            )
        return finding_id

    def record_definition_finding(
        self,
        run_id: str,
        *,
        finding: DefinitionFinding,
    ) -> str:
        """Persist a definition-inconsistency finding into the shared findings table.

        Uses ``detector_type='definition_inconsistency'``. The NLI and gate-score
        columns are intentionally left null — the definition detector bypasses
        the NLI gate (term-grouping replaces it).
        """
        a_id = finding.pair.a.assertion_id
        b_id = finding.pair.b.assertion_id
        finding_id = hash_id(run_id, "definition", a_id, b_id)
        spans_json = json.dumps(finding.verdict.evidence_spans)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO findings ("
                "finding_id, run_id, assertion_a_id, assertion_b_id, "
                "gate_score, nli_label, nli_p_contradiction, nli_p_entailment, nli_p_neutral, "
                "judge_verdict, judge_confidence, judge_rationale, evidence_spans_json, "
                "detector_type"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    finding_id,
                    run_id,
                    a_id,
                    b_id,
                    None,
                    None,
                    None,
                    None,
                    None,
                    finding.verdict.verdict,
                    finding.verdict.confidence,
                    finding.verdict.rationale,
                    spans_json,
                    "definition_inconsistency",
                ),
            )
        return finding_id

    def record_multi_party_finding(
        self,
        run_id: str,
        *,
        assertion_ids: list[str],
        doc_ids: list[str],
        triangle_edge_scores: list[tuple[str, str, float]] | None = None,
        judge_verdict: str,
        judge_confidence: float | None = None,
        judge_rationale: str | None = None,
        evidence_spans: list[str] | None = None,
    ) -> str:
        """Insert a row into ``multi_party_findings``.

        ``finding_id`` is a content hash over the run and the sorted assertion
        ids, so re-recording the same triangle within a run replaces the
        existing row (idempotent).
        """
        if len(assertion_ids) < 3:
            raise ValueError("multi-party finding needs at least 3 assertion ids")
        sorted_ids = sorted(assertion_ids)
        if len({d for d in doc_ids}) < 2:
            raise ValueError("multi-party finding spans must include >= 2 distinct doc ids")
        finding_id = hash_id(run_id, *sorted_ids)
        edges_payload = (
            json.dumps([[a, b, s] for a, b, s in triangle_edge_scores])
            if triangle_edge_scores is not None
            else None
        )
        spans_payload = json.dumps(evidence_spans if evidence_spans is not None else [])
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO multi_party_findings ("
                "finding_id, run_id, assertion_ids_json, doc_ids_json, "
                "triangle_edge_scores_json, judge_verdict, judge_confidence, "
                "judge_rationale, evidence_spans_json"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    finding_id,
                    run_id,
                    json.dumps(sorted_ids),
                    json.dumps(list(doc_ids)),
                    edges_payload,
                    judge_verdict,
                    judge_confidence,
                    judge_rationale,
                    spans_payload,
                ),
            )
            # FK side table (migration 0005) — referential integrity into
            # assertions. The INSERT OR REPLACE above cascade-deletes any prior
            # join rows for this finding_id, so this only ever inserts fresh.
            self._conn.executemany(
                "INSERT INTO multi_party_finding_assertions "
                "(finding_id, assertion_id, position) VALUES (?, ?, ?)",
                [(finding_id, aid, idx) for idx, aid in enumerate(sorted_ids)],
            )
        return finding_id

    # --- reads --------------------------------------------------------------

    def get_run(self, run_id: str) -> PipelineRun | None:
        row = self._conn.execute(
            "SELECT * FROM pipeline_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return _row_to_run(row) if row else None

    def most_recent_run(self) -> PipelineRun | None:
        """Return the most recently started run, or ``None`` if no runs exist."""
        row = self._conn.execute(
            "SELECT * FROM pipeline_runs ORDER BY started_at DESC, run_id DESC LIMIT 1"
        ).fetchone()
        return _row_to_run(row) if row else None

    def iter_findings(
        self,
        *,
        run_id: str | None = None,
        verdict: str | None = None,
        detector_type: str | None = None,
    ) -> Iterator[Finding]:
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if verdict is not None:
            clauses.append("judge_verdict = ?")
            params.append(verdict)
        if detector_type is not None:
            clauses.append("detector_type = ?")
            params.append(detector_type)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        cursor = self._conn.execute(
            f"SELECT * FROM findings {where} ORDER BY created_at, finding_id",
            params,
        )
        for row in cursor:
            yield _row_to_finding(row)

    def get_finding(self, finding_id: str) -> Finding | None:
        row = self._conn.execute(
            "SELECT * FROM findings WHERE finding_id = ?", (finding_id,)
        ).fetchone()
        return _row_to_finding(row) if row else None

    def iter_multi_party_findings(
        self,
        *,
        run_id: str | None = None,
        verdict: str | None = None,
    ) -> Iterator[MultiPartyFinding]:
        """Iterate multi-party (triangle) findings, optionally filtered by run / verdict."""
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if verdict is not None:
            clauses.append("judge_verdict = ?")
            params.append(verdict)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        cursor = self._conn.execute(
            f"SELECT * FROM multi_party_findings {where} ORDER BY created_at, finding_id",
            params,
        )
        for row in cursor:
            yield _row_to_multi_party_finding(row)

    def get_multi_party_finding(self, finding_id: str) -> MultiPartyFinding | None:
        row = self._conn.execute(
            "SELECT * FROM multi_party_findings WHERE finding_id = ?", (finding_id,)
        ).fetchone()
        return _row_to_multi_party_finding(row) if row else None
