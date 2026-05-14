"""Tests for the typer CLI.

Provider construction inside the CLI is monkey-patched to return fakes so
``ingest`` and ``check`` can run end-to-end without network access or HF
downloads. Format, store-stats, store-rebuild, and report are exercised
directly because they don't require any LLM/NLI calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.check.gate import AllPairsGate
from consistency_checker.check.llm_judge import FixtureJudge, JudgeVerdict
from consistency_checker.check.nli_checker import FixtureNliChecker, NliResult
from consistency_checker.cli.main import app
from consistency_checker.config import Config
from consistency_checker.extract.atomic_facts import FixtureExtractor
from consistency_checker.extract.schema import Assertion, Document
from consistency_checker.index.assertion_store import AssertionStore
from tests.conftest import HashEmbedder


def write_config(tmp_path: Path, corpus_dir: Path, *, judge_provider: str = "anthropic") -> Path:
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(
        f"""
corpus_dir: {corpus_dir}
judge_provider: {judge_provider}
judge_model: claude-test-model
data_dir: {tmp_path / "store"}
log_dir: {tmp_path / "logs"}
embedder_model: hash
nli_model: fixture
gate_similarity_threshold: -1.0
nli_contradiction_threshold: 0.0
""".strip()
    )
    return cfg_path


def _seed_existing_store(cfg: Config) -> None:
    """Pre-populate the SQLite + FAISS store so ``check`` and ``report`` have data."""
    from consistency_checker.index.embedder import embed_pending
    from consistency_checker.index.faiss_store import FaissStore

    store = AssertionStore(cfg.db_path)
    store.migrate()
    doc_a = Document.from_content("Alpha body.", source_path="alpha.md", title="Alpha")
    doc_b = Document.from_content("Beta body.", source_path="beta.txt", title="Beta")
    store.add_document(doc_a)
    store.add_document(doc_b)
    a = Assertion.build(doc_a.doc_id, "Revenue grew 12%.")
    b = Assertion.build(doc_b.doc_id, "Revenue declined 5%.")
    store.add_assertions([a, b])
    embedder = HashEmbedder(dim=64)
    fs = FaissStore.open_or_create(
        index_path=cfg.faiss_path,
        id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
        dim=embedder.dim,
    )
    embed_pending(store, fs, embedder)
    store.close()


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# --- help / no args ---------------------------------------------------------


def test_cli_no_args_prints_help(runner: CliRunner) -> None:
    result = runner.invoke(app, [])
    assert result.exit_code in (0, 2)
    assert "consistency" in result.stdout.lower()


def test_cli_ingest_help(runner: CliRunner) -> None:
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "corpus" in result.stdout.lower()


# --- ingest -----------------------------------------------------------------


def test_ingest_runs_with_fakes(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("First fact about widgets. Second fact about gadgets.")
    cfg_path = write_config(tmp_path, corpus)

    # Pre-compute chunk ids for the FixtureExtractor.
    from consistency_checker.config import Config as _Config
    from consistency_checker.corpus.chunker import chunk_document
    from consistency_checker.corpus.loader import load_path

    cfg = _Config.from_yaml(cfg_path)
    loaded = load_path(corpus / "a.txt")
    chunks = chunk_document(
        loaded, max_chars=cfg.chunk_max_chars, overlap_chars=cfg.chunk_overlap_chars
    )
    fixtures = {c.chunk_id: [f"Atomic fact {i}."] for i, c in enumerate(chunks)}

    def fake_extractor(_cfg: Any) -> Any:
        return FixtureExtractor(fixtures)

    def fake_embedder(_cfg: Any) -> Any:
        return HashEmbedder(dim=64)

    monkeypatch.setattr("consistency_checker.cli.main.make_extractor", fake_extractor)
    monkeypatch.setattr("consistency_checker.cli.main.make_embedder", fake_embedder)

    result = runner.invoke(app, ["ingest", str(corpus), "--config", str(cfg_path)])
    assert result.exit_code == 0, result.stdout
    assert "Ingested" in result.stdout

    # Verify the store actually got data.
    store = AssertionStore(cfg.db_path)
    store.migrate()
    stats = store.stats()
    store.close()
    assert stats["documents"] == 1
    assert stats["assertions"] >= 1
    assert stats["embedded_assertions"] == stats["assertions"]


# --- check ------------------------------------------------------------------


def test_check_runs_with_fake_nli_and_judge(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)

    # Build fake NLI + judge that agree on contradiction for our only pair.
    store = AssertionStore(cfg.db_path)
    a, b = list(store.iter_assertions())
    store.close()

    nli_fixtures = {
        (a.assertion_text, b.assertion_text): NliResult.from_scores(
            p_contradiction=0.9, p_entailment=0.05, p_neutral=0.05
        ),
        (b.assertion_text, a.assertion_text): NliResult.from_scores(
            p_contradiction=0.85, p_entailment=0.05, p_neutral=0.10
        ),
    }

    def fake_embedder(_cfg: Any) -> Any:
        return HashEmbedder(dim=64)

    canonical = (
        min(a.assertion_id, b.assertion_id),
        max(a.assertion_id, b.assertion_id),
    )
    judge_fixtures = {
        canonical: JudgeVerdict(
            assertion_a_id=canonical[0],
            assertion_b_id=canonical[1],
            verdict="contradiction",
            confidence=0.92,
            rationale="opposite signs",
            evidence_spans=["grew 12%", "declined 5%"],
        )
    }

    class _FakeNliFactory:
        def __init__(self, model_name: str) -> None:
            self._inner = FixtureNliChecker(nli_fixtures)

        def score(self, premise: str, hypothesis: str) -> NliResult:
            return self._inner.score(premise, hypothesis)

    def fake_judge(_cfg: Any) -> Any:
        return FixtureJudge(judge_fixtures)

    monkeypatch.setattr("consistency_checker.cli.main.make_embedder", fake_embedder)
    monkeypatch.setattr("consistency_checker.cli.main.make_judge", fake_judge)
    monkeypatch.setattr(
        "consistency_checker.check.nli_checker.TransformerNliChecker", _FakeNliFactory
    )

    result = runner.invoke(app, ["check", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.stdout
    assert "contradictions" in result.stdout

    # Verify a finding landed. The Alpha-revenue pair triggers the numeric
    # short-circuit (ADR-0005), so the verdict label is numeric_short_circuit;
    # check both labels because either is acceptable for "a contradiction landed".
    store = AssertionStore(cfg.db_path)
    logger = AuditLogger(store)
    findings = [
        *logger.iter_findings(verdict="contradiction"),
        *logger.iter_findings(verdict="numeric_short_circuit"),
    ]
    store.close()
    assert len(findings) == 1


# --- check --deep -----------------------------------------------------------


def test_check_deep_flag_enables_multi_party(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--deep`` runs the multi-party pass and surfaces multi-party findings."""
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)

    # Seed a 3-doc store so a triangle exists.
    from consistency_checker.check.multi_party_judge import (
        FixtureMultiPartyJudge,
        MultiPartyJudgeVerdict,
    )
    from consistency_checker.index.embedder import embed_pending
    from consistency_checker.index.faiss_store import FaissStore

    store = AssertionStore(cfg.db_path)
    store.migrate()
    docs = [
        Document.from_content("Policy A.", source_path="a.md", title="A"),
        Document.from_content("Policy B.", source_path="b.md", title="B"),
        Document.from_content("Policy C.", source_path="c.md", title="C"),
    ]
    for d in docs:
        store.add_document(d)
    assertions = [
        Assertion.build(docs[0].doc_id, "All employees get four weeks vacation."),
        Assertion.build(docs[1].doc_id, "Engineers are employees."),
        Assertion.build(docs[2].doc_id, "Engineers get two weeks vacation."),
    ]
    store.add_assertions(assertions)
    embedder = HashEmbedder(dim=64)
    fs = FaissStore.open_or_create(
        index_path=cfg.faiss_path,
        id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
        dim=embedder.dim,
    )
    embed_pending(store, fs, embedder)
    store.close()

    canonical_ids = tuple(sorted(a.assertion_id for a in assertions))
    multi_party_fixtures = {
        canonical_ids: MultiPartyJudgeVerdict(
            assertion_ids=canonical_ids,
            verdict="multi_party_contradiction",
            confidence=0.91,
            rationale="A ∧ B ⇒ ¬C",
            contradicting_subset=("A", "B", "C"),
        )
    }

    def fake_embedder(_cfg: Any) -> Any:
        return HashEmbedder(dim=64)

    def fake_judge(_cfg: Any) -> Any:
        return FixtureJudge({})

    def fake_multi_party_judge(_cfg: Any) -> Any:
        return FixtureMultiPartyJudge(multi_party_fixtures)

    class _FakeNliFactory:
        def __init__(self, model_name: str) -> None:
            self._inner = FixtureNliChecker({})

        def score(self, premise: str, hypothesis: str) -> NliResult:
            return self._inner.score(premise, hypothesis)

    monkeypatch.setattr("consistency_checker.cli.main.make_embedder", fake_embedder)
    monkeypatch.setattr("consistency_checker.cli.main.make_judge", fake_judge)
    monkeypatch.setattr(
        "consistency_checker.cli.main.make_multi_party_judge", fake_multi_party_judge
    )
    monkeypatch.setattr(
        "consistency_checker.check.nli_checker.TransformerNliChecker", _FakeNliFactory
    )

    result = runner.invoke(app, ["check", "--deep", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.stdout
    assert "multi-party" in result.stdout
    assert "triangles" in result.stdout

    store = AssertionStore(cfg.db_path)
    logger = AuditLogger(store)
    multi = list(logger.iter_multi_party_findings(verdict="multi_party_contradiction"))
    store.close()
    assert len(multi) == 1


def test_check_without_deep_flag_skips_multi_party_factory(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``--deep``, ``make_multi_party_judge`` is never called."""
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)

    multi_party_calls: list[Any] = []

    def fake_embedder(_cfg: Any) -> Any:
        return HashEmbedder(dim=64)

    def fake_judge(_cfg: Any) -> Any:
        return FixtureJudge({})

    def spy_multi_party(_cfg: Any) -> Any:
        multi_party_calls.append(_cfg)
        raise AssertionError("make_multi_party_judge must not be called without --deep")

    class _FakeNliFactory:
        def __init__(self, model_name: str) -> None:
            self._inner = FixtureNliChecker({})

        def score(self, premise: str, hypothesis: str) -> NliResult:
            return self._inner.score(premise, hypothesis)

    monkeypatch.setattr("consistency_checker.cli.main.make_embedder", fake_embedder)
    monkeypatch.setattr("consistency_checker.cli.main.make_judge", fake_judge)
    monkeypatch.setattr("consistency_checker.cli.main.make_multi_party_judge", spy_multi_party)
    monkeypatch.setattr(
        "consistency_checker.check.nli_checker.TransformerNliChecker", _FakeNliFactory
    )

    result = runner.invoke(app, ["check", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.stdout
    assert multi_party_calls == []
    assert "multi-party" not in result.stdout


# --- report -----------------------------------------------------------------


def test_report_writes_markdown(runner: CliRunner, tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)

    # Pre-record a finding so the report has content.
    store = AssertionStore(cfg.db_path)
    logger = AuditLogger(store)
    rid = logger.begin_run(run_id="cli_report_test")
    for pair in AllPairsGate().candidates(store):
        logger.record_finding(
            rid,
            candidate=pair,
            nli=NliResult.from_scores(p_contradiction=0.8, p_entailment=0.1, p_neutral=0.1),
            verdict=JudgeVerdict(
                assertion_a_id=pair.a.assertion_id,
                assertion_b_id=pair.b.assertion_id,
                verdict="contradiction",
                confidence=0.9,
                rationale="opposing signs",
            ),
        )
    logger.end_run(rid, n_assertions=2, n_pairs_gated=1, n_pairs_judged=1)
    store.close()

    out = tmp_path / "out.md"
    result = runner.invoke(
        app, ["report", "--out", str(out), "--config", str(cfg_path), "--run", "cli_report_test"]
    )
    assert result.exit_code == 0, result.stdout
    assert out.exists()
    body = out.read_text()
    assert "Consistency check report" in body
    assert "opposing signs" in body


def test_report_no_runs_errors(runner: CliRunner, tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    # Empty store, no runs.
    store = AssertionStore(cfg.db_path)
    store.migrate()
    store.close()

    result = runner.invoke(
        app, ["report", "--out", str(tmp_path / "empty.md"), "--config", str(cfg_path)]
    )
    assert result.exit_code == 2
    assert "No runs found" in result.stderr or "No runs found" in result.stdout


# --- export -----------------------------------------------------------------


def test_export_csv(runner: CliRunner, tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)

    out = tmp_path / "assertions.csv"
    result = runner.invoke(app, ["export", "csv", "--out", str(out), "--config", str(cfg_path)])
    assert result.exit_code == 0, result.stdout
    text = out.read_text()
    assert "doc_id,assertion_id,assertion_text" in text


def test_export_jsonl(runner: CliRunner, tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)

    out = tmp_path / "assertions.jsonl"
    result = runner.invoke(app, ["export", "jsonl", "--out", str(out), "--config", str(cfg_path)])
    assert result.exit_code == 0, result.stdout
    lines = out.read_text().splitlines()
    assert lines
    json.loads(lines[0])  # well-formed JSON


def test_export_bad_format_errors(runner: CliRunner, tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)
    result = runner.invoke(
        app, ["export", "xml", "--out", str(tmp_path / "x.xml"), "--config", str(cfg_path)]
    )
    assert result.exit_code != 0


# --- store maintenance -----------------------------------------------------


def test_store_stats(runner: CliRunner, tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)
    result = runner.invoke(app, ["store", "stats", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.stdout
    assert "documents" in result.stdout
    assert "assertions" in result.stdout


def test_store_rebuild_index(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)

    def fake_embedder(_cfg: Any) -> Any:
        return HashEmbedder(dim=64)

    monkeypatch.setattr("consistency_checker.cli.main.make_embedder", fake_embedder)
    result = runner.invoke(app, ["store", "rebuild-index", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.stdout
    assert "Rebuilt FAISS index" in result.stdout


# --- config errors ---------------------------------------------------------


def test_missing_config_errors(runner: CliRunner, tmp_path: Path) -> None:
    """Pointing --config at a non-existent file must fail clearly."""
    result = runner.invoke(app, ["store", "stats", "--config", str(tmp_path / "missing.yml")])
    assert result.exit_code != 0
