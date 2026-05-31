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
from tests.conftest import HashEmbedder, strip_ansi


def write_config(
    tmp_path: Path,
    corpus_dir: Path,
    *,
    judge_provider: str = "anthropic",
    pairwise_enabled: bool = True,
) -> Path:
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
pairwise_enabled: {str(pairwise_enabled).lower()}
""".strip()
    )
    return cfg_path


def _seed_existing_store(cfg: Config) -> None:
    """Pre-populate the SQLite + FAISS store so ``check`` and ``report`` have data."""
    from consistency_checker.index.embedder import embed_pending
    from consistency_checker.index.faiss_store import FaissStore

    store = AssertionStore(cfg.db_path)
    store.migrate()
    # Use "default" to match the Task-5 scaffold in cli/main.py check command.
    _cid = store.get_or_create_corpus("default", str(cfg.corpus_dir), "moonshot")
    doc_a = Document.from_content("Alpha body.", source_path="alpha.md", title="Alpha")
    doc_b = Document.from_content("Beta body.", source_path="beta.txt", title="Beta")
    store.add_document(doc_a, corpus_id=_cid)
    store.add_document(doc_b, corpus_id=_cid)
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

    result = runner.invoke(
        app, ["ingest", str(corpus), "--config", str(cfg_path), "--corpus", "default"]
    )
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

        def release(self) -> None:
            self._inner.release()

    def fake_judge(_cfg: Any) -> Any:
        return FixtureJudge(judge_fixtures)

    monkeypatch.setattr("consistency_checker.cli.main.make_embedder", fake_embedder)
    monkeypatch.setattr("consistency_checker.cli.main.make_judge", fake_judge)
    monkeypatch.setattr(
        "consistency_checker.check.nli_checker.TransformerNliChecker", _FakeNliFactory
    )

    result = runner.invoke(app, ["check", "--corpus", "default", "--config", str(cfg_path)])
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
    # Use "default" to match the Task-5 scaffold corpus_id in cli/main.py check.
    _cid = store.get_or_create_corpus("default", str(cfg.corpus_dir), "moonshot")
    docs = [
        Document.from_content("Policy A.", source_path="a.md", title="A"),
        Document.from_content("Policy B.", source_path="b.md", title="B"),
        Document.from_content("Policy C.", source_path="c.md", title="C"),
    ]
    for d in docs:
        store.add_document(d, corpus_id=_cid)
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

        def release(self) -> None:
            self._inner.release()

    monkeypatch.setattr("consistency_checker.cli.main.make_embedder", fake_embedder)
    monkeypatch.setattr("consistency_checker.cli.main.make_judge", fake_judge)
    monkeypatch.setattr(
        "consistency_checker.cli.main.make_multi_party_judge", fake_multi_party_judge
    )
    monkeypatch.setattr(
        "consistency_checker.check.nli_checker.TransformerNliChecker", _FakeNliFactory
    )

    result = runner.invoke(
        app, ["check", "--deep", "--corpus", "default", "--config", str(cfg_path)]
    )
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

        def release(self) -> None:
            self._inner.release()

    monkeypatch.setattr("consistency_checker.cli.main.make_embedder", fake_embedder)
    monkeypatch.setattr("consistency_checker.cli.main.make_judge", fake_judge)
    monkeypatch.setattr("consistency_checker.cli.main.make_multi_party_judge", spy_multi_party)
    monkeypatch.setattr(
        "consistency_checker.check.nli_checker.TransformerNliChecker", _FakeNliFactory
    )

    result = runner.invoke(app, ["check", "--corpus", "default", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.stdout
    assert multi_party_calls == []
    assert "multi-party" not in result.stdout


# --- Task 3: --pairwise/--no-pairwise tri-state + lazy NLI load -------------


def test_check_no_pairwise_flag_disables_pairwise_when_config_on(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--no-pairwise`` skips NLI construction even when config has pairwise_enabled=True."""
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")  # pairwise_enabled: true
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)

    def fake_embedder(_cfg: Any) -> Any:
        return HashEmbedder(dim=64)

    def fake_judge(_cfg: Any) -> Any:
        return FixtureJudge({})

    class _ForbiddenNliFactory:
        def __init__(self, model_name: str) -> None:
            raise AssertionError("NLI should not be constructed when --no-pairwise")

    monkeypatch.setattr("consistency_checker.cli.main.make_embedder", fake_embedder)
    monkeypatch.setattr("consistency_checker.cli.main.make_judge", fake_judge)
    monkeypatch.setattr(
        "consistency_checker.check.nli_checker.TransformerNliChecker", _ForbiddenNliFactory
    )

    result = runner.invoke(
        app, ["check", "--no-pairwise", "--corpus", "default", "--config", str(cfg_path)]
    )
    assert result.exit_code == 0, result.stdout
    assert "pairwise=off" in result.stdout


def test_check_pairwise_flag_overrides_config_off(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--pairwise`` constructs NLI even when config has pairwise_enabled=False."""
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused", pairwise_enabled=False)
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)

    nli_calls: list[str | None] = []

    def fake_embedder(_cfg: Any) -> Any:
        return HashEmbedder(dim=64)

    def fake_judge(_cfg: Any) -> Any:
        return FixtureJudge({})

    class _SpyNliFactory:
        def __init__(self, model_name: str | None = None) -> None:
            nli_calls.append(model_name)
            self._inner = FixtureNliChecker({})

        def score(self, premise: str, hypothesis: str) -> NliResult:
            return self._inner.score(premise, hypothesis)

        def release(self) -> None:
            self._inner.release()

    monkeypatch.setattr("consistency_checker.cli.main.make_embedder", fake_embedder)
    monkeypatch.setattr("consistency_checker.cli.main.make_judge", fake_judge)
    monkeypatch.setattr(
        "consistency_checker.check.nli_checker.TransformerNliChecker", _SpyNliFactory
    )

    result = runner.invoke(
        app, ["check", "--pairwise", "--corpus", "default", "--config", str(cfg_path)]
    )
    assert result.exit_code == 0, result.stdout
    assert len(nli_calls) == 1
    assert "gated" in result.stdout


def test_check_omitted_respects_config(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No --pairwise flag → config value wins (pairwise_enabled=true → NLI built)."""
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")  # pairwise_enabled: true
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)

    nli_calls: list[str | None] = []

    def fake_embedder(_cfg: Any) -> Any:
        return HashEmbedder(dim=64)

    def fake_judge(_cfg: Any) -> Any:
        return FixtureJudge({})

    class _SpyNliFactory:
        def __init__(self, model_name: str | None = None) -> None:
            nli_calls.append(model_name)
            self._inner = FixtureNliChecker({})

        def score(self, premise: str, hypothesis: str) -> NliResult:
            return self._inner.score(premise, hypothesis)

        def release(self) -> None:
            self._inner.release()

    monkeypatch.setattr("consistency_checker.cli.main.make_embedder", fake_embedder)
    monkeypatch.setattr("consistency_checker.cli.main.make_judge", fake_judge)
    monkeypatch.setattr(
        "consistency_checker.check.nli_checker.TransformerNliChecker", _SpyNliFactory
    )

    result = runner.invoke(app, ["check", "--corpus", "default", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.stdout
    assert len(nli_calls) == 1


def test_check_deep_without_pairwise_rejected(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--deep --no-pairwise`` fails fast with a helpful BadParameter message."""
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)

    result = runner.invoke(
        app,
        [
            "check",
            "--deep",
            "--no-pairwise",
            "--corpus",
            "default",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code != 0
    combined = (result.stdout or "") + (result.stderr or "") + str(result.exception or "")
    assert "--deep requires" in combined


def test_check_deep_with_pairwise_accepted(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--deep --pairwise`` is accepted and runs both passes."""
    from consistency_checker.check.multi_party_judge import FixtureMultiPartyJudge

    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)

    def fake_embedder(_cfg: Any) -> Any:
        return HashEmbedder(dim=64)

    def fake_judge(_cfg: Any) -> Any:
        return FixtureJudge({})

    def fake_multi_party_judge(_cfg: Any) -> Any:
        return FixtureMultiPartyJudge({})

    class _SpyNliFactory:
        def __init__(self, model_name: str | None = None) -> None:
            self._inner = FixtureNliChecker({})

        def score(self, premise: str, hypothesis: str) -> NliResult:
            return self._inner.score(premise, hypothesis)

        def release(self) -> None:
            self._inner.release()

    monkeypatch.setattr("consistency_checker.cli.main.make_embedder", fake_embedder)
    monkeypatch.setattr("consistency_checker.cli.main.make_judge", fake_judge)
    monkeypatch.setattr(
        "consistency_checker.cli.main.make_multi_party_judge", fake_multi_party_judge
    )
    monkeypatch.setattr(
        "consistency_checker.check.nli_checker.TransformerNliChecker", _SpyNliFactory
    )

    result = runner.invoke(
        app, ["check", "--deep", "--pairwise", "--corpus", "default", "--config", str(cfg_path)]
    )
    assert result.exit_code == 0, result.stdout
    assert "gated" in result.stdout
    assert "multi-party" in result.stdout


def test_estimate_cost_no_pairwise_flag_zeros_candidate_pairs(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``estimate-cost --no-pairwise`` reports 0 gated pairs and the disabled footnote."""
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)

    result = runner.invoke(
        app,
        [
            "estimate-cost",
            "--no-pairwise",
            "--corpus",
            "default",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Stage A - gated pairs:        0" in result.stdout
    assert "Pairwise detector disabled" in result.stdout


def test_estimate_cost_pairwise_flag_includes_candidate_pairs(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``estimate-cost --pairwise`` exercises the FAISS scan and omits the disabled note."""
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused", pairwise_enabled=False)
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)

    result = runner.invoke(
        app,
        [
            "estimate-cost",
            "--pairwise",
            "--corpus",
            "default",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Stage A - gated pairs:        0" not in result.stdout
    assert "Pairwise detector disabled" not in result.stdout


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
    result = runner.invoke(
        app, ["export", "csv", "--out", str(out), "--corpus", "default", "--config", str(cfg_path)]
    )
    assert result.exit_code == 0, result.stdout
    text = out.read_text()
    assert "doc_id,assertion_id,assertion_text" in text


def test_export_jsonl(runner: CliRunner, tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)

    out = tmp_path / "assertions.jsonl"
    result = runner.invoke(
        app,
        ["export", "jsonl", "--out", str(out), "--corpus", "default", "--config", str(cfg_path)],
    )
    assert result.exit_code == 0, result.stdout
    lines = out.read_text().splitlines()
    assert lines
    json.loads(lines[0])  # well-formed JSON


def test_export_bad_format_errors(runner: CliRunner, tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)
    result = runner.invoke(
        app,
        [
            "export",
            "xml",
            "--out",
            str(tmp_path / "x.xml"),
            "--corpus",
            "default",
            "--config",
            str(cfg_path),
        ],
    )
    assert result.exit_code != 0


# --- G0b: optional --out -----------------------------------------------------


def test_report_without_out_writes_to_default_reports_dir(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Omitting --out writes to <data_dir>/reports/cc_report_*.md."""
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)
    store = AssertionStore(cfg.db_path)
    logger = AuditLogger(store)
    rid = logger.begin_run(run_id="abcd1234efgh5678")
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
                rationale="opposing",
            ),
        )
    logger.end_run(rid, n_assertions=2, n_pairs_gated=1, n_pairs_judged=1)
    store.close()

    result = runner.invoke(app, ["report", "--config", str(cfg_path), "--run", rid])
    assert result.exit_code == 0, result.stdout
    reports_dir = cfg.data_dir / "reports"
    generated = list(reports_dir.glob("cc_report_*_abcd1234.md"))
    assert len(generated) == 1, (
        f"expected one report under {reports_dir}, found {list(reports_dir.glob('*'))}"
    )
    assert "Consistency check report" in generated[0].read_text()


def test_export_csv_without_out_writes_to_default_reports_dir(
    runner: CliRunner, tmp_path: Path
) -> None:
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)
    result = runner.invoke(app, ["export", "csv", "--corpus", "default", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.stdout
    generated = list((cfg.data_dir / "reports").glob("cc_assertions_*.csv"))
    assert len(generated) == 1
    assert "doc_id,assertion_id,assertion_text" in generated[0].read_text()


def test_export_jsonl_without_out_writes_to_default_reports_dir(
    runner: CliRunner, tmp_path: Path
) -> None:
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)
    result = runner.invoke(
        app, ["export", "jsonl", "--corpus", "default", "--config", str(cfg_path)]
    )
    assert result.exit_code == 0, result.stdout
    generated = list((cfg.data_dir / "reports").glob("cc_assertions_*.jsonl"))
    assert len(generated) == 1
    json.loads(generated[0].read_text().splitlines()[0])


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


def test_estimate_cost_command_prints_ceiling(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)

    def fake_embedder(_cfg: Any) -> Any:
        return HashEmbedder(dim=64)

    monkeypatch.setattr("consistency_checker.cli.main.make_embedder", fake_embedder)

    result = runner.invoke(app, ["estimate-cost", "--corpus", "default", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.stdout
    assert "Run cost estimate" in result.stdout
    assert "Assertions in store:" in result.stdout
    assert "Estimated API cost:" in result.stdout
    assert "CEILING" in result.stdout


def test_estimate_cost_per_call_flags_flow_through(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--per-call-low / --per-call-high are wired into the estimate."""
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)

    captured: dict[str, float] = {}

    def fake_estimate_cost(_cfg: Any, **kwargs: Any) -> Any:
        captured["per_call_low"] = kwargs["per_call_low"]
        captured["per_call_high"] = kwargs["per_call_high"]
        from consistency_checker.pipeline import CostEstimate

        return CostEstimate(
            n_assertions=0,
            n_candidate_pairs=0,
            n_definition_pairs=0,
            judge_calls_ceiling=0,
            est_cost_low=0.0,
            est_cost_high=0.0,
            per_call_low=kwargs["per_call_low"],
            per_call_high=kwargs["per_call_high"],
        )

    monkeypatch.setattr("consistency_checker.cli.main.run_estimate_cost", fake_estimate_cost)

    result = runner.invoke(
        app,
        [
            "estimate-cost",
            "--corpus",
            "default",
            "--config",
            str(cfg_path),
            "--per-call-low",
            "0.001",
            "--per-call-high",
            "0.020",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert captured["per_call_low"] == 0.001
    assert captured["per_call_high"] == 0.020


# --- org-scope flag + corpus warnings --------------------------------------


def test_check_accepts_org_scope_flag(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--org-scope`` succeeds and propagates org_scope_enabled=True into the config."""
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)

    captured: dict[str, Any] = {}

    def fake_embedder(_cfg: Any) -> Any:
        return HashEmbedder(dim=64)

    def fake_judge(_cfg: Any) -> Any:
        return FixtureJudge({})

    def fake_definition_checker(_cfg: Any) -> Any:
        captured["org_scope_enabled"] = _cfg.org_scope_enabled
        # Return None — pipeline.check handles definition_checker=None gracefully.
        from consistency_checker.check.definition_checker import DefinitionChecker
        from consistency_checker.check.definition_judge import FixtureDefinitionJudge

        return DefinitionChecker(
            judge=FixtureDefinitionJudge({}),
            org_scope_enabled=_cfg.org_scope_enabled,
        )

    class _FakeNliFactory:
        def __init__(self, model_name: str) -> None:
            self._inner = FixtureNliChecker({})

        def score(self, premise: str, hypothesis: str) -> NliResult:
            return self._inner.score(premise, hypothesis)

        def release(self) -> None:
            self._inner.release()

    monkeypatch.setattr("consistency_checker.cli.main.make_embedder", fake_embedder)
    monkeypatch.setattr("consistency_checker.cli.main.make_judge", fake_judge)
    monkeypatch.setattr(
        "consistency_checker.cli.main.make_definition_checker", fake_definition_checker
    )
    monkeypatch.setattr(
        "consistency_checker.check.nli_checker.TransformerNliChecker", _FakeNliFactory
    )

    result = runner.invoke(
        app, ["check", "--org-scope", "--corpus", "default", "--config", str(cfg_path)]
    )
    assert result.exit_code == 0, result.stdout
    assert captured.get("org_scope_enabled") is True


def _write_two_org_corpus(corpus: Path) -> None:
    corpus.mkdir(exist_ok=True)
    (corpus / "acme.txt").write_text("Bylaws of Acme Foundation, Inc. Article I.")
    (corpus / "globex.txt").write_text("Charter of Globex LLC. Section 1.")


def _multi_org_fixture_extractor() -> Any:
    from consistency_checker.extract.atomic_facts import (
        FixtureExtractor,
        OrgIdentification,
    )

    return FixtureExtractor(
        fixtures={},
        org_fixtures={
            ("acme", "Bylaws of Acme"): OrgIdentification(
                label="Acme Foundation, Inc.", reason="org_found"
            ),
            ("globex", "Charter of Globex"): OrgIdentification(
                label="Globex LLC", reason="org_found"
            ),
        },
    )


def test_ingest_prints_corpus_warning_for_multi_org(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    _write_two_org_corpus(corpus)
    cfg_path = write_config(tmp_path, corpus)

    extractor = _multi_org_fixture_extractor()

    def fake_extractor(_cfg: Any) -> Any:
        return extractor

    def fake_embedder(_cfg: Any) -> Any:
        return HashEmbedder(dim=64)

    monkeypatch.setattr("consistency_checker.cli.main.make_extractor", fake_extractor)
    monkeypatch.setattr("consistency_checker.cli.main.make_embedder", fake_embedder)

    result = runner.invoke(
        app, ["ingest", str(corpus), "--config", str(cfg_path), "--corpus", "default"]
    )
    assert result.exit_code == 0, result.stdout
    assert "Corpus spans 2 organizations" in result.stdout
    # Default --no-org-scope: hint to enable --org-scope is shown.
    assert "--org-scope" in result.stdout


def test_ingest_no_warning_for_single_org(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from consistency_checker.extract.atomic_facts import (
        FixtureExtractor,
        OrgIdentification,
    )

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "acme.txt").write_text("Bylaws of Acme Foundation, Inc. Article I.")
    cfg_path = write_config(tmp_path, corpus)

    extractor = FixtureExtractor(
        fixtures={},
        org_fixtures={
            ("acme", "Bylaws of Acme"): OrgIdentification(
                label="Acme Foundation, Inc.", reason="org_found"
            ),
        },
    )

    def fake_extractor(_cfg: Any) -> Any:
        return extractor

    def fake_embedder(_cfg: Any) -> Any:
        return HashEmbedder(dim=64)

    monkeypatch.setattr("consistency_checker.cli.main.make_extractor", fake_extractor)
    monkeypatch.setattr("consistency_checker.cli.main.make_embedder", fake_embedder)

    result = runner.invoke(
        app, ["ingest", str(corpus), "--config", str(cfg_path), "--corpus", "default"]
    )
    assert result.exit_code == 0, result.stdout
    assert "Corpus spans" not in result.stdout


# --- eval -----------------------------------------------------------------


def _seed_reviewed_finding(cfg: Config) -> None:
    """Record one pair finding + matching reviewer verdict so eval has data."""
    from consistency_checker.check.gate import CandidatePair

    store = AssertionStore(cfg.db_path)
    store.migrate()
    _cid = store.get_or_create_corpus("test", "/test", "moonshot")
    doc_a = Document.from_content("A body.", source_path="a.txt", title="A")
    doc_b = Document.from_content("B body.", source_path="b.txt", title="B")
    store.add_document(doc_a, corpus_id=_cid)
    store.add_document(doc_b, corpus_id=_cid)
    a = Assertion.build(doc_a.doc_id, "Revenue grew 12%.")
    b = Assertion.build(doc_b.doc_id, "Revenue declined 5%.")
    store.add_assertions([a, b])
    logger = AuditLogger(store)
    rid = logger.begin_run()
    logger.record_finding(
        rid,
        candidate=CandidatePair(a=a, b=b, score=0.9),
        nli=NliResult.from_scores(p_contradiction=0.7, p_entailment=0.1, p_neutral=0.2),
        verdict=JudgeVerdict(
            assertion_a_id=a.assertion_id,
            assertion_b_id=b.assertion_id,
            verdict="contradiction",
            confidence=0.92,
            rationale="r",
        ),
    )
    logger.set_reviewer_verdict(
        pair_key=":".join(sorted([a.assertion_id, b.assertion_id])),
        detector_type="contradiction",
        verdict="confirmed",
    )
    store.close()


def test_eval_prints_precision_and_calibration(runner: CliRunner, tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_reviewed_finding(cfg)
    result = runner.invoke(app, ["eval", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.stdout
    assert "Per-detector precision" in result.stdout
    assert "contradiction" in result.stdout
    assert "100.0%" in result.stdout  # 1 confirmed, 0 false_positive
    assert "Calibration on 'contradiction'" in result.stdout


def test_eval_empty_db_does_not_error(runner: CliRunner, tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    # Store exists but no findings / verdicts.
    AssertionStore(cfg.db_path).migrate()
    result = runner.invoke(app, ["eval", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.stdout
    assert "No reviewed findings" in result.stdout


def test_eval_out_writes_csvs(runner: CliRunner, tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_reviewed_finding(cfg)
    out_dir = tmp_path / "eval_out"
    result = runner.invoke(app, ["eval", "--config", str(cfg_path), "--out", str(out_dir)])
    assert result.exit_code == 0, result.stdout
    written = sorted(p.name for p in out_dir.iterdir())
    assert any(name.startswith("cc_eval_precision_") for name in written)
    assert any(name.startswith("cc_eval_calibration_") for name in written)


def test_eval_detector_flag_switches_calibration_table(runner: CliRunner, tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_reviewed_finding(cfg)
    result = runner.invoke(
        app,
        ["eval", "--config", str(cfg_path), "--detector", "definition_inconsistency"],
    )
    assert result.exit_code == 0, result.stdout
    assert "definition_inconsistency" in result.stdout


def test_eval_rejects_unknown_detector(runner: CliRunner, tmp_path: Path) -> None:
    """Typo in --detector must fail loudly instead of silently rendering an empty table."""
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_reviewed_finding(cfg)
    result = runner.invoke(app, ["eval", "--config", str(cfg_path), "--detector", "contraddiction"])
    assert result.exit_code != 0
    assert "must be one of" in (result.stdout + result.stderr).lower()


def test_ingest_without_corpus_errors_in_non_tty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / "config.yml"
    cfg.write_text(f"corpus_dir: {tmp_path}\ndata_dir: {tmp_path}\n", encoding="utf-8")
    # Make sure no stdin TTY
    runner = CliRunner()
    res = runner.invoke(app, ["ingest", str(tmp_path), "--config", str(cfg)])
    # The Click/Typer error message embeds "--corpus is required"
    assert res.exit_code != 0
    out = strip_ansi((res.output or "") + str(res.exception or ""))
    assert "--corpus is required" in out


def test_ingest_with_corpus_persists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Mirror existing ingest test fixtures. Confirm corpus_id is set on every doc."""
    from consistency_checker.extract.atomic_facts import FixtureExtractor, OrgIdentification

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("Atkins bylaws text", encoding="utf-8")
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(
        f"corpus_dir: {tmp_path}\ndata_dir: {tmp_path}\njudge_provider: moonshot\n",
        encoding="utf-8",
    )
    fx = FixtureExtractor(
        {}, org_fixtures={("doc", "Atkins"): OrgIdentification("Atkins", "org_found")}
    )
    # cli.main does `from consistency_checker.pipeline import make_extractor`, so
    # the symbol lives in cli.main's namespace; patching pipeline.make_extractor
    # alone misses that binding.
    monkeypatch.setattr("consistency_checker.cli.main.make_extractor", lambda c: fx)

    runner = CliRunner()
    res = runner.invoke(
        app, ["ingest", str(tmp_path), "--config", str(cfg_path), "--corpus", "atkins"]
    )
    assert res.exit_code == 0, res.output

    store = AssertionStore(tmp_path / "assertions.db")
    store.migrate()
    names = [c.corpus_name for c in store.list_corpora()]
    assert "atkins" in names
    atkins_id = next(c.corpus_id for c in store.list_corpora() if c.corpus_name == "atkins")
    rows = store._conn.execute("SELECT corpus_id FROM documents").fetchall()
    assert rows and all(r[0] == atkins_id for r in rows)
    store.close()


# --- Task 8: --corpus required on check / estimate-cost / export / report ----


def test_check_without_corpus_errors_in_non_tty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / "config.yml"
    cfg.write_text(f"corpus_dir: {tmp_path}\ndata_dir: {tmp_path}\n", encoding="utf-8")
    runner = CliRunner()
    res = runner.invoke(app, ["check", "--config", str(cfg)])
    assert res.exit_code != 0
    out = strip_ansi((res.output or "") + str(res.exception or ""))
    assert "--corpus is required" in out


def test_check_with_corpus_passes_corpus_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--corpus atkins flows through to pipeline.check as corpus_id."""
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)

    # Re-seed under "atkins" so the corpus exists.
    store = AssertionStore(cfg.db_path)
    store.migrate()
    atkins_id = store.get_or_create_corpus("atkins", str(cfg.corpus_dir), "moonshot")
    store.close()

    captured: dict[str, Any] = {}

    def fake_check(_cfg: Any, *, store: Any, faiss_store: Any, corpus_id: str, **kw: Any) -> Any:
        captured["corpus_id"] = corpus_id
        from consistency_checker.pipeline import CheckResult

        return CheckResult(
            run_id="fake-run",
            n_assertions=0,
            n_pairs_gated=0,
            n_pairs_judged=0,
            n_findings=0,
        )

    monkeypatch.setattr("consistency_checker.cli.main.run_check", fake_check)

    def fake_embedder(_cfg: Any) -> Any:
        return HashEmbedder(dim=64)

    def fake_judge(_cfg: Any) -> Any:
        return FixtureJudge({})

    class _FakeNliFactory:
        def __init__(self, model_name: str) -> None:
            pass

        def score(self, premise: str, hypothesis: str) -> Any:
            from consistency_checker.check.nli_checker import NliResult

            return NliResult.from_scores(p_contradiction=0.0, p_entailment=1.0, p_neutral=0.0)

        def release(self) -> None:
            pass

    monkeypatch.setattr("consistency_checker.cli.main.make_embedder", fake_embedder)
    monkeypatch.setattr("consistency_checker.cli.main.make_judge", fake_judge)
    monkeypatch.setattr(
        "consistency_checker.check.nli_checker.TransformerNliChecker", _FakeNliFactory
    )

    runner = CliRunner()
    res = runner.invoke(app, ["check", "--config", str(cfg_path), "--corpus", "atkins"])
    assert res.exit_code == 0, res.output
    assert captured["corpus_id"] == atkins_id


def test_estimate_cost_without_corpus_errors_in_non_tty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from consistency_checker.index.faiss_store import FaissStore

    cfg = tmp_path / "config.yml"
    cfg.write_text(f"corpus_dir: {tmp_path}\ndata_dir: {tmp_path}\n", encoding="utf-8")
    # Seed the FAISS store so estimate-cost doesn't fail on the missing-index error first.
    # data_dir defaults to tmp_path, so faiss_path is tmp_path/assertions.faiss.
    store = AssertionStore(tmp_path / "assertions.db")
    store.migrate()
    store.close()
    fs = FaissStore.open_or_create(
        index_path=tmp_path / "assertions.faiss",
        id_map_path=tmp_path / "assertions.idmap.json",
        dim=64,
    )
    fs.save()
    runner = CliRunner()
    res = runner.invoke(app, ["estimate-cost", "--config", str(cfg)])
    assert res.exit_code != 0
    out = strip_ansi((res.output or "") + str(res.exception or ""))
    assert "--corpus is required" in out


def test_export_without_corpus_errors_in_non_tty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / "config.yml"
    cfg.write_text(f"corpus_dir: {tmp_path}\ndata_dir: {tmp_path}\n", encoding="utf-8")
    # data_dir defaults to tmp_path; db_path is tmp_path/assertions.db.
    store = AssertionStore(tmp_path / "assertions.db")
    store.migrate()
    store.close()
    runner = CliRunner()
    res = runner.invoke(app, ["export", "csv", "--config", str(cfg)])
    assert res.exit_code != 0
    out = strip_ansi((res.output or "") + str(res.exception or ""))
    assert "--corpus is required" in out


def test_export_with_corpus_scopes_output(tmp_path: Path) -> None:
    """export csv with --corpus only writes assertions from that corpus."""
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)

    store = AssertionStore(cfg.db_path)
    store.migrate()
    cid_a = store.get_or_create_corpus("corp_a", str(tmp_path), "moonshot")
    store.get_or_create_corpus("corp_b", str(tmp_path), "moonshot")
    doc_a = Document.from_content("Alpha text.", source_path="a.txt", title="A")
    store.add_document(doc_a, corpus_id=cid_a)
    from consistency_checker.extract.schema import Assertion

    store.add_assertions([Assertion.build(doc_a.doc_id, "Alpha assertion.")])
    store.close()

    out = tmp_path / "out.csv"
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["export", "csv", "--out", str(out), "--config", str(cfg_path), "--corpus", "corp_a"],
    )
    assert res.exit_code == 0, res.output
    text = out.read_text()
    assert "Alpha assertion." in text


def test_export_errors_on_unknown_corpus_name(tmp_path: Path) -> None:
    """export with an unknown --corpus name must error rather than silently exporting 0 rows."""
    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)
    _seed_existing_store(cfg)

    out = tmp_path / "out.csv"
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["export", "csv", "--out", str(out), "--config", str(cfg_path), "--corpus", "typo_name"],
    )
    assert res.exit_code != 0, "expected non-zero exit for unknown corpus name"


def test_report_infers_corpus_from_run_when_not_specified(tmp_path: Path) -> None:
    """report --run <id> without --corpus infers corpus from pipeline_runs.corpus_id."""
    from consistency_checker.audit.logger import AuditLogger

    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)

    store = AssertionStore(cfg.db_path)
    store.migrate()
    cid = store.get_or_create_corpus("myrun", str(tmp_path), "moonshot")
    logger = AuditLogger(store)
    rid = logger.begin_run(run_id="rpt_infer_test", corpus_id=cid)
    logger.end_run(rid, n_assertions=0, n_pairs_gated=0, n_pairs_judged=0)
    store.close()

    out = tmp_path / "report_inferred.md"
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["report", "--run", rid, "--out", str(out), "--config", str(cfg_path)],
    )
    assert res.exit_code == 0, res.output
    assert out.exists()


def _ocr_flag_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, config_ocr_line: str
) -> dict[str, Any]:
    """Shared setup for the three CLI --ocr/--no-ocr flag tests; returns the captured dict."""
    from consistency_checker.extract.atomic_facts import FixtureExtractor, OrgIdentification

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("Atkins bylaws text", encoding="utf-8")
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(
        f"corpus_dir: {tmp_path}\ndata_dir: {tmp_path}\n"
        f"judge_provider: moonshot\n{config_ocr_line}",
        encoding="utf-8",
    )
    fx = FixtureExtractor(
        {}, org_fixtures={("doc", "Atkins"): OrgIdentification("Atkins", "org_found")}
    )
    monkeypatch.setattr("consistency_checker.cli.main.make_extractor", lambda c: fx)

    captured: dict[str, Any] = {}

    def fake_ingest(config: Any, **kwargs: Any) -> Any:
        captured["ocr_enabled"] = config.ocr_enabled
        from consistency_checker.pipeline import IngestResult

        return IngestResult(n_documents=0, n_chunks=0, n_assertions=0, n_embedded=0)

    monkeypatch.setattr("consistency_checker.cli.main.run_ingest", fake_ingest)
    return captured


def test_ingest_no_ocr_flag_disables_ocr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``--no-ocr`` forces ``config.ocr_enabled=False`` regardless of config."""
    captured = _ocr_flag_harness(monkeypatch, tmp_path, config_ocr_line="")
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "ingest",
            str(tmp_path),
            "--config",
            str(tmp_path / "config.yml"),
            "--corpus",
            "atkins",
            "--no-ocr",
        ],
    )
    assert res.exit_code == 0, res.output
    assert captured["ocr_enabled"] is False


def test_ingest_ocr_flag_overrides_config_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--ocr`` forces on even when the config file disables it."""
    captured = _ocr_flag_harness(monkeypatch, tmp_path, config_ocr_line="ocr_enabled: false\n")
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "ingest",
            str(tmp_path),
            "--config",
            str(tmp_path / "config.yml"),
            "--corpus",
            "atkins",
            "--ocr",
        ],
    )
    assert res.exit_code == 0, res.output
    assert captured["ocr_enabled"] is True


def test_ingest_no_ocr_flag_omitted_respects_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No flag → config value wins. Was a bug pre-fix: --ocr default True silently overrode."""
    captured = _ocr_flag_harness(monkeypatch, tmp_path, config_ocr_line="ocr_enabled: false\n")
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["ingest", str(tmp_path), "--config", str(tmp_path / "config.yml"), "--corpus", "atkins"],
    )
    assert res.exit_code == 0, res.output
    assert captured["ocr_enabled"] is False


def test_report_errors_when_corpus_mismatches_run(tmp_path: Path) -> None:
    """report --corpus <other> that differs from run's corpus_id must error."""
    from consistency_checker.audit.logger import AuditLogger

    cfg_path = write_config(tmp_path, tmp_path / "corpus_unused")
    cfg = Config.from_yaml(cfg_path)

    store = AssertionStore(cfg.db_path)
    store.migrate()
    cid_a = store.get_or_create_corpus("corp_a", str(tmp_path), "moonshot")
    store.get_or_create_corpus("corp_b", str(tmp_path), "moonshot")
    logger = AuditLogger(store)
    rid = logger.begin_run(run_id="rpt_mismatch_test", corpus_id=cid_a)
    logger.end_run(rid, n_assertions=0, n_pairs_gated=0, n_pairs_judged=0)
    store.close()

    out = tmp_path / "report_mismatch.md"
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "report",
            "--run",
            rid,
            "--out",
            str(out),
            "--config",
            str(cfg_path),
            "--corpus",
            "corp_b",
        ],
    )
    assert res.exit_code != 0
    out_text = (res.output or "") + str(res.exception or "")
    assert "mismatch" in out_text.lower() or "corpus" in out_text.lower()
