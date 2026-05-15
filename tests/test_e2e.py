"""End-to-end smoke test.

Runs ingest → check → report through the public pipeline functions against a
five-document fixture corpus that contains:

- a planted clear contradiction (revenue grew 12% vs. declined 5%, same fiscal year),
- a near-contradiction the judge should call "uncertain" (year-mismatch on Beta),
- surrounding noise that must not produce any false-positive contradictions.

Two flavours are exercised:

- ``e2e_fixture`` — uses FixtureExtractor, HashEmbedder, FixtureNliChecker,
  and FixtureJudge so everything stays hermetic and CI-safe.
- ``live`` — requires ``ANTHROPIC_API_KEY``, downloads the NLI model, and
  hits the real APIs. Verifies the full stack still finds the planted pair.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from consistency_checker.audit.logger import AuditLogger
from consistency_checker.audit.report import render_report
from consistency_checker.check.llm_judge import FixtureJudge, JudgeVerdict
from consistency_checker.check.nli_checker import FixtureNliChecker, NliResult
from consistency_checker.config import Config
from consistency_checker.corpus.chunker import chunk_document
from consistency_checker.corpus.loader import load_corpus
from consistency_checker.extract.atomic_facts import FixtureExtractor
from consistency_checker.extract.schema import Assertion, hash_id
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.index.faiss_store import FaissStore
from consistency_checker.pipeline import check as run_check
from consistency_checker.pipeline import ingest as run_ingest
from tests.conftest import HashEmbedder

CORPUS_DIR = Path(__file__).parent / "fixtures" / "e2e_corpus"


def _write_config(tmp_path: Path, *, corpus_dir: Path) -> Path:
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(
        f"""
corpus_dir: {corpus_dir}
judge_provider: fixture
judge_model: test-model
data_dir: {tmp_path / "store"}
log_dir: {tmp_path / "logs"}
embedder_model: hash
nli_model: fixture
gate_top_k: 10
gate_similarity_threshold: -1.0
nli_contradiction_threshold: 0.0
""".strip()
    )
    return cfg_path


# Map each fixture file to the atomic assertions a FActScore extractor would produce.
# Keys are the chunk text the chunker emits; we'll resolve to chunk_ids in setup.
_FIXTURE_ASSERTIONS_BY_FILE: dict[str, list[str]] = {
    "01_alpha_annual.md": [
        "The Alpha division shipped two products in fiscal year 2025.",
        "Revenue from Alpha grew 12% year-over-year in fiscal 2025.",
        "Alpha team headcount remained flat throughout fiscal 2025.",
    ],
    "02_alpha_press.txt": [
        "Revenue from Alpha declined 5% in fiscal 2025.",
        "Customer satisfaction for Alpha remained at 4.7 out of 5.",
    ],
    "03_beta_brief.md": [
        "The Beta initiative began in fiscal year 2024.",
        "Beta initial customers were drawn from the enterprise segment.",
        "The Beta program operates across three regions.",
    ],
    "04_beta_history.txt": [
        "The Beta initiative was launched during fiscal 2023.",
        "Beta adoption grew steadily through two subsequent years.",
    ],
    "05_gamma_memo.md": [
        "The Gamma program coordinates marketing across Alpha and Beta.",
        "Marketing spend in fiscal 2025 totaled $12 million.",
        "No headcount changes are planned for fiscal 2026.",
    ],
}


def _build_fixtures(
    cfg: Config,
) -> tuple[
    dict[str, list[str]],
    dict[tuple[str, str], NliResult],
    dict[tuple[str, str], JudgeVerdict],
    tuple[str, str],
    tuple[str, str],
]:
    """Pre-compute chunk-id → assertion-texts, NLI fixtures, and judge fixtures.

    Returns also the canonical assertion-id pair for (a) the planted clear
    contradiction and (b) the near-contradiction whose verdict should be
    "uncertain" — the test asserts on both.
    """
    extractor_fixture: dict[str, list[str]] = {}
    text_to_doc_id: dict[str, str] = {}
    text_to_assertion: dict[str, Assertion] = {}

    for loaded in load_corpus(cfg.corpus_dir):
        file_name = Path(loaded.document.source_path).name
        chunks = chunk_document(
            loaded,
            max_chars=cfg.chunk_max_chars,
            overlap_chars=cfg.chunk_overlap_chars,
        )
        assertion_texts = _FIXTURE_ASSERTIONS_BY_FILE.get(file_name, [])
        # Spread the assertions evenly across the chunks; in this fixture each
        # file has a single chunk so they all bind to one chunk_id.
        if not chunks:
            continue
        primary_chunk = chunks[0]
        extractor_fixture[primary_chunk.chunk_id] = assertion_texts
        for text in assertion_texts:
            text_to_doc_id[text] = loaded.document.doc_id
            text_to_assertion[text] = Assertion.build(
                loaded.document.doc_id,
                text,
                chunk_id=primary_chunk.chunk_id,
                char_start=primary_chunk.char_start,
                char_end=primary_chunk.char_end,
            )

    # Planted clear contradiction:
    clear_a_text = "Revenue from Alpha grew 12% year-over-year in fiscal 2025."
    clear_b_text = "Revenue from Alpha declined 5% in fiscal 2025."

    # Near-contradiction (judge should rate this uncertain — different year
    # framings about the same initiative; resolvable only with extra context):
    near_a_text = "The Beta initiative began in fiscal year 2024."
    near_b_text = "The Beta initiative was launched during fiscal 2023."

    clear_a = text_to_assertion[clear_a_text]
    clear_b = text_to_assertion[clear_b_text]
    near_a = text_to_assertion[near_a_text]
    near_b = text_to_assertion[near_b_text]

    nli_fixtures: dict[tuple[str, str], NliResult] = {
        (clear_a_text, clear_b_text): NliResult.from_scores(
            p_contradiction=0.86, p_entailment=0.04, p_neutral=0.10
        ),
        (clear_b_text, clear_a_text): NliResult.from_scores(
            p_contradiction=0.81, p_entailment=0.04, p_neutral=0.15
        ),
        (near_a_text, near_b_text): NliResult.from_scores(
            p_contradiction=0.55, p_entailment=0.10, p_neutral=0.35
        ),
        (near_b_text, near_a_text): NliResult.from_scores(
            p_contradiction=0.52, p_entailment=0.10, p_neutral=0.38
        ),
    }

    def _canonical(a: Assertion, b: Assertion) -> tuple[str, str]:
        return (
            min(a.assertion_id, b.assertion_id),
            max(a.assertion_id, b.assertion_id),
        )

    clear_key = _canonical(clear_a, clear_b)
    near_key = _canonical(near_a, near_b)
    judge_fixtures: dict[tuple[str, str], JudgeVerdict] = {
        clear_key: JudgeVerdict(
            assertion_a_id=clear_key[0],
            assertion_b_id=clear_key[1],
            verdict="contradiction",
            confidence=0.91,
            rationale="Opposite revenue signs at the same fiscal-year scope.",
            evidence_spans=["grew 12%", "declined 5%"],
        ),
        near_key: JudgeVerdict(
            assertion_a_id=near_key[0],
            assertion_b_id=near_key[1],
            verdict="uncertain",
            confidence=0.45,
            rationale="The two assertions disagree on the Beta start year but neither "
            "rules out the other given the document scope provided.",
            evidence_spans=["fiscal year 2024", "fiscal 2023"],
        ),
    }

    return extractor_fixture, nli_fixtures, judge_fixtures, clear_key, near_key


def _hash_id_for(content: str) -> str:
    return hash_id(content)


@pytest.mark.e2e_fixture
def test_end_to_end_mixed_format_corpus(
    tmp_path: Path,
    sample_pdf_path: Path,
    sample_docx_path: Path,
) -> None:
    """Pipeline ingests a corpus mixing .txt, .md, .pdf, .docx without crashing."""
    corpus = tmp_path / "mixed_corpus"
    corpus.mkdir()
    (corpus / "01_plaintext.txt").write_text("A first plaintext fact about widgets.")
    (corpus / "02_markdown.md").write_text("# Heading\n\nA markdown fact about gadgets.\n")
    (corpus / "03_pdf.pdf").write_bytes(sample_pdf_path.read_bytes())
    (corpus / "04_docx.docx").write_bytes(sample_docx_path.read_bytes())

    cfg_path = _write_config(tmp_path, corpus_dir=corpus)
    cfg = Config.from_yaml(cfg_path)

    # Pre-walk the corpus to discover chunk ids, then build a FixtureExtractor
    # that emits one synthetic assertion per chunk so downstream stages have
    # something to operate on.
    extractor_fixture: dict[str, list[str]] = {}
    for loaded in load_corpus(corpus):
        for chunk in chunk_document(
            loaded,
            max_chars=cfg.chunk_max_chars,
            overlap_chars=cfg.chunk_overlap_chars,
        ):
            ext = Path(loaded.document.source_path).suffix
            extractor_fixture[chunk.chunk_id] = [f"Extracted from {ext}: {chunk.text[:60]}"]

    embedder = HashEmbedder(dim=64)
    store = AssertionStore(cfg.db_path)
    store.migrate()
    faiss_store = FaissStore.open_or_create(
        index_path=cfg.faiss_path,
        id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
        dim=embedder.dim,
    )

    ingest_result = run_ingest(
        cfg,
        store=store,
        faiss_store=faiss_store,
        extractor=FixtureExtractor(extractor_fixture),
        embedder=embedder,
    )
    assert ingest_result.n_documents == 4
    assert ingest_result.n_assertions >= 4
    assert ingest_result.n_embedded == ingest_result.n_assertions

    audit_logger = AuditLogger(store)
    run_id = audit_logger.begin_run()
    check_result = run_check(
        cfg,
        store=store,
        faiss_store=faiss_store,
        nli_checker=FixtureNliChecker({}),
        judge=FixtureJudge({}),
        audit_logger=audit_logger,
        run_id=run_id,
    )
    # No fixtures wired for any pair → FixtureJudge falls back to "uncertain",
    # so the mixed-format corpus must produce zero contradictions but every
    # other stage (ingest, embed, gate, NLI, judge, audit) must complete.
    assert check_result.n_findings == 0
    assert check_result.n_pairs_judged > 0

    store.close()


@pytest.mark.e2e_fixture
def test_end_to_end_pipeline_fixture(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path, corpus_dir=CORPUS_DIR)
    cfg = Config.from_yaml(cfg_path)

    extractor_fixture, nli_fixtures, judge_fixtures, clear_key, _near_key = _build_fixtures(cfg)

    embedder = HashEmbedder(dim=64)
    store = AssertionStore(cfg.db_path)
    store.migrate()
    faiss_store = FaissStore.open_or_create(
        index_path=cfg.faiss_path,
        id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
        dim=embedder.dim,
    )

    # 1. Ingest.
    ingest_result = run_ingest(
        cfg,
        store=store,
        faiss_store=faiss_store,
        extractor=FixtureExtractor(extractor_fixture),
        embedder=embedder,
    )
    assert ingest_result.n_documents == 5
    expected_n_assertions = sum(len(v) for v in _FIXTURE_ASSERTIONS_BY_FILE.values())
    assert ingest_result.n_assertions == expected_n_assertions
    assert ingest_result.n_embedded == expected_n_assertions

    # 2. Check.
    audit_logger = AuditLogger(store)
    run_id = audit_logger.begin_run()
    check_result = run_check(
        cfg,
        store=store,
        faiss_store=faiss_store,
        nli_checker=FixtureNliChecker(nli_fixtures),
        judge=FixtureJudge(judge_fixtures),
        audit_logger=audit_logger,
        run_id=run_id,
    )

    # The planted contradiction must surface; the near-contradiction must not
    # be counted as a contradiction (it's logged as uncertain). The revenue
    # sign-flip is now resolved deterministically by the numeric short-circuit
    # (ADR-0005), so the finding's verdict label is numeric_short_circuit
    # rather than contradiction — but it still counts in n_findings.
    assert check_result.n_findings == 1, (
        f"expected exactly one contradiction; got {check_result.n_findings} "
        f"(n_pairs_judged={check_result.n_pairs_judged})"
    )

    # 3. Report.
    contradiction_findings = [
        *audit_logger.iter_findings(run_id=check_result.run_id, verdict="contradiction"),
        *audit_logger.iter_findings(run_id=check_result.run_id, verdict="numeric_short_circuit"),
    ]
    assert len(contradiction_findings) == 1
    planted = contradiction_findings[0]
    assert planted.assertion_a_id == clear_key[0]
    assert planted.assertion_b_id == clear_key[1]
    assert planted.judge_confidence is not None
    assert planted.judge_confidence > 0.7, (
        f"planted contradiction confidence {planted.judge_confidence} should be > 0.7"
    )

    report_text = render_report(store, audit_logger, run_id=check_result.run_id)
    # The revenue pair short-circuits to a deterministic numeric verdict
    # (ADR-0005), so the rationale comes from the short-circuit, not the
    # fixture judge. Accept either marker — the LLM-derived rationale would
    # appear when the short-circuit predicate doesn't fire.
    assert ("Numeric short-circuit" in report_text) or ("Opposite revenue signs" in report_text)
    # The near-contradiction was uncertain; it must not appear in the report.
    assert "fiscal 2023" not in report_text

    # Uncertain findings still live in the audit DB for replay / threshold tuning.
    uncertain = list(audit_logger.iter_findings(run_id=check_result.run_id, verdict="uncertain"))
    assert len(uncertain) >= 1

    store.close()


@pytest.mark.live
def test_end_to_end_pipeline_live(tmp_path: Path) -> None:
    """Full stack: real Anthropic judge + real NLI model on the fixture corpus."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    if os.environ.get("CC_SKIP_HF_DOWNLOAD") == "1":
        pytest.skip("HF download skipped by env flag")

    from consistency_checker.check.llm_judge import LLMJudge
    from consistency_checker.check.nli_checker import TransformerNliChecker
    from consistency_checker.check.providers.anthropic import AnthropicProvider
    from consistency_checker.extract.atomic_facts import AnthropicExtractor
    from consistency_checker.index.embedder import SentenceTransformerEmbedder

    cfg_path = _write_config(tmp_path, corpus_dir=CORPUS_DIR)
    cfg = Config.from_yaml(cfg_path).model_copy(
        update={
            "judge_provider": "anthropic",
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "nli_model": ("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"),
            "gate_similarity_threshold": 0.5,
            "nli_contradiction_threshold": 0.5,
        }
    )

    embedder = SentenceTransformerEmbedder(model_name=cfg.embedder_model)
    store = AssertionStore(cfg.db_path)
    store.migrate()
    faiss_store = FaissStore.open_or_create(
        index_path=cfg.faiss_path,
        id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
        dim=embedder.dim,
    )

    run_ingest(
        cfg,
        store=store,
        faiss_store=faiss_store,
        extractor=AnthropicExtractor(model=cfg.judge_model),
        embedder=embedder,
    )

    audit_logger = AuditLogger(store)
    run_id = audit_logger.begin_run()
    check_result = run_check(
        cfg,
        store=store,
        faiss_store=faiss_store,
        nli_checker=TransformerNliChecker(model_name=cfg.nli_model),
        judge=LLMJudge(AnthropicProvider(model=cfg.judge_model)),
        audit_logger=audit_logger,
        run_id=run_id,
    )
    assert check_result.n_findings >= 1
    contradictions = list(
        audit_logger.iter_findings(run_id=check_result.run_id, verdict="contradiction")
    )
    # Look for any finding that quotes both the "grew" and "declined" sides.
    found_revenue_pair = False
    for f in contradictions:
        spans_joined = " ".join(f.evidence_spans).lower()
        if "grew" in spans_joined and "declined" in spans_joined:
            found_revenue_pair = True
            assert f.judge_confidence is not None
            assert f.judge_confidence > 0.7
            break
    assert found_revenue_pair, "Expected the live judge to surface the Alpha revenue contradiction"
    store.close()
