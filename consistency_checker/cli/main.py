"""Typer-based ``consistency-check`` CLI.

Thin wrapper around :mod:`consistency_checker.pipeline`. Subcommands:

- ``ingest <corpus_dir>`` — load + chunk + extract + embed.
- ``check`` — Stage A + Stage B + audit.
- ``estimate-cost`` — pre-flight API-spend ceiling for a check run.
- ``report`` — render markdown from the latest run.
- ``export csv|jsonl`` — emit assertions for downstream tooling.
- ``store stats|rebuild-index`` — store maintenance.
"""

from __future__ import annotations

from pathlib import Path

import typer

from consistency_checker.audit.eval import (
    DEFAULT_CALIBRATION_BINS,
    compute_calibration,
    compute_detector_precision,
    eval_filename,
    format_calibration_table,
    format_precision_table,
    iter_eval_rows,
    write_calibration_csv,
    write_precision_csv,
)
from consistency_checker.audit.logger import AuditLogger
from consistency_checker.audit.naming import (
    export_csv_filename,
    export_jsonl_filename,
    report_filename,
)
from consistency_checker.audit.report import render_report
from consistency_checker.check.nli_checker import NliChecker
from consistency_checker.cli.corpus_prompt import resolve_corpus
from consistency_checker.cli.warnings import (
    render_corpus_warning,
    render_fragmentation_warning,
    render_identification_failure_notice,
    summarize_buckets,
)
from consistency_checker.config import Config, load_local_env
from consistency_checker.corpus.loader import load_path
from consistency_checker.index.assertion_store import AssertionStore, CrossCorpusDocumentError
from consistency_checker.index.faiss_store import FaissStore
from consistency_checker.logging_setup import configure as configure_logging
from consistency_checker.pipeline import (
    CostCeilingExceeded,
    make_definition_checker,
    make_embedder,
    make_extractor,
    make_judge,
    make_multi_party_judge,
    rebuild_faiss,
)
from consistency_checker.pipeline import (
    check as run_check,
)
from consistency_checker.pipeline import (
    estimate_cost as run_estimate_cost,
)
from consistency_checker.pipeline import (
    ingest as run_ingest,
)

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Consistency checker CLI.")
store_app = typer.Typer(help="Assertion-store maintenance commands.")
app.add_typer(store_app, name="store")
corpus_app = typer.Typer(help="Inspect or maintain corpora.")
app.add_typer(corpus_app, name="corpus")


@app.callback()
def _bootstrap() -> None:
    """Load secrets from a local ``.env`` before any subcommand runs."""
    load_local_env()


def _load_config(config_path: Path | None) -> Config:
    path = config_path or Path("config.yml")
    if not path.exists():
        raise typer.BadParameter(f"Config file {path} not found.")
    return Config.from_yaml(path)


def _open_store(config: Config) -> AssertionStore:
    store = AssertionStore(config.db_path)
    store.migrate()
    return store


def _open_faiss(config: Config, dim: int) -> FaissStore:
    return FaissStore.open_or_create(
        index_path=config.faiss_path,
        id_map_path=config.faiss_path.with_suffix(".idmap.json"),
        dim=dim,
    )


_CORPUS_PROVIDER_CHECK_ALLOWED = ("moonshot", "anthropic")


def _provider_for_corpus(judge_provider: str) -> str:
    """Clamp config.judge_provider to a value the corpora.judge_provider CHECK
    constraint accepts. Why: schema allows only ('moonshot', 'anthropic'); see
    ADR-0013 'Consequences' — future spec may widen the CHECK.
    """
    return judge_provider if judge_provider in _CORPUS_PROVIDER_CHECK_ALLOWED else "moonshot"


def _warn_if_model_download_needed(model_name: str) -> None:
    """Print a one-line warning if the HF model isn't in the local cache."""
    try:
        from huggingface_hub import try_to_load_from_cache

        result = try_to_load_from_cache(model_name, "config.json")
        if result is None:
            typer.echo(
                f"Note: first run will download ~440 MB ({model_name}). "
                "This may take a few minutes — the tool is not hung."
            )
    except Exception:
        pass  # huggingface_hub not available or lookup failed — skip silently


# Estimated peak RSS for a typical check() run: mpnet embedder + DeBERTa-large
# NLI + python/sqlite/faiss baseline. Used to warn when MemAvailable is low.
_NLI_PEAK_ESTIMATE_MB = 2500


def _available_memory_mb() -> int | None:
    """Return MemAvailable in MB via psutil, or None if psutil isn't installed."""
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.virtual_memory().available // (1024 * 1024))


def _preflight_memory(cfg: Config) -> None:
    """Abort or warn before the NLI model is loaded.

    - When ``cfg.max_memory_mb`` is set and exceeds MemAvailable, abort.
    - When MemAvailable is below the NLI peak estimate, print a soft warning.
    - When psutil isn't installed, skip silently — the check is best-effort.
    """
    available_mb = _available_memory_mb()
    if available_mb is None:
        return
    if cfg.max_memory_mb is not None and available_mb < cfg.max_memory_mb:
        raise typer.BadParameter(
            f"Available memory {available_mb} MB is below max_memory_mb="
            f"{cfg.max_memory_mb}. Close other processes, lower the threshold, "
            "or use a smaller `nli_model` (e.g. DeBERTa-v3-base)."
        )
    if available_mb < _NLI_PEAK_ESTIMATE_MB:
        typer.echo(
            f"Warning: {available_mb} MB available; the NLI + embedder stack "
            f"typically needs ~{_NLI_PEAK_ESTIMATE_MB} MB. The check may OOM. "
            "Consider closing other processes or using a smaller `nli_model`."
        )


# --- corpus warnings -------------------------------------------------------


def _emit_corpus_warnings(store: AssertionStore, cfg: Config) -> None:
    """Print Task 12 corpus warnings using a single pass over documents."""
    docs = list(store.iter_documents())
    rows = [(d.doc_id, d.org_label) for d in docs]
    summary = summarize_buckets(rows)

    warn = render_corpus_warning(
        summary.known,
        summary.unknown_count,
        scope_enabled=cfg.org_scope_enabled,
    )
    if warn:
        typer.echo(warn)

    frag = render_fragmentation_warning(summary.known)
    if frag:
        typer.echo(frag)

    failures = sum(1 for d in docs if d.org_reason in {"llm_error", "truncated"})
    notice = render_identification_failure_notice(failures=failures, total=len(rows))
    if notice:
        typer.echo(notice)


# --- ingest -----------------------------------------------------------------


@app.command(help="Load documents, chunk, extract atomic facts, and embed.")
def ingest(
    corpus_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    config: Path = typer.Option(Path("config.yml"), "--config", "-c"),
    corpus: str | None = typer.Option(
        None,
        "--corpus",
        help="Corpus name. Required (interactive picker on TTY).",
    ),
    org_scope: bool = typer.Option(
        False,
        "--org-scope/--no-org-scope",
        help="Suppress cross-organization definition pairs and write them to the audit trail.",
    ),
    ocr: bool | None = typer.Option(
        None,
        "--ocr/--no-ocr",
        help=(
            "Auto-escalate scanned PDFs to OCR (hi_res) when fast extraction "
            "is empty. Pass --no-ocr to force off, --ocr to force on, or omit "
            "to use the config's ocr_enabled value."
        ),
    ),
) -> None:
    cfg = _load_config(config)
    cfg = cfg.model_copy(update={"corpus_dir": corpus_dir, "org_scope_enabled": org_scope})
    if ocr is not None:
        cfg = cfg.model_copy(update={"ocr_enabled": ocr})
    configure_logging(cfg.log_dir)
    # Resolve --corpus first so a missing flag fails fast without doing the
    # expensive extractor/embedder bootstrap (sentence-transformers download,
    # Moonshot key check, etc.).
    store = _open_store(cfg)
    provider_for_corpus = _provider_for_corpus(cfg.judge_provider)
    corpus_id = resolve_corpus(store, corpus, str(cfg.corpus_dir), provider_for_corpus)
    extractor = make_extractor(cfg)
    embedder = make_embedder(cfg)
    faiss_store = _open_faiss(cfg, dim=embedder.dim)
    try:
        result = run_ingest(
            cfg,
            store=store,
            faiss_store=faiss_store,
            extractor=extractor,
            embedder=embedder,
            corpus_id=corpus_id,
        )
    except CrossCorpusDocumentError as err:
        names_by_id = {c.corpus_id: c.corpus_name for c in store.list_corpora()}
        existing_name = names_by_id.get(err.existing_corpus_id, err.existing_corpus_id)
        requested_name = names_by_id.get(err.requested_corpus_id, err.requested_corpus_id)
        store.close()
        typer.echo(
            f"Error: document {err.doc_id} already exists under corpus "
            f"'{existing_name}' and cannot be re-ingested into corpus "
            f"'{requested_name}'. Either edit the source file so the content "
            f"hash changes, or move the existing doc with "
            f"`consistency-check corpus reassign` (note: that only relabels "
            f"the doc, it does not re-extract its assertions).",
            err=True,
        )
        raise typer.Exit(code=2) from err
    _emit_corpus_warnings(store, cfg)
    store.close()
    typer.echo(
        f"Ingested {result.n_documents} documents, {result.n_chunks} chunks, "
        f"{result.n_assertions} assertions, {result.n_embedded} newly embedded."
    )


# --- check ------------------------------------------------------------------


@app.command(help="Run the Stage A + Stage B contradiction scan against the indexed corpus.")
def check(
    config: Path = typer.Option(Path("config.yml"), "--config", "-c"),
    corpus: str | None = typer.Option(
        None,
        "--corpus",
        help="Corpus name. Required (interactive picker on TTY).",
    ),
    deep: bool = typer.Option(
        False,
        "--deep",
        help="Enable the multi-document conditional contradiction pass (ADR-0006).",
    ),
    no_definitions: bool = typer.Option(
        False,
        "--no-definitions",
        help="Skip the definition-inconsistency stage (ADR-0009). Default: enabled.",
    ),
    org_scope: bool = typer.Option(
        False,
        "--org-scope/--no-org-scope",
        help="Suppress cross-organization definition pairs and write them to the audit trail.",
    ),
    pairwise: bool | None = typer.Option(
        None,
        "--pairwise/--no-pairwise",
        help=(
            "Run the pairwise contradiction detector (NLI gate + LLM judge). "
            "Pass --pairwise to force on, --no-pairwise to force off, or omit "
            "to use the config's pairwise_enabled value (shipped default: False "
            "— see ADR-0015 for the eval data behind that default)."
        ),
    ),
    max_cost: float | None = typer.Option(
        None,
        "--max-cost",
        min=0,
        help=(
            "Hard ceiling on estimated API spend in USD. Aborts the run before "
            "any judge/NLI bootstrap if estimate-cost's high-end projection "
            "exceeds this value. Conservative: false-positive aborts (a safe "
            "run rejected) but never false-negatives. Omit to disable (or set "
            "via Config.max_cost_usd). See ADR-0016."
        ),
    ),
) -> None:
    cfg = _load_config(config)
    if deep:
        cfg = cfg.model_copy(update={"enable_multi_party": True})
    cfg = cfg.model_copy(update={"org_scope_enabled": org_scope})
    if pairwise is not None:
        cfg = cfg.model_copy(update={"pairwise_enabled": pairwise})
    if max_cost is not None:
        cfg = cfg.model_copy(update={"max_cost_usd": max_cost})
    if cfg.enable_multi_party and not cfg.pairwise_enabled:
        raise typer.BadParameter(
            "--deep requires the pairwise gate output. Re-run with --pairwise."
        )
    configure_logging(cfg.log_dir)
    # Resolve --corpus first so a missing flag fails fast without the
    # sentence-transformers model load and OOM pre-flight.
    store = _open_store(cfg)
    provider_for_corpus = _provider_for_corpus(cfg.judge_provider)
    corpus_id = resolve_corpus(store, corpus, str(cfg.corpus_dir), provider_for_corpus)
    # OOM pre-flight gates the NLI download specifically, but the embedder
    # load itself is ~400 MB sentence-transformers — run the check before that
    # too, so a tight-memory machine bails before any model touches RAM. Skip
    # entirely when pairwise is off (no NLI to protect).
    if cfg.pairwise_enabled:
        _preflight_memory(cfg)
    embedder = make_embedder(cfg)
    faiss_store = _open_faiss(cfg, dim=embedder.dim)
    _emit_corpus_warnings(store, cfg)
    audit_logger = AuditLogger(store)

    nli: NliChecker | None = None
    if cfg.pairwise_enabled:
        from consistency_checker.check.nli_checker import TransformerNliChecker

        _warn_if_model_download_needed(cfg.nli_model or TransformerNliChecker.DEFAULT_MODEL)
        nli = TransformerNliChecker(model_name=cfg.nli_model)
    judge = make_judge(cfg)
    multi_party = make_multi_party_judge(cfg) if cfg.enable_multi_party else None
    definition_checker = None if no_definitions else make_definition_checker(cfg)

    run_id = audit_logger.begin_run(
        corpus_id=corpus_id,
        config={
            "embedder_model": cfg.embedder_model,
            "nli_model": cfg.nli_model,
            "judge_provider": cfg.judge_provider,
            "judge_model": cfg.judge_model,
            "nli_contradiction_threshold": cfg.nli_contradiction_threshold,
            "gate_top_k": cfg.gate_top_k,
            "gate_similarity_threshold": cfg.gate_similarity_threshold,
            "enable_multi_party": cfg.enable_multi_party,
            "max_triangles_per_run": cfg.max_triangles_per_run,
            "definitions_enabled": not no_definitions,
            "pairwise_enabled": cfg.pairwise_enabled,
            "max_cost_usd": cfg.max_cost_usd,
        },
    )
    try:
        result = run_check(
            cfg,
            store=store,
            faiss_store=faiss_store,
            nli_checker=nli,
            judge=judge,
            audit_logger=audit_logger,
            multi_party_judge=multi_party,
            definition_checker=definition_checker,
            run_id=run_id,
            corpus_id=corpus_id,
        )
    except CostCeilingExceeded as exc:
        store.close()
        typer.echo(
            f"Estimated cost ${exc.estimated_high:.4f} exceeds --max-cost "
            f"${exc.ceiling:.4f}. Options: --no-pairwise, --no-definitions, "
            f"narrow the corpus with a more specific --corpus, or raise the "
            f"ceiling.",
            err=True,
        )
        raise typer.Exit(code=2) from exc
    store.close()
    deep_suffix = (
        f" / {result.n_triangles_judged} triangles / {result.n_multi_party_findings} multi-party"
        if cfg.enable_multi_party
        else ""
    )
    def_suffix = (
        f" / {result.n_definition_pairs_judged} def-pairs / "
        f"{result.n_definition_findings} def-findings"
        if not no_definitions
        else ""
    )
    pairwise_summary = (
        f"{result.n_pairs_gated} gated / {result.n_pairs_judged} judged / "
        f"{result.n_findings} contradictions"
        if cfg.pairwise_enabled
        else "pairwise=off"
    )
    typer.echo(f"Run {result.run_id} — {pairwise_summary}{deep_suffix}{def_suffix}")


# --- estimate-cost ----------------------------------------------------------


@app.command(
    "estimate-cost",
    help="Estimate API spend for a check run without making any LLM or NLI calls.",
)
def estimate_cost(
    config: Path = typer.Option(Path("config.yml"), "--config", "-c"),
    corpus: str | None = typer.Option(
        None,
        "--corpus",
        help="Corpus name. Required (interactive picker on TTY).",
    ),
    per_call_low: float | None = typer.Option(
        None,
        "--per-call-low",
        help=(
            "Low end of per-judge-call cost in USD. Default: provider-specific "
            "(anthropic/openai $0.003, moonshot $0.0001, fixture $0.000)."
        ),
    ),
    per_call_high: float | None = typer.Option(
        None,
        "--per-call-high",
        help=(
            "High end of per-judge-call cost in USD. Default: provider-specific "
            "(anthropic/openai $0.010, moonshot $0.001, fixture $0.000)."
        ),
    ),
    pairwise: bool | None = typer.Option(
        None,
        "--pairwise/--no-pairwise",
        help=(
            "Include the pairwise candidate-pair count. Pass --pairwise to "
            "force on, --no-pairwise to force off, or omit to use the config's "
            "pairwise_enabled value. When off, Stage A gated pairs is 0."
        ),
    ),
) -> None:
    cfg = _load_config(config)
    if pairwise is not None:
        cfg = cfg.model_copy(update={"pairwise_enabled": pairwise})
    store = _open_store(cfg)
    # `dim=None` reads the dimension from the existing FAISS index. This
    # skips the sentence-transformers model load that `make_embedder` would
    # trigger — a multi-GB download on a cold cache that would defeat the
    # whole "fast pre-flight" point of this command.
    try:
        faiss_store = FaissStore.open_or_create(
            index_path=cfg.faiss_path,
            id_map_path=cfg.faiss_path.with_suffix(".idmap.json"),
            dim=None,
        )
    except ValueError as exc:
        store.close()
        raise typer.BadParameter(
            f"{exc} Run `consistency-check ingest <corpus_dir>` first."
        ) from exc
    provider_for_corpus = _provider_for_corpus(cfg.judge_provider)
    corpus_id = resolve_corpus(store, corpus, str(cfg.corpus_dir), provider_for_corpus)
    est = run_estimate_cost(
        cfg,
        store=store,
        faiss_store=faiss_store,
        per_call_low=per_call_low,
        per_call_high=per_call_high,
        corpus_id=corpus_id,
    )
    store.close()
    if cfg.pairwise_enabled:
        footnote = (
            "   varies by model + prompt size. This is a CEILING - many gate-pass\n"
            "   pairs are filtered by NLI before reaching the judge, so real spend\n"
            "   is usually 30-70% lower.)"
        )
    else:
        footnote = (
            "   Pairwise detector disabled — Stage A count is 0. Pass --pairwise to include it.)"
        )
    typer.echo(
        "Run cost estimate\n"
        f"  Assertions in store:          {est.n_assertions}\n"
        f"  Stage A - gated pairs:        {est.n_candidate_pairs}\n"
        f"  Definition pairs to judge:    {est.n_definition_pairs}\n"
        f"  Total judge calls (ceiling):  {est.judge_calls_ceiling}\n"
        "\n"
        f"  Estimated API cost:  ${est.est_cost_low:.2f} to ${est.est_cost_high:.2f}\n"
        f"  (assumes ${est.per_call_low:.4f} to ${est.per_call_high:.4f} per judge call;\n"
        + footnote
    )


# --- report -----------------------------------------------------------------


@app.command(help="Render a markdown report for a run (defaults to the most recent).")
def report(
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help=(
            "Output markdown path. Omit to write to "
            "<data_dir>/reports/cc_report_<timestamp>_<run_id_short>.md."
        ),
    ),
    run_id: str | None = typer.Option(None, "--run", help="Run id; default: most recent."),
    min_confidence: float = typer.Option(0.0, "--min-confidence", min=0.0, max=1.0),
    corpus: str | None = typer.Option(
        None,
        "--corpus",
        help=(
            "Optional corpus name. When omitted, inferred from the run's corpus_id. "
            "Providing a name that does not match the run's corpus is an error."
        ),
    ),
    config: Path = typer.Option(Path("config.yml"), "--config", "-c"),
) -> None:
    cfg = _load_config(config)
    store = _open_store(cfg)
    audit_logger = AuditLogger(store)

    target_run = run_id
    if target_run is None:
        recent = audit_logger.most_recent_run()
        if recent is None:
            typer.echo("No runs found in the audit store.", err=True)
            raise typer.Exit(code=2)
        target_run = recent.run_id

    run_row = audit_logger.get_run(target_run)
    run_corpus_id: str | None = run_row.corpus_id if run_row is not None else None

    if corpus is not None:
        # Resolve the explicitly-supplied corpus name to an id and validate.
        existing = store.list_corpora()
        match = next((c for c in existing if c.corpus_name == corpus), None)
        if match is None:
            store.close()
            available = ", ".join(c.corpus_name for c in existing) or "<none>"
            raise typer.BadParameter(f"--corpus {corpus!r} not found (available: {available})")
        if run_corpus_id is not None and match.corpus_id != run_corpus_id:
            run_corpus_name = next(
                (c.corpus_name for c in existing if c.corpus_id == run_corpus_id), run_corpus_id
            )
            store.close()
            raise typer.BadParameter(
                f"--corpus mismatch: run {target_run!r} belongs to corpus "
                f"{run_corpus_name!r} but --corpus {corpus!r} was supplied."
            )

    text = render_report(store, audit_logger, run_id=target_run, min_confidence=min_confidence)
    if out is None:
        started = run_row.started_at if run_row is not None else None
        out = cfg.data_dir / "reports" / report_filename(target_run, started_at=started)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    store.close()
    typer.echo(f"Wrote report for run {target_run} to {out}")


# --- export -----------------------------------------------------------------


@app.command(help="Export assertions to CSV or JSONL.")
def export(
    fmt: str = typer.Argument(..., metavar="FORMAT", help="csv | jsonl"),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help=("Output path. Omit to write to <data_dir>/reports/cc_assertions_<timestamp>.<ext>."),
    ),
    corpus: str | None = typer.Option(
        None,
        "--corpus",
        help="Corpus name. Required (interactive picker on TTY).",
    ),
    config: Path = typer.Option(Path("config.yml"), "--config", "-c"),
) -> None:
    cfg = _load_config(config)
    store = _open_store(cfg)
    provider_for_corpus = _provider_for_corpus(cfg.judge_provider)
    corpus_id = resolve_corpus(
        store, corpus, str(cfg.corpus_dir), provider_for_corpus, allow_create=False
    )
    if fmt == "csv":
        if out is None:
            out = cfg.data_dir / "reports" / export_csv_filename()
        store.export_csv(out, corpus_id=corpus_id)
    elif fmt == "jsonl":
        if out is None:
            out = cfg.data_dir / "reports" / export_jsonl_filename()
        store.export_jsonl(out, corpus_id=corpus_id)
    else:
        store.close()
        raise typer.BadParameter("FORMAT must be 'csv' or 'jsonl'")
    store.close()
    typer.echo(f"Exported assertions to {out}")


# --- store maintenance ------------------------------------------------------


@app.command(
    help=(
        "Mine reviewer_verdicts for per-detector precision + judge-confidence "
        "calibration. Reads the audit DB only; no LLM calls."
    )
)
def eval(
    config: Path = typer.Option(Path("config.yml"), "--config", "-c"),
    detector: str = typer.Option(
        "contradiction",
        "--detector",
        help=(
            "Which detector_type to render the calibration table for. "
            "Precision table always covers all detectors."
        ),
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help=(
            "Optional output directory. When set, writes "
            "cc_eval_precision_<ts>.csv + cc_eval_calibration_<ts>.csv "
            "alongside the printed tables."
        ),
    ),
) -> None:
    valid_detectors = {"contradiction", "definition_inconsistency", "multi_party"}
    if detector not in valid_detectors:
        raise typer.BadParameter(
            f"--detector must be one of {sorted(valid_detectors)}; got {detector!r}"
        )
    cfg = _load_config(config)
    store = _open_store(cfg)
    try:
        rows = list(iter_eval_rows(store))
    finally:
        store.close()
    precisions = compute_detector_precision(rows)
    calibration = compute_calibration(rows, detector_type=detector, bins=DEFAULT_CALIBRATION_BINS)
    typer.echo("Per-detector precision (excludes 'dismissed' from denominator)")
    typer.echo(format_precision_table(precisions))
    typer.echo("")
    typer.echo(format_calibration_table(calibration, detector_type=detector))
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        precision_path = out / eval_filename("precision")
        calibration_path = out / eval_filename("calibration")
        write_precision_csv(precisions, precision_path)
        bins_by_detector = {
            p.detector_type: compute_calibration(
                rows, detector_type=p.detector_type, bins=DEFAULT_CALIBRATION_BINS
            )
            for p in precisions
        }
        write_calibration_csv(bins_by_detector, calibration_path)
        typer.echo("")
        typer.echo(f"Wrote {precision_path}")
        typer.echo(f"Wrote {calibration_path}")


@store_app.command("stats", help="Print store statistics.")
def store_stats(
    config: Path = typer.Option(Path("config.yml"), "--config", "-c"),
) -> None:
    cfg = _load_config(config)
    store = _open_store(cfg)
    stats = store.stats()
    store.close()
    for key, value in stats.items():
        typer.echo(f"{key}: {value}")


@app.command(help="Start the FastAPI + HTMX web UI on localhost.")
def serve(
    config: Path = typer.Option(Path("config.yml"), "--config", "-c"),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8000, "--port", help="Bind port."),
    open_browser: bool = typer.Option(
        False, "--open", help="Open the default browser at the bound URL after startup."
    ),
) -> None:
    """Boot uvicorn against ``create_app(config)`` and optionally launch a browser."""
    import threading
    import webbrowser

    import uvicorn

    from consistency_checker.web.app import create_app

    cfg = _load_config(config)
    configure_logging(cfg.log_dir)
    web_app = create_app(cfg)

    if open_browser:
        url = f"http://{host}:{port}"
        # Defer the browser launch so uvicorn has a moment to bind. The
        # daemon flag lets the process exit immediately if the user kills
        # uvicorn within the 0.5 s window.
        timer = threading.Timer(0.5, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()

    uvicorn.run(web_app, host=host, port=port, log_level="info")


@store_app.command(
    "reidentify-orgs",
    help="Backfill or refresh document.org_label / org_reason via the LLM identifier.",
)
def store_reidentify_orgs(
    db: Path = typer.Option(..., "--db", help="Path to the SQLite DB."),
    all_docs: bool = typer.Option(False, "--all", help="Reidentify every document."),
    null_only: bool = typer.Option(
        True,
        "--null-only",
        help="Only documents whose org_label IS NULL. Overridden by --all.",
    ),
    corpus: str | None = typer.Option(
        None,
        "--corpus",
        help="Corpus name. Required (interactive picker on TTY).",
    ),
    config: Path = typer.Option(
        Path("config.yml"),
        "--config",
        "-c",
        help="Config used to construct the extractor; falls back to a minimal default.",
    ),
) -> None:
    cfg = Config.from_yaml(config) if config.exists() else Config(corpus_dir=Path("."))
    store = AssertionStore(db)
    try:
        store.migrate()
        # Resolve --corpus before building the extractor so a missing
        # --corpus surfaces immediately, even when no judge API key is set
        # in the environment (e.g. CI without MOONSHOT_API_KEY).
        provider_for_corpus = _provider_for_corpus(cfg.judge_provider)
        corpus_id = resolve_corpus(
            store, corpus, str(cfg.corpus_dir), provider_for_corpus, allow_create=False
        )
        extractor = make_extractor(cfg)
        walk_all = all_docs
        for doc in store.iter_documents(corpus_id=corpus_id):
            if not walk_all and null_only and doc.org_label is not None:
                continue
            try:
                loaded = load_path(
                    Path(doc.source_path),
                    junk_filter_enabled=cfg.junk_filter_enabled,
                )
            except (FileNotFoundError, ValueError, NotImplementedError, OSError) as exc:
                typer.echo(f"{doc.doc_id}: SKIP ({exc})")
                continue
            res = extractor.identify_org(title=doc.title, text=loaded.text)
            store.update_org_label(doc.doc_id, res.label, res.reason)
            typer.echo(f"{doc.doc_id}: {res.reason} -> {res.label!r}")
    finally:
        store.close()


@store_app.command("rebuild-index", help="Regenerate the FAISS index from SQLite.")
def store_rebuild_index(
    config: Path = typer.Option(Path("config.yml"), "--config", "-c"),
) -> None:
    cfg = _load_config(config)
    configure_logging(cfg.log_dir)
    # Wipe the existing index by writing to a fresh path then atomically rename.
    embedder = make_embedder(cfg)
    if cfg.faiss_path.exists():
        cfg.faiss_path.unlink()
    id_map_path = cfg.faiss_path.with_suffix(".idmap.json")
    if id_map_path.exists():
        id_map_path.unlink()

    store = _open_store(cfg)
    faiss_store = _open_faiss(cfg, dim=embedder.dim)
    n = rebuild_faiss(store=store, faiss_store=faiss_store, embedder=embedder)
    store.close()
    typer.echo(f"Rebuilt FAISS index from {n} assertions.")


# --- corpus subcommands -------------------------------------------------------


@corpus_app.command("list")
def corpus_list(
    db: Path = typer.Option(..., "--db", help="Path to the SQLite DB."),
) -> None:
    """List corpora with document counts and paths."""
    from consistency_checker.index.assertion_store import AssertionStore

    store = AssertionStore(db)
    store.migrate()
    try:
        rows = store.list_corpora()
        if not rows:
            typer.echo("No corpora.")
            return
        for c in rows:
            stats = store.stats(corpus_id=c.corpus_id)
            typer.echo(f"{c.corpus_name:30s}  {stats['documents']:>5d} docs   ({c.corpus_path})")
    finally:
        store.close()


@corpus_app.command("delete")
def corpus_delete(
    name: str,
    db: Path = typer.Option(..., "--db"),
    yes: bool = typer.Option(
        False,
        "--yes-i-mean-it",
        help="Required confirmation; cascades to all docs/assertions.",
    ),
) -> None:
    """Delete a corpus and all its documents/assertions/findings."""
    from consistency_checker.index.assertion_store import AssertionStore

    if not yes:
        raise typer.BadParameter("Refusing to delete without --yes-i-mean-it")
    store = AssertionStore(db)
    store.migrate()
    try:
        match = next((c for c in store.list_corpora() if c.corpus_name == name), None)
        if not match:
            available = ", ".join(c.corpus_name for c in store.list_corpora()) or "<none>"
            raise typer.BadParameter(f"No corpus named {name!r} (available: {available})")
        store.delete_corpus(match.corpus_id)
        typer.echo(f"Deleted corpus {name!r}.")
    finally:
        store.close()


@corpus_app.command("reassign")
def corpus_reassign(
    db: Path = typer.Option(..., "--db"),
    src: str = typer.Option(..., "--from", help="Source corpus name."),
    dst: str = typer.Option(..., "--to", help="Destination corpus name (created if absent)."),
    where: str = typer.Option(
        "",
        "--where",
        help=("Safe-listed WHERE clause filter on documents (e.g. \"org_label LIKE 'ATKINS%'\")."),
    ),
) -> None:
    """Move documents from one corpus to another.

    The --where clause is restricted to a safe subset (column=literal /
    column LIKE pattern, joined by AND/OR) to prevent SQL injection.
    """
    import re

    from consistency_checker.index.assertion_store import AssertionStore

    # Safe-list: alphanumeric column names, _, single-quoted string literals
    # with % wildcards, =, LIKE, AND, OR. Reject anything else.
    if where and not re.fullmatch(
        r"\s*[A-Za-z_][A-Za-z0-9_]*\s*(?:=|LIKE)\s*'[^']*'"
        r"(?:\s+(?:AND|OR)\s+[A-Za-z_][A-Za-z0-9_]*\s*(?:=|LIKE)\s*'[^']*')*\s*",
        where,
    ):
        raise typer.BadParameter(
            "--where allows only column=literal or column LIKE 'pattern' "
            "joined by AND/OR (single-quoted string literals)."
        )

    store = AssertionStore(db)
    store.migrate()
    try:
        src_match = next((c for c in store.list_corpora() if c.corpus_name == src), None)
        if src_match is None:
            available = ", ".join(c.corpus_name for c in store.list_corpora()) or "<none>"
            raise typer.BadParameter(f"--from corpus {src!r} not found (available: {available})")
        dst_id = store.get_or_create_corpus(dst, "(reassigned)", "moonshot")
        sql = "UPDATE documents SET corpus_id = ? WHERE corpus_id = ?"
        params: list[str] = [dst_id, src_match.corpus_id]
        if where:
            sql += f" AND ({where})"
        with store._conn:
            cur = store._conn.execute(sql, params)
            n = cur.rowcount
        word = "document" if n == 1 else "documents"
        typer.echo(f"Moved {n} {word} from {src} to {dst}.")
    finally:
        store.close()


if __name__ == "__main__":
    app()
