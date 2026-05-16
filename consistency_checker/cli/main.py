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
from consistency_checker.config import Config
from consistency_checker.index.assertion_store import AssertionStore
from consistency_checker.index.faiss_store import FaissStore
from consistency_checker.logging_setup import configure as configure_logging
from consistency_checker.pipeline import (
    check as run_check,
)
from consistency_checker.pipeline import (
    estimate_cost as run_estimate_cost,
)
from consistency_checker.pipeline import (
    ingest as run_ingest,
)
from consistency_checker.pipeline import (
    make_definition_checker,
    make_embedder,
    make_extractor,
    make_judge,
    make_multi_party_judge,
    rebuild_faiss,
)

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Consistency checker CLI.")
store_app = typer.Typer(help="Assertion-store maintenance commands.")
app.add_typer(store_app, name="store")


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


def _warn_if_model_download_needed(model_name: str) -> None:
    """Print a one-line warning if the HF model isn't in the local cache."""
    try:
        from huggingface_hub import try_to_load_from_cache

        result = try_to_load_from_cache(model_name, "config.json")
        if result is None:
            typer.echo(
                f"Note: first run will download ~800 MB ({model_name}). "
                "This may take a few minutes — the tool is not hung."
            )
    except Exception:
        pass  # huggingface_hub not available or lookup failed — skip silently


# --- ingest -----------------------------------------------------------------


@app.command(help="Load documents, chunk, extract atomic facts, and embed.")
def ingest(
    corpus_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    config: Path = typer.Option(Path("config.yml"), "--config", "-c"),
) -> None:
    cfg = _load_config(config)
    cfg = cfg.model_copy(update={"corpus_dir": corpus_dir})
    configure_logging(cfg.log_dir)
    extractor = make_extractor(cfg)
    embedder = make_embedder(cfg)
    store = _open_store(cfg)
    faiss_store = _open_faiss(cfg, dim=embedder.dim)
    result = run_ingest(
        cfg, store=store, faiss_store=faiss_store, extractor=extractor, embedder=embedder
    )
    store.close()
    typer.echo(
        f"Ingested {result.n_documents} documents, {result.n_chunks} chunks, "
        f"{result.n_assertions} assertions, {result.n_embedded} newly embedded."
    )


# --- check ------------------------------------------------------------------


@app.command(help="Run the Stage A + Stage B contradiction scan against the indexed corpus.")
def check(
    config: Path = typer.Option(Path("config.yml"), "--config", "-c"),
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
) -> None:
    cfg = _load_config(config)
    if deep:
        cfg = cfg.model_copy(update={"enable_multi_party": True})
    configure_logging(cfg.log_dir)
    embedder = make_embedder(cfg)
    store = _open_store(cfg)
    faiss_store = _open_faiss(cfg, dim=embedder.dim)
    audit_logger = AuditLogger(store)

    # Lazy NLI import — keeps `ingest` fast when the model is not yet cached.
    from consistency_checker.check.nli_checker import TransformerNliChecker

    _warn_if_model_download_needed(cfg.nli_model or TransformerNliChecker.DEFAULT_MODEL)
    nli = TransformerNliChecker(model_name=cfg.nli_model)
    judge = make_judge(cfg)
    multi_party = make_multi_party_judge(cfg) if cfg.enable_multi_party else None
    definition_checker = None if no_definitions else make_definition_checker(cfg)

    run_id = audit_logger.begin_run(
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
        }
    )
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
    )
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
    typer.echo(
        f"Run {result.run_id} — "
        f"{result.n_pairs_gated} gated / {result.n_pairs_judged} judged / "
        f"{result.n_findings} contradictions{deep_suffix}{def_suffix}"
    )


# --- estimate-cost ----------------------------------------------------------


@app.command(
    "estimate-cost",
    help="Estimate API spend for a check run without making any LLM or NLI calls.",
)
def estimate_cost(
    config: Path = typer.Option(Path("config.yml"), "--config", "-c"),
    per_call_low: float = typer.Option(
        0.003,
        "--per-call-low",
        help="Low end of per-judge-call cost in USD; override to match your model.",
    ),
    per_call_high: float = typer.Option(
        0.010,
        "--per-call-high",
        help="High end of per-judge-call cost in USD; override to match your model.",
    ),
) -> None:
    cfg = _load_config(config)
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
    est = run_estimate_cost(
        cfg,
        store=store,
        faiss_store=faiss_store,
        per_call_low=per_call_low,
        per_call_high=per_call_high,
    )
    store.close()
    typer.echo(
        "Run cost estimate\n"
        f"  Assertions in store:          {est.n_assertions}\n"
        f"  Stage A - gated pairs:        {est.n_candidate_pairs}\n"
        f"  Definition pairs to judge:    {est.n_definition_pairs}\n"
        f"  Total judge calls (ceiling):  {est.judge_calls_ceiling}\n"
        "\n"
        f"  Estimated API cost:  ${est.est_cost_low:.2f} to ${est.est_cost_high:.2f}\n"
        f"  (assumes ${est.per_call_low:.3f} to ${est.per_call_high:.3f} per judge call;\n"
        "   varies by model + prompt size. This is a CEILING - many gate-pass\n"
        "   pairs are filtered by NLI before reaching the judge, so real spend\n"
        "   is usually 30-70% lower.)"
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

    text = render_report(store, audit_logger, run_id=target_run, min_confidence=min_confidence)
    if out is None:
        run = audit_logger.get_run(target_run)
        started = run.started_at if run is not None else None
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
    config: Path = typer.Option(Path("config.yml"), "--config", "-c"),
) -> None:
    cfg = _load_config(config)
    store = _open_store(cfg)
    if fmt == "csv":
        if out is None:
            out = cfg.data_dir / "reports" / export_csv_filename()
        store.export_csv(out)
    elif fmt == "jsonl":
        if out is None:
            out = cfg.data_dir / "reports" / export_jsonl_filename()
        store.export_jsonl(out)
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


if __name__ == "__main__":
    app()
