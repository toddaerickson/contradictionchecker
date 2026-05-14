"""Typer-based ``consistency-check`` CLI.

Thin wrapper around :mod:`consistency_checker.pipeline`. Subcommands:

- ``ingest <corpus_dir>`` — load + chunk + extract + embed.
- ``check`` — Stage A + Stage B + audit.
- ``report`` — render markdown from the latest run.
- ``export csv|jsonl`` — emit assertions for downstream tooling.
- ``store stats|rebuild-index`` — store maintenance.
"""

from __future__ import annotations

from pathlib import Path

import typer

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
    ingest as run_ingest,
)
from consistency_checker.pipeline import (
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

    nli = TransformerNliChecker(model_name=cfg.nli_model)
    judge = make_judge(cfg)
    multi_party = make_multi_party_judge(cfg) if cfg.enable_multi_party else None

    result = run_check(
        cfg,
        store=store,
        faiss_store=faiss_store,
        nli_checker=nli,
        judge=judge,
        audit_logger=audit_logger,
        multi_party_judge=multi_party,
    )
    store.close()
    deep_suffix = (
        f" / {result.n_triangles_judged} triangles / {result.n_multi_party_findings} multi-party"
        if cfg.enable_multi_party
        else ""
    )
    typer.echo(
        f"Run {result.run_id} — "
        f"{result.n_pairs_gated} gated / {result.n_pairs_judged} judged / "
        f"{result.n_findings} contradictions{deep_suffix}"
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
