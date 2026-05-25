"""TTY-aware corpus resolution.

Why: --corpus is required on every mutating CLI command (per spec §3).
When the operator forgets to pass it on a TTY we interactively prompt;
in a scripted/piped environment we fail fast with the available list.
Lives in its own module so the policy is unit-testable without typer
machinery.
"""

from __future__ import annotations

import sys

import typer

from consistency_checker.index.assertion_store import AssertionStore


class CorpusRequiredError(typer.BadParameter):
    pass


def resolve_corpus(
    store: AssertionStore,
    name: str | None,
    path: str | None,
    judge_provider: str,
    *,
    isatty: bool | None = None,
) -> str:
    """Return a corpus_id for ``name``. Creates the corpus if name is new.

    If name is None and stdin is a TTY → interactive picker.
    If name is None and not a TTY    → CorpusRequiredError listing available corpora.
    """
    if isatty is None:
        isatty = sys.stdin.isatty()

    if name is not None:
        return store.get_or_create_corpus(name, path or "(unset)", judge_provider)

    existing = store.list_corpora()
    if not isatty:
        names = ", ".join(c.corpus_name for c in existing) or "<none>"
        raise CorpusRequiredError(f"--corpus is required (available: {names})")

    if not existing:
        new_name = typer.prompt("No corpora yet. Name a new one")
        return store.get_or_create_corpus(new_name, path or "(unset)", judge_provider)

    typer.echo("Available corpora:")
    for i, c in enumerate(existing, 1):
        typer.echo(f"  {i}. {c.corpus_name:30s}  ({c.corpus_path})")
    typer.echo("  [new]  create a new corpus")
    choice = typer.prompt("Pick one").strip()
    if choice == "new":
        new_name = typer.prompt("Name")
        return store.get_or_create_corpus(new_name, path or "(unset)", judge_provider)
    if choice.isdigit() and 1 <= int(choice) <= len(existing):
        return existing[int(choice) - 1].corpus_id
    # One re-prompt then give up.
    choice = typer.prompt("Invalid; pick a number from the list, or [new]").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(existing):
        return existing[int(choice) - 1].corpus_id
    if choice == "new":
        new_name = typer.prompt("Name")
        return store.get_or_create_corpus(new_name, path or "(unset)", judge_provider)
    raise CorpusRequiredError("No valid corpus selection.")
