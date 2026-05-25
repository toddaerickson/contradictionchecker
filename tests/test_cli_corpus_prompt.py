import pytest


def test_resolve_corpus_returns_passed_name_if_given(tmp_path):
    from consistency_checker.cli.corpus_prompt import resolve_corpus
    from consistency_checker.index.assertion_store import AssertionStore

    store = AssertionStore(tmp_path / "t.db")
    store.migrate()
    store.get_or_create_corpus("atkins", "/atkins", "moonshot")
    cid = resolve_corpus(store, "atkins", "/atkins", "moonshot", isatty=False)
    assert cid is not None
    store.close()


def test_resolve_corpus_creates_new_corpus_if_name_unknown(tmp_path):
    from consistency_checker.cli.corpus_prompt import resolve_corpus
    from consistency_checker.index.assertion_store import AssertionStore

    store = AssertionStore(tmp_path / "t.db")
    store.migrate()
    cid = resolve_corpus(store, "newcorpus", "/newpath", "moonshot", isatty=False)
    assert cid is not None
    assert any(c.corpus_name == "newcorpus" for c in store.list_corpora())
    store.close()


def test_resolve_corpus_raises_when_missing_and_non_tty(tmp_path):
    from consistency_checker.cli.corpus_prompt import CorpusRequiredError, resolve_corpus
    from consistency_checker.index.assertion_store import AssertionStore

    store = AssertionStore(tmp_path / "t.db")
    store.migrate()
    store.get_or_create_corpus("alpha", "/a", "moonshot")
    store.get_or_create_corpus("beta", "/b", "moonshot")
    with pytest.raises(CorpusRequiredError, match="alpha"):
        resolve_corpus(store, None, None, "moonshot", isatty=False)
    store.close()


def test_resolve_corpus_interactive_picks_from_list(tmp_path, monkeypatch):
    from consistency_checker.cli.corpus_prompt import resolve_corpus
    from consistency_checker.index.assertion_store import AssertionStore

    store = AssertionStore(tmp_path / "t.db")
    store.migrate()
    store.get_or_create_corpus("alpha", "/a", "moonshot")
    store.get_or_create_corpus("beta", "/b", "moonshot")
    # Simulate user typing "2" → beta
    monkeypatch.setattr("typer.prompt", lambda *a, **k: "2")
    cid = resolve_corpus(store, None, None, "moonshot", isatty=True)
    beta_id = next(c.corpus_id for c in store.list_corpora() if c.corpus_name == "beta")
    assert cid == beta_id
    store.close()


def test_resolve_corpus_interactive_new_creates(tmp_path, monkeypatch):
    from consistency_checker.cli.corpus_prompt import resolve_corpus
    from consistency_checker.index.assertion_store import AssertionStore

    store = AssertionStore(tmp_path / "t.db")
    store.migrate()
    store.get_or_create_corpus("alpha", "/a", "moonshot")
    # Simulate user typing "new" then "atkins"
    answers = iter(["new", "atkins"])
    monkeypatch.setattr("typer.prompt", lambda *a, **k: next(answers))
    resolve_corpus(store, None, None, "moonshot", isatty=True)
    assert any(c.corpus_name == "atkins" for c in store.list_corpora())
    store.close()


def test_resolve_corpus_interactive_empty_store_prompts_for_name(tmp_path, monkeypatch):
    from consistency_checker.cli.corpus_prompt import resolve_corpus
    from consistency_checker.index.assertion_store import AssertionStore

    store = AssertionStore(tmp_path / "t.db")
    store.migrate()
    monkeypatch.setattr("typer.prompt", lambda *a, **k: "first-corpus")
    resolve_corpus(store, None, "/somepath", "moonshot", isatty=True)
    assert any(c.corpus_name == "first-corpus" for c in store.list_corpora())
    store.close()


def test_resolve_corpus_errors_on_unknown_when_allow_create_false(tmp_path):
    import pytest

    from consistency_checker.cli.corpus_prompt import CorpusRequiredError, resolve_corpus
    from consistency_checker.index.assertion_store import AssertionStore

    store = AssertionStore(tmp_path / "t.db")
    store.migrate()
    store.get_or_create_corpus("atkins", "/atkins", "moonshot")
    with pytest.raises(CorpusRequiredError, match="acme_typo"):
        resolve_corpus(store, "acme_typo", "/p", "moonshot", isatty=False, allow_create=False)
    store.close()


def test_resolve_corpus_returns_existing_when_allow_create_false_and_name_matches(tmp_path):
    from consistency_checker.cli.corpus_prompt import resolve_corpus
    from consistency_checker.index.assertion_store import AssertionStore

    store = AssertionStore(tmp_path / "t.db")
    store.migrate()
    expected = store.get_or_create_corpus("atkins", "/atkins", "moonshot")
    got = resolve_corpus(store, "atkins", "/atkins", "moonshot", isatty=False, allow_create=False)
    assert got == expected
    store.close()
