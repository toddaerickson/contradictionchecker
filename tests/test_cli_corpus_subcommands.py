from typer.testing import CliRunner

from tests.conftest import strip_ansi


def test_corpus_list_shows_each_corpus_with_doc_count(tmp_path):
    from consistency_checker.cli.main import app
    from consistency_checker.extract.schema import Document
    from consistency_checker.index.assertion_store import AssertionStore

    db = tmp_path / "t.db"
    store = AssertionStore(db)
    store.migrate()
    a = store.get_or_create_corpus("alpha", "/a", "moonshot")
    store.add_document(Document(doc_id="d1", source_path="/x"), corpus_id=a)
    store.close()

    runner = CliRunner()
    res = runner.invoke(app, ["corpus", "list", "--db", str(db)])
    assert res.exit_code == 0, res.output
    assert "alpha" in res.output
    assert "1" in res.output  # doc count


def test_corpus_list_handles_empty_store(tmp_path):
    from consistency_checker.cli.main import app
    from consistency_checker.index.assertion_store import AssertionStore

    db = tmp_path / "t.db"
    store = AssertionStore(db)
    store.migrate()
    store.close()
    runner = CliRunner()
    res = runner.invoke(app, ["corpus", "list", "--db", str(db)])
    assert res.exit_code == 0
    # Either prints nothing or a "no corpora" message — accept either.


def test_corpus_delete_requires_yes_i_mean_it(tmp_path):
    from consistency_checker.cli.main import app
    from consistency_checker.index.assertion_store import AssertionStore

    db = tmp_path / "t.db"
    store = AssertionStore(db)
    store.migrate()
    store.get_or_create_corpus("alpha", "/a", "moonshot")
    store.close()

    runner = CliRunner()
    res = runner.invoke(app, ["corpus", "delete", "alpha", "--db", str(db)])
    assert res.exit_code != 0
    out = strip_ansi(res.output + str(res.exception or ""))
    assert "--yes-i-mean-it" in out

    res2 = runner.invoke(app, ["corpus", "delete", "alpha", "--yes-i-mean-it", "--db", str(db)])
    assert res2.exit_code == 0, res2.output
    store = AssertionStore(db)
    assert store.list_corpora() == []
    store.close()


def test_corpus_delete_errors_on_unknown_corpus(tmp_path):
    from consistency_checker.cli.main import app
    from consistency_checker.index.assertion_store import AssertionStore

    db = tmp_path / "t.db"
    store = AssertionStore(db)
    store.migrate()
    store.get_or_create_corpus("alpha", "/a", "moonshot")
    store.close()
    runner = CliRunner()
    res = runner.invoke(app, ["corpus", "delete", "nope", "--yes-i-mean-it", "--db", str(db)])
    assert res.exit_code != 0
    assert "nope" in (res.output + str(res.exception or ""))


def test_corpus_reassign_moves_matching_rows(tmp_path):
    from consistency_checker.cli.main import app
    from consistency_checker.extract.schema import Document
    from consistency_checker.index.assertion_store import AssertionStore

    db = tmp_path / "t.db"
    store = AssertionStore(db)
    store.migrate()
    legacy = store.get_or_create_corpus("legacy", "/legacy", "moonshot")
    store.add_document(
        Document(
            doc_id="d1", source_path="/x", org_label="ATKINS NUTRITIONALS", org_reason="org_found"
        ),
        corpus_id=legacy,
    )
    store.add_document(
        Document(
            doc_id="d2", source_path="/y", org_label="Lockhart Springs", org_reason="org_found"
        ),
        corpus_id=legacy,
    )
    store.close()

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "corpus",
            "reassign",
            "--db",
            str(db),
            "--from",
            "legacy",
            "--to",
            "atkins",
            "--where",
            "org_label LIKE 'ATKINS%'",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Moved 1 document" in res.output

    store = AssertionStore(db)
    atkins_id = next(c.corpus_id for c in store.list_corpora() if c.corpus_name == "atkins")
    d1_cid = store._conn.execute("SELECT corpus_id FROM documents WHERE doc_id='d1'").fetchone()[0]
    d2_cid = store._conn.execute("SELECT corpus_id FROM documents WHERE doc_id='d2'").fetchone()[0]
    assert d1_cid == atkins_id
    assert d2_cid == legacy
    store.close()


def test_corpus_reassign_rejects_unsafe_where_clause(tmp_path):
    from consistency_checker.cli.main import app
    from consistency_checker.index.assertion_store import AssertionStore

    db = tmp_path / "t.db"
    store = AssertionStore(db)
    store.migrate()
    store.get_or_create_corpus("legacy", "/l", "moonshot")
    store.close()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "corpus",
            "reassign",
            "--db",
            str(db),
            "--from",
            "legacy",
            "--to",
            "x",
            "--where",
            "1=1; DROP TABLE documents",
        ],
    )
    assert res.exit_code != 0
    out = res.output + str(res.exception or "")
    assert "--where" in out or "where" in out.lower()


def test_corpus_reassign_errors_on_unknown_source(tmp_path):
    from consistency_checker.cli.main import app
    from consistency_checker.index.assertion_store import AssertionStore

    db = tmp_path / "t.db"
    store = AssertionStore(db)
    store.migrate()
    store.close()
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["corpus", "reassign", "--db", str(db), "--from", "nope", "--to", "atkins"],
    )
    assert res.exit_code != 0
