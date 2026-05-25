from typer.testing import CliRunner


def test_reidentify_orgs_null_only_updates_null_rows(monkeypatch, tmp_path):
    from consistency_checker.cli.main import app
    from consistency_checker.extract.atomic_facts import (
        FixtureExtractor,
        OrgIdentification,
    )
    from consistency_checker.extract.schema import Document
    from consistency_checker.index.assertion_store import AssertionStore

    # Write a corpus file (the subcommand reads source_path).
    doc_path = tmp_path / "doc1.txt"
    doc_path.write_text("anything", encoding="utf-8")

    db = tmp_path / "t.db"
    store = AssertionStore(db)
    store.migrate()
    _cid = store.get_or_create_corpus("test", "/test", "moonshot")
    store.add_document(Document(doc_id="d1", source_path=str(doc_path)), corpus_id=_cid)
    store.add_document(
        Document(
            doc_id="d2",
            source_path=str(doc_path),
            org_label="Existing",
            org_reason="org_found",
        ),
        corpus_id=_cid,
    )
    store.close()

    fx = FixtureExtractor(
        {},
        org_fixtures={(None, "anything"): OrgIdentification("Filled-In", "org_found")},
    )
    # Monkeypatch make_extractor so the subcommand uses our fixture
    import consistency_checker.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "make_extractor", lambda cfg: fx)
    import consistency_checker.cli.main as cli_main

    monkeypatch.setattr(cli_main, "make_extractor", lambda cfg: fx, raising=False)

    runner = CliRunner()
    res = runner.invoke(app, ["store", "reidentify-orgs", "--db", str(db), "--null-only"])
    assert res.exit_code == 0, res.output

    store = AssertionStore(db)
    assert store.get_document("d1").org_label == "Filled-In"
    assert store.get_document("d2").org_label == "Existing"  # untouched
    store.close()


def test_reidentify_orgs_all_overwrites_existing(monkeypatch, tmp_path):
    from consistency_checker.cli.main import app
    from consistency_checker.extract.atomic_facts import (
        FixtureExtractor,
        OrgIdentification,
    )
    from consistency_checker.extract.schema import Document
    from consistency_checker.index.assertion_store import AssertionStore

    doc_path = tmp_path / "doc1.txt"
    doc_path.write_text("anything", encoding="utf-8")

    db = tmp_path / "t.db"
    store = AssertionStore(db)
    store.migrate()
    _cid = store.get_or_create_corpus("test", "/test", "moonshot")
    store.add_document(
        Document(
            doc_id="d2",
            source_path=str(doc_path),
            org_label="Existing",
            org_reason="org_found",
        ),
        corpus_id=_cid,
    )
    store.close()

    fx = FixtureExtractor(
        {},
        org_fixtures={(None, "anything"): OrgIdentification("New Value", "org_found")},
    )
    import consistency_checker.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "make_extractor", lambda cfg: fx)
    import consistency_checker.cli.main as cli_main

    monkeypatch.setattr(cli_main, "make_extractor", lambda cfg: fx, raising=False)

    runner = CliRunner()
    res = runner.invoke(app, ["store", "reidentify-orgs", "--db", str(db), "--all"])
    assert res.exit_code == 0, res.output

    store = AssertionStore(db)
    assert store.get_document("d2").org_label == "New Value"
    store.close()
