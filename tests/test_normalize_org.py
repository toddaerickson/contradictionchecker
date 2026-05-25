from consistency_checker.check.definition_terms import normalize_org


def test_casefolds_and_collapses_whitespace():
    assert normalize_org("  Acme   Foundation  ") == "acme foundation"


def test_strips_leading_the():
    assert normalize_org("The Acme Foundation") == "acme foundation"


def test_does_not_strip_internal_the():
    assert normalize_org("Friends of the Acme") == "friends of the acme"


def test_strips_trailing_legal_suffixes():
    for suffix in ["Inc", "LLC", "L.P.", "LP", "Corporation", "Corp",
                   "Company", "Co", "Ltd", "Limited"]:
        full = f"Acme {suffix}"
        got = normalize_org(full)
        assert got == "acme", f"{full!r} -> {got!r}"


def test_collapses_punctuation():
    assert normalize_org("Acme, Inc.") == "acme"


def test_distinct_orgs_with_same_token_do_not_merge():
    assert normalize_org("Acme Trust") != normalize_org("Acme Foundation")
    assert normalize_org("Acme Corp") != normalize_org("Acme Trust")


def test_suffix_alone_does_not_reduce_to_empty():
    assert normalize_org("Inc") == "inc"
    assert normalize_org("Trust") == "trust"
    assert normalize_org("Foundation") == "foundation"


def test_idempotent():
    for raw in ["The Acme Foundation, Inc.", "  beta TRUST  ", "Gamma LLC"]:
        once = normalize_org(raw)
        assert normalize_org(once) == once


def test_empty_and_whitespace_only():
    assert normalize_org("") == ""
    assert normalize_org("   ") == ""


def test_lp_with_dots_collapses_to_single_word():
    assert normalize_org("Acme L.P.") == "acme"
    assert normalize_org("Acme LP") == "acme"
