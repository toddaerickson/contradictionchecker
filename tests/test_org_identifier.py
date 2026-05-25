from pathlib import Path

from consistency_checker.extract.atomic_facts import (
    FixtureExtractor,
    OrgIdentification,
)


def test_orgidentification_carries_reason():
    res = OrgIdentification(label="Acme", reason="org_found")
    assert res.label == "Acme"
    assert res.reason == "org_found"


def test_fixture_identify_returns_canned_value():
    fx = FixtureExtractor(
        {"chunk1": []},
        org_fixtures={
            ("the title", "body starts here"): OrgIdentification(
                label="Acme Foundation, Inc.",
                reason="org_found",
            ),
        },
    )
    res = fx.identify_org(title="the title", text="body starts here is the rest")
    assert res.label == "Acme Foundation, Inc."
    assert res.reason == "org_found"


def test_fixture_identify_returns_no_org_when_unkeyed():
    fx = FixtureExtractor({"chunk1": []})
    res = fx.identify_org(title="anything", text="anything")
    assert res.label is None
    assert res.reason == "no_org"


PROMPTS_DIR = Path("consistency_checker/extract/prompts")


def test_org_identifier_system_prompt_exists_and_constrains_output():
    p = PROMPTS_DIR / "org_identifier_system.txt"
    assert p.exists()
    body = p.read_text(encoding="utf-8")
    assert "primary" in body.lower()
    assert "issuing" in body.lower() or "issued" in body.lower()
    assert "no_org" in body
    assert "BEGIN DOCUMENT" in body or "begin document" in body.lower()


def test_org_identifier_user_prompt_has_required_placeholders():
    p = PROMPTS_DIR / "org_identifier_user.txt"
    assert p.exists()
    body = p.read_text(encoding="utf-8")
    assert "{title}" in body
    assert "{text_prefix}" in body
