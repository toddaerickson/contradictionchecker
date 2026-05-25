from pathlib import Path
from unittest.mock import MagicMock

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


def _stub_tool_use_response(label: str | None, reason: str) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "identify_org"
    block.input = {"label": label, "reason": reason}
    resp = MagicMock()
    resp.content = [block]
    return resp


def test_anthropic_identify_org_returns_parsed_label():
    from consistency_checker.extract.atomic_facts import AnthropicExtractor

    client = MagicMock()
    client.messages.create.return_value = _stub_tool_use_response(
        "Acme Foundation, Inc.", "org_found"
    )
    ex = AnthropicExtractor(client=client, model="claude-sonnet-4-6")
    res = ex.identify_org(title="Bylaws", text="Bylaws of Acme Foundation, Inc.")
    assert res.label == "Acme Foundation, Inc."
    assert res.reason == "org_found"
    call_kwargs = client.messages.create.call_args.kwargs
    user_msg = call_kwargs["messages"][0]["content"]
    assert "BEGIN DOCUMENT" in user_msg and "END DOCUMENT" in user_msg


def test_anthropic_identify_org_returns_no_org_on_null_label():
    from consistency_checker.extract.atomic_facts import AnthropicExtractor

    client = MagicMock()
    client.messages.create.return_value = _stub_tool_use_response(None, "no_org")
    ex = AnthropicExtractor(client=client, model="claude-sonnet-4-6")
    res = ex.identify_org(title=None, text="some joint venture text")
    assert res.label is None
    assert res.reason == "no_org"


def test_anthropic_identify_org_returns_llm_error_on_exception():
    from consistency_checker.extract.atomic_facts import AnthropicExtractor

    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("network down")
    ex = AnthropicExtractor(client=client, model="claude-sonnet-4-6")
    res = ex.identify_org(title="x", text="y")
    assert res.label is None
    assert res.reason == "llm_error"


def test_anthropic_identify_org_marks_truncated_when_input_long():
    from consistency_checker.extract.atomic_facts import ORG_PROMPT_CHAR_CAP, AnthropicExtractor

    client = MagicMock()
    client.messages.create.return_value = _stub_tool_use_response(None, "no_org")
    ex = AnthropicExtractor(client=client, model="claude-sonnet-4-6")
    long_text = "a" * (ORG_PROMPT_CHAR_CAP + 500)
    res = ex.identify_org(title=None, text=long_text)
    assert res.label is None
    assert res.reason == "truncated"


def _stub_moonshot_response(label: str | None, reason: str) -> MagicMock:
    resp = MagicMock()
    choice = MagicMock()
    choice.message.parsed = MagicMock()
    choice.message.parsed.label = label
    choice.message.parsed.reason = reason
    resp.choices = [choice]
    return resp


def test_moonshot_identify_org_returns_parsed_label():
    from consistency_checker.extract.atomic_facts import MoonshotExtractor

    client = MagicMock()
    client.beta.chat.completions.parse.return_value = _stub_moonshot_response(
        "Beta Trust", "org_found"
    )
    ex = MoonshotExtractor(client=client, model="kimi-k2.6")
    res = ex.identify_org(title="Trust Indenture", text="Beta Trust hereby ...")
    assert res.label == "Beta Trust"
    assert res.reason == "org_found"


def test_moonshot_identify_org_truncated_when_long_and_no_org():
    from consistency_checker.extract.atomic_facts import ORG_PROMPT_CHAR_CAP, MoonshotExtractor

    client = MagicMock()
    client.beta.chat.completions.parse.return_value = _stub_moonshot_response(None, "no_org")
    ex = MoonshotExtractor(client=client, model="kimi-k2.6")
    res = ex.identify_org(title=None, text="a" * (ORG_PROMPT_CHAR_CAP + 1))
    assert res.reason == "truncated"


def test_moonshot_identify_org_llm_error_on_exception():
    from consistency_checker.extract.atomic_facts import MoonshotExtractor

    client = MagicMock()
    client.beta.chat.completions.parse.side_effect = RuntimeError("boom")
    ex = MoonshotExtractor(client=client, model="kimi-k2.6")
    res = ex.identify_org(title="x", text="y")
    assert res.reason == "llm_error"
