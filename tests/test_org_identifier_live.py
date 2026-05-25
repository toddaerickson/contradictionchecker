import os

import pytest

pytestmark = pytest.mark.live


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no anthropic key")
def test_anthropic_identify_org_live_recognizes_named_org():
    import anthropic

    from consistency_checker.extract.atomic_facts import AnthropicExtractor

    client = anthropic.Anthropic()
    ex = AnthropicExtractor(client=client, model="claude-sonnet-4-6")
    text = "BYLAWS OF ACME FOUNDATION, INC.\n\nArticle I. The Corporation. ..."
    res = ex.identify_org(title="Bylaws", text=text)
    assert res.reason == "org_found"
    assert "Acme" in (res.label or "")


@pytest.mark.skipif(not os.environ.get("MOONSHOT_API_KEY"), reason="no moonshot key")
def test_moonshot_identify_org_live_recognizes_named_org():
    import os as _os

    from openai import OpenAI

    from consistency_checker.extract.atomic_facts import MoonshotExtractor

    client = OpenAI(
        api_key=_os.environ["MOONSHOT_API_KEY"],
        base_url="https://api.moonshot.ai/v1",
    )
    ex = MoonshotExtractor(client=client, model="kimi-k2.6")
    text = "BYLAWS OF ACME FOUNDATION, INC.\n\nArticle I. ..."
    res = ex.identify_org(title="Bylaws", text=text)
    assert res.reason == "org_found"
    assert "Acme" in (res.label or "")
